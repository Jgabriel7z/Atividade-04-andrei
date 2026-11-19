"""
eval/eval_versoes.py
────────────────────
Mede os critérios da melhoria (Atividade 03, "Como vamos validar") no conjunto
de exemplo: Regulamento Interno, Contrato Modelo e Manual do Colaborador.

  1. Artigos alterados detectados pela comparação          meta 100%
  2. Citações diretas encontradas pela análise de impacto   meta 100%
  3. Precisão das sugestões semânticas (sem falso alarme)  meta ≥ 70%
  +  Documentos de outro tema não recebem sugestão           meta 0 sugestões

O 4º critério (Hit Rate da CF/88 não cair) é medido com `python src/test.py --sample`.

Uso (a partir da raiz do projeto):
    python src/eval/eval_versoes.py
    python src/eval/eval_versoes.py --sweep          # precisão por limiar
    python src/eval/eval_versoes.py --llm            # inclui o resumo da LLM (precisa de GROQ_API_KEY)
    python src/eval/eval_versoes.py --verificar-llm  # a LLM confirma cada sugestão (precisa de GROQ_API_KEY)
    python src/eval/eval_versoes.py --base-real copia.db   # mede também uma base registrada pelo app
    python src/eval/eval_versoes.py --output r.json
"""

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from eval.exemplos import EXEMPLOS, gerar_pdfs          # noqa: E402
from impact import analyze_impact, MIN_DOC_SIMILARITY, MIN_RELEVANCE  # noqa: E402
from ingester import extract_text_from_pdf             # noqa: E402
from retriever import DocumentRetriever                # noqa: E402
from versioning import VersionStore, publish_version, sha256_file  # noqa: E402


def _n(s: str) -> str:
    return " ".join(s.split())


def montar_base(tmp: Path, retriever=None):
    """Cria a base temporária com as 3 versões 1 e a versão 2 do Regulamento."""
    gab = json.loads((EXEMPLOS / "gabarito.json").read_text(encoding="utf-8"))
    pdfs = gerar_pdfs()
    store = VersionStore(tmp / "versions.db")
    retriever = retriever or DocumentRetriever(db_path=tmp / "chroma")

    def pub(chave, arquivo, autor, motivo):
        d = gab["documentos"][chave]
        pdf = pdfs[Path(arquivo).stem]
        return publish_version(
            store, retriever, name=d["nome"], aliases=d["apelidos"],
            text=extract_text_from_pdf(str(pdf)), filename=pdf.name,
            sha256=sha256_file(pdf), author=autor, reason=motivo,
        )

    pub("regulamento", "regulamento_v1.txt", "Assembleia", "versão inicial")
    pub("contrato", "contrato_modelo.txt", "Jurídico", "versão inicial")
    pub("manual", "manual.txt", "RH", "versão inicial")
    r2 = pub("regulamento", "regulamento_v2.txt", "Assembleia", "revisão de jornada, férias e reembolso")
    return gab, store, retriever, r2


def medir(gab, store, retriever, r2, threshold: float, verificar=None,
          doc_threshold: float = MIN_DOC_SIMILARITY) -> dict:
    diff = r2.diff
    alt = gab["alteracoes"]

    # 1. Comparação
    esperado = ([("alterado", k) for k in alt["alterados"]]
                + [("removido", k) for k in alt["removidos"]]
                + [("incluído", k) for k in alt["incluidos"]])
    achado = ({("alterado", c.key) for c in diff.changed}
              | {("removido", a.key) for a in diff.removed}
              | {("incluído", a.key) for a in diff.added})
    c1 = sum(1 for e in esperado if e in achado) / len(esperado)
    extras = sorted(achado - set(esperado))

    # 2 e 3. Impacto
    rep = analyze_impact(store, retriever, r2.document.id, diff, min_relevance=threshold,
                         verificar=verificar, min_doc_similarity=doc_threshold)

    refs = gab["referencias_cruzadas"]
    encontradas, faltando = [], []
    for ref in refs:
        ok = any(c.document_id == ref["documento"] and c.article == ref["artigo"]
                 and _n(ref["trecho"]) in _n(c.snippet) for c in rep.direct)
        (encontradas if ok else faltando).append(ref)
    c2 = len(encontradas) / len(refs)

    usados = set()
    falsos_diretos = []
    for c in rep.direct:
        match = [i for i, ref in enumerate(refs)
                 if c.document_id == ref["documento"] and c.article == ref["artigo"]
                 and _n(ref["trecho"]) in _n(c.snippet) and i not in usados]
        if match:
            usados.add(match[0])
        else:
            falsos_diretos.append({"documento": c.document_id, "artigo": c.article, "trecho": c.snippet})

    # Correta: a frase depende de algum artigo alterado/removido (não é falso alarme).
    # Artigo exato: além disso, o artigo apontado é o mesmo do gabarito.
    por_artigo = {k: [_n(t) for t in ts] for k, ts in gab["trechos_dependentes"].items()
                  if isinstance(ts, list)}
    dependentes = [t for ts in por_artigo.values() for t in ts]
    sug = []
    for s in rep.suggestions:
        correta = any(t in _n(s.text) for t in dependentes)
        exato = any(t in _n(s.text) for t in por_artigo.get(s.article, []))
        sug.append({"documento": s.document_id, "artigo": s.article,
                    "relevancia": round(s.relevance, 4), "tema": round(s.topic, 4),
                    "correta": correta, "artigo_exato": exato, "frase": _n(s.text),
                    "motivo_llm": s.reason})
    recusadas = [{"documento": s.document_id, "artigo": s.article, "frase": _n(s.text),
                  "correta": any(t in _n(s.text) for t in dependentes), "motivo_llm": s.reason}
                 for s in rep.rejected]
    c3 = (sum(x["correta"] for x in sug) / len(sug)) if sug else None
    c3_exato = (sum(x["artigo_exato"] for x in sug) / len(sug)) if sug else None

    return {
        "limiar": threshold,
        "criterio_1_comparacao": c1, "comparacao_extras": [f"{t} {k}" for t, k in extras],
        "criterio_2_citacoes": c2, "citacoes_faltando": faltando, "citacoes_falsas": falsos_diretos,
        "criterio_3_precisao": c3, "precisao_artigo_exato": c3_exato, "sugestoes": sug,
        "recusadas_pela_llm": recusadas,
        "diff_texto": diff.as_text(),
    }


# ── Documentos de outro tema ──────────────────────────────────
# Textos curtos inventados, de assuntos sem relação com o Regulamento da
# cooperativa, com frases que "parecem" artigos dele ("antecedência mínima",
# "prazo de 30 dias", "taxa", "horário"). Reproduzem o caso real do teste ao
# vivo (Regulamento x edital de mestrado x proposta de projeto) sem depender
# daqueles arquivos. Nenhum deles pode receber sugestão.
OUTROS_TEMAS = {
    "Edital de Seleção do Mestrado": (
        "EDITAL DE SELEÇÃO PARA O MESTRADO EM INFORMÁTICA\n"
        "Programa de Pós-Graduação em Informática\n\n"
        "1. DAS VAGAS\n"
        "1.1 São oferecidas 20 (vinte) vagas para o mestrado, distribuídas entre as linhas de pesquisa do programa.\n\n"
        "2. DA INSCRIÇÃO\n"
        "2.1 As inscrições são feitas pelo sistema acadêmico, com envio do currículo e do projeto de pesquisa.\n"
        "2.2 O programa não cobra taxa de inscrição.\n\n"
        "3. DA ENTREVISTA\n"
        "3.1 O candidato deverá estar conectado com antecedência mínima de 10 (dez) minutos do horário agendado para a entrevista.\n\n"
        "4. DOS RECURSOS\n"
        "4.1 Pedidos de reconsideração do resultado devem ser enviados em até 2 (dois) dias úteis após a divulgação.\n\n"
        "5. DA MATRÍCULA\n"
        "5.1 Os aprovados fazem a matrícula em até 30 (trinta) dias após o resultado final, com a entrega do diploma de graduação.\n"
    ),
    "Proposta de Projeto de Mestrado": (
        "PROPOSTA DE PROJETO DE MESTRADO\n"
        "Audiodescrição automática de vídeos com visão computacional\n\n"
        "Resumo\n"
        "Este projeto propõe um sistema que gera audiodescrição de vídeos para pessoas cegas, "
        "combinando detecção de objetos, reconhecimento de ações e modelos de linguagem.\n\n"
        "Metodologia\n"
        "Os vídeos serão anotados por voluntários e usados para ajustar um modelo de legendagem densa. "
        "A avaliação terá testes com usuários cegos e métricas automáticas de qualidade de texto.\n\n"
        "Cronograma\n"
        "Nos primeiros 12 (doze) meses, conclusão dos créditos e revisão sistemática; "
        "nos 12 (doze) meses seguintes, desenvolvimento, experimentos e escrita da dissertação.\n"
    ),
    "Regulamento do Condomínio": (
        "REGULAMENTO DO CONDOMÍNIO RESIDENCIAL JARDIM DAS ACÁCIAS\n\n"
        "Art. 1º Este Regulamento disciplina o uso das áreas comuns do Condomínio.\n\n"
        "Art. 2º O horário de silêncio vai das 22h às 7h, de domingo a quinta-feira.\n\n"
        "Art. 3º A taxa condominial vence no dia 10 (dez) de cada mês; o atraso gera multa de 2% (dois por cento).\n\n"
        "Art. 4º Mudanças devem ser agendadas com a administração com antecedência mínima de 5 (cinco) dias úteis.\n\n"
        "Art. 5º O salão de festas é reservado pelo aplicativo, com taxa de limpeza paga no ato da reserva.\n"
    ),
}


def montar_outros_temas(tmp: Path, retriever=None):
    """Base temporária: Regulamento v1, os documentos de OUTROS_TEMAS e o Regulamento v2."""
    gab = json.loads((EXEMPLOS / "gabarito.json").read_text(encoding="utf-8"))
    reg = gab["documentos"]["regulamento"]
    store = VersionStore(tmp / "versions.db")
    retriever = retriever or DocumentRetriever(db_path=tmp / "chroma")

    def pub(nome, apelidos, texto, arquivo):
        return publish_version(store, retriever, name=nome, aliases=apelidos, text=texto,
                               filename=arquivo,
                               sha256=hashlib.sha256(texto.encode("utf-8")).hexdigest())

    pub(reg["nome"], reg["apelidos"], (EXEMPLOS / reg["v1"]).read_text(encoding="utf-8"), reg["v1"])
    for nome, texto in OUTROS_TEMAS.items():
        pub(nome, [nome], texto, nome.lower().replace(" ", "_") + ".txt")
    r2 = pub(reg["nome"], reg["apelidos"], (EXEMPLOS / reg["v2"]).read_text(encoding="utf-8"), reg["v2"])
    return store, retriever, r2


def medir_outros_temas(store, retriever, r2, verificar=None,
                       doc_threshold: float = MIN_DOC_SIMILARITY) -> dict:
    rep = analyze_impact(store, retriever, r2.document.id, r2.diff, verificar=verificar,
                         min_doc_similarity=doc_threshold)
    return {
        "sugestoes": [{"documento": s.document_id, "artigo": s.article, "frase": _n(s.text)}
                      for s in rep.suggestions],
        "citacoes": len(rep.direct),
        "fora_do_tema": {k: round(v, 3) for k, v in rep.off_topic.items()},
        "recusadas_pela_llm": len(rep.rejected),
    }


def medir_base_real(caminho_db: Path, tmp: Path, retriever=None, verificar=None) -> list:
    """
    Reconstrói, numa pasta temporária, uma base registrada pelo app (cópia do
    SQLite de versões, com o texto de cada versão) e mede o impacto da última
    mudança de cada documento com duas versões ou mais.
    """
    import sqlite3
    from versioning import compare_versions
    con = sqlite3.connect(str(caminho_db))
    con.row_factory = sqlite3.Row
    docs = {r["id"]: r for r in con.execute("SELECT * FROM documents")}
    store = VersionStore(tmp / "versions.db")
    retriever = retriever or DocumentRetriever(db_path=tmp / "chroma")
    for v in con.execute("SELECT * FROM versions ORDER BY id"):
        d = docs[v["document_id"]]
        publish_version(store, retriever, name=d["name"], aliases=[a for a in d["aliases"].split("|") if a],
                        text=v["text"], filename=v["filename"], sha256=v["sha256"], document_id=d["id"])
    out = []
    for doc_id in docs:
        vs = [r["version"] for r in con.execute(
            "SELECT version FROM versions WHERE document_id=? ORDER BY version", (doc_id,))]
        if len(vs) < 2:
            continue
        a, b = store.get_version(doc_id, vs[-2]), store.get_version(doc_id, vs[-1])
        rep = analyze_impact(store, retriever, doc_id, compare_versions(a.text, b.text),
                             verificar=verificar)
        out.append({"documento": doc_id, "versoes": [vs[-2], vs[-1]],
                    "artigos": rep.affected_articles, "citacoes": len(rep.direct),
                    "sugestoes": [{"documento": s.document_id, "artigo": s.article,
                                   "frase": _n(s.text)} for s in rep.suggestions],
                    "fora_do_tema": {k: round(v, 3) for k, v in rep.off_topic.items()},
                    "recusadas_pela_llm": len(rep.rejected)})
    con.close()
    return out


def imprimir(r: dict) -> None:
    pct = lambda v: "—" if v is None else f"{v:.0%}"  # noqa: E731
    print(f"\nLimiar das sugestões: {r['limiar']}")
    print(f"1. Artigos alterados detectados ......... {pct(r['criterio_1_comparacao'])}  (meta 100%)")
    if r["comparacao_extras"]:
        print(f"   além do gabarito: {', '.join(r['comparacao_extras'])}")
    print(f"2. Citações diretas encontradas .......... {pct(r['criterio_2_citacoes'])}  (meta 100%)")
    for f in r["citacoes_faltando"]:
        print(f"   FALTOU: {f['documento']} art. {f['artigo']}: {f['trecho']}")
    print(f"   citações apontadas fora do gabarito: {len(r['citacoes_falsas'])}")
    for f in r["citacoes_falsas"]:
        print(f"   FALSA: {f['documento']} art. {f['artigo']}: {f['trecho'][:100]}")
    n = len(r["sugestoes"])
    certas = sum(s["correta"] for s in r["sugestoes"])
    print(f"3. Precisão das sugestões ............... {pct(r['criterio_3_precisao'])}  "
          f"({certas}/{n}, meta ≥ 70%)")
    exatas = sum(s["artigo_exato"] for s in r["sugestoes"])
    print(f"   com o artigo exato do gabarito ........ {pct(r['precisao_artigo_exato'])}  ({exatas}/{n})")
    for s in r["sugestoes"]:
        mark = ("ok " if s["artigo_exato"] else "~ok") if s["correta"] else "ERR"
        print(f"   [{mark}] {s['relevancia']:.3f} art. {s['artigo']} {s['documento']}: {s['frase'][:90]}")
        if s["motivo_llm"]:
            print(f"         LLM: {s['motivo_llm'][:150]}")
    for s in r["recusadas_pela_llm"]:
        print(f"   [recusada pela LLM, {'era correta' if s['correta'] else 'era falsa'}] "
              f"art. {s['artigo']} {s['documento']}: {s['frase'][:80]}")
        print(f"         LLM: {s['motivo_llm'][:150]}")
    print("\nDiferenças:\n" + r["diff_texto"])


def imprimir_outros(titulo: str, o: dict) -> None:
    n = len(o["sugestoes"])
    print(f"\n{titulo}: {n} sugestões (meta 0); citações diretas {o['citacoes']}; "
          f"recusadas pela LLM {o['recusadas_pela_llm']}")
    for d, v in o["fora_do_tema"].items():
        print(f"   fora do tema: {d} ({v:.3f})")
    for s in o["sugestoes"]:
        print(f"   SUGESTÃO FALSA: art. {s['artigo']} {s['documento']}: {s['frase'][:90]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--threshold", type=float, default=MIN_RELEVANCE)
    ap.add_argument("--sweep", action="store_true", help="mede a precisão em vários limiares")
    ap.add_argument("--llm", action="store_true", help="gera o resumo da mudança com a LLM")
    ap.add_argument("--verificar-llm", action="store_true",
                    help="a LLM confirma ou recusa cada sugestão (LLMChain.verify_dependency)")
    ap.add_argument("--base-real", metavar="SQLITE",
                    help="cópia do versions.db de uma base registrada pelo app, para medir também")
    ap.add_argument("--output", metavar="JSON")
    args = ap.parse_args()

    # No Windows o ChromaDB segura os arquivos abertos até o processo sair.
    verificar = None
    if args.verificar_llm:
        from llm_chain import LLMChain
        verificar = LLMChain(stream=False).verify_dependency

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        gab, store, retriever, r2 = montar_base(Path(d) / "exemplo")
        r = medir(gab, store, retriever, r2, args.threshold, verificar)
        imprimir(r)

        # Documentos de outro tema: nenhuma sugestão, com e sem LLM.
        o_store, o_retr, o_r2 = montar_outros_temas(Path(d) / "outros")
        r["outros_temas"] = medir_outros_temas(o_store, o_retr, o_r2, verificar)
        imprimir_outros("Documentos de outro tema", r["outros_temas"])
        if args.base_real:
            r["base_real"] = medir_base_real(Path(args.base_real), Path(d) / "real",
                                             verificar=verificar)
            for b in r["base_real"]:
                imprimir_outros(f"Base real, {b['documento']} v{b['versoes'][0]} -> v{b['versoes'][1]}", b)

        if args.sweep:
            print("\nlimiar  sugestões  corretas  precisão  artigo exato")
            for t in [0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.90]:
                s = medir(gab, store, retriever, r2, t)
                n = len(s["sugestoes"])
                c = sum(x["correta"] for x in s["sugestoes"])
                e = sum(x["artigo_exato"] for x in s["sugestoes"])
                f = lambda v: "—" if not n else f"{v / n:.0%}"  # noqa: E731
                print(f"{t:.2f}    {n:>5}      {c:>5}     {f(c):>5}     {f(e):>5}")

            # Filtro de tema: 0 desliga (é a regra anterior, só frase x artigo).
            print("\ntema    exemplo: sugestões  corretas  precisão   outro tema: sugestões")
            for t in [0.0, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90, 0.92]:
                s = medir(gab, store, retriever, r2, args.threshold, doc_threshold=t)
                n = len(s["sugestoes"])
                c = sum(x["correta"] for x in s["sugestoes"])
                o = len(medir_outros_temas(o_store, o_retr, o_r2, doc_threshold=t)["sugestoes"])
                f = "—" if not n else f"{c / n:.0%}"
                print(f"{t:.2f}             {n:>5}     {c:>5}     {f:>5}              {o:>5}")

        if args.llm:
            from llm_chain import LLMChain
            prev = store.get_version(r2.document.id, r2.version.version - 1)
            resumo = LLMChain(stream=False).summarize_change(
                prev.text, r2.version.text, r2.document.name, r2.diff.as_text())
            print("\nResumo da LLM:\n" + resumo)
            r["resumo_llm"] = resumo

        if args.output:
            Path(args.output).write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nRelatório salvo em {args.output}")

    ok = (r["criterio_1_comparacao"] == 1 and r["criterio_2_citacoes"] == 1
          and (r["criterio_3_precisao"] or 0) >= 0.7 and not r["outros_temas"]["sugestoes"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
