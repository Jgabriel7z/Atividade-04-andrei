"""
eval/eval_versoes.py
────────────────────
Mede os critérios da melhoria (Atividade 03, "Como vamos validar") no conjunto
de exemplo: Regulamento Interno, Contrato Modelo e Manual do Colaborador.

  1. Artigos alterados detectados pela comparação          meta 100%
  2. Citações diretas encontradas pela análise de impacto   meta 100%
  3. Precisão das sugestões semânticas (sem falso alarme)  meta ≥ 70%

O 4º critério (Hit Rate da CF/88 não cair) é medido com `python src/test.py --sample`.

Uso (a partir da raiz do projeto):
    python src/eval/eval_versoes.py
    python src/eval/eval_versoes.py --sweep          # precisão por limiar
    python src/eval/eval_versoes.py --llm            # inclui o resumo da LLM (precisa de GROQ_API_KEY)
    python src/eval/eval_versoes.py --output r.json
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SRC))

from eval.exemplos import EXEMPLOS, gerar_pdfs          # noqa: E402
from impact import analyze_impact, MIN_RELEVANCE        # noqa: E402
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


def medir(gab, store, retriever, r2, threshold: float) -> dict:
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
    rep = analyze_impact(store, retriever, r2.document.id, diff, min_relevance=threshold)

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
                    "relevancia": round(s.relevance, 4), "correta": correta,
                    "artigo_exato": exato, "frase": _n(s.text)})
    c3 = (sum(x["correta"] for x in sug) / len(sug)) if sug else None
    c3_exato = (sum(x["artigo_exato"] for x in sug) / len(sug)) if sug else None

    return {
        "limiar": threshold,
        "criterio_1_comparacao": c1, "comparacao_extras": [f"{t} {k}" for t, k in extras],
        "criterio_2_citacoes": c2, "citacoes_faltando": faltando, "citacoes_falsas": falsos_diretos,
        "criterio_3_precisao": c3, "precisao_artigo_exato": c3_exato, "sugestoes": sug,
        "diff_texto": diff.as_text(),
    }


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
    print("\nDiferenças:\n" + r["diff_texto"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--threshold", type=float, default=MIN_RELEVANCE)
    ap.add_argument("--sweep", action="store_true", help="mede a precisão em vários limiares")
    ap.add_argument("--llm", action="store_true", help="gera o resumo da mudança com a LLM")
    ap.add_argument("--output", metavar="JSON")
    args = ap.parse_args()

    # No Windows o ChromaDB segura os arquivos abertos até o processo sair.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        gab, store, retriever, r2 = montar_base(Path(d))
        r = medir(gab, store, retriever, r2, args.threshold)
        imprimir(r)

        if args.sweep:
            print("\nlimiar  sugestões  corretas  precisão  artigo exato")
            for t in [0.80, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.90]:
                s = medir(gab, store, retriever, r2, t)
                n = len(s["sugestoes"])
                c = sum(x["correta"] for x in s["sugestoes"])
                e = sum(x["artigo_exato"] for x in s["sugestoes"])
                f = lambda v: "—" if not n else f"{v / n:.0%}"  # noqa: E731
                print(f"{t:.2f}    {n:>5}      {c:>5}     {f(c):>5}     {f(e):>5}")

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
          and (r["criterio_3_precisao"] or 0) >= 0.7)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
