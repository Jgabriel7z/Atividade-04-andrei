"""
test_suite/test_impact.py
─────────────────────────
Testes de src/impact.py: reconhecimento de citações (regex) e análise de
impacto ponta a ponta com os documentos de exemplo (data/exemplos/),
em banco temporário. Inclui o caso de documentos de outro tema (nenhuma
sugestão, com e sem LLM) e o verificador por LLM simulado.
"""

import shutil
import tempfile
import time
from pathlib import Path
from typing import Tuple

from test_suite.conftest import _title
from test_suite.runner import TestRunner
from impact import analyze_impact, find_citations, format_report, split_sentences

AL = ["Regulamento Interno", "Regulamento"]


def _arts(text: str):
    return [m.articles for m in find_citations(text, AL)]


def test_impact() -> Tuple[int, int]:
    print(_title("MÓDULO: impact.py"))
    runner = TestRunner()
    tmp = Path(tempfile.mkdtemp(prefix="eixoai_impact_"))

    # ── Citações diretas ────────────────────────────────────

    def t_cit_simples():
        assert _arts("segue o art. 3º do Regulamento Interno.") == [["3"]]
        assert _arts("conforme o artigo 15 do Regulamento.") == [["15"]]
        assert _arts("nos termos do Art. 15 do Regulamento Interno") == [["15"]]

    def t_cit_paragrafo():
        assert _arts("prazo previsto no art. 12, § 2º, do Regulamento.") == [["12"]]
        assert _arts("o inciso II do art. 4º do Regulamento Interno") == [["4"]]

    def t_cit_lista_e_intervalo():
        assert _arts("Ver arts. 3º e 7º do Regulamento Interno.") == [["3", "7"]]
        assert _arts("Ver arts. 3º, 5º e 9º do Regulamento.") == [["3", "5", "9"]]
        assert _arts("nos arts. 2º a 4º do Regulamento Interno.") == [["2", "3", "4"]]

    def t_cit_documento_antes():
        assert _arts("as regras do Regulamento Interno, art. 12, § 1º.") == [["12"]]

    def t_cit_sufixo():
        assert _arts("o art. 12-A do Regulamento") == [["12-A"]]

    def t_cit_sem_acento_e_maiusculas():
        assert _arts("ARTIGO 7º DO REGULAMENTO INTERNO") == [["7"]]
        assert _arts("conforme o artigo 7o do regulamento") == [["7"]]

    def t_cit_negativos():
        assert _arts("na forma do art. 12 do Código Civil.") == []
        assert _arts("o art. 12 do Código Civil e o Regulamento Interno.") == []
        assert _arts("A Constituição, em seu art. 5º, garante a igualdade.") == []
        assert _arts("art. 12; o Regulamento trata de outra coisa.") == []
        assert _arts("o Regulamento Interno foi aprovado.") == []

    def t_cabecalho_proprio_nao_e_citacao():
        # Artigo do próprio manual que menciona o Regulamento não é citação do art. 3 dele
        txt = "Art. 3º Os prazos previstos no Regulamento Interno valem aqui.\nArt. 4º Fim."
        assert _arts(txt) == [], _arts(txt)
        assert _arts("Art. 3º Fim.\nconforme o art. 3º do Regulamento.") == [["3"]]

    def t_palavra_no_meio_desfaz():
        assert _arts("o prazo do art. 12, 30 dias do Regulamento") == []
        assert _arts("Os prazos do art. 12 previstos no Regulamento") == []

    def t_relatorio_sem_artigos():
        from versioning import compare_versions
        from impact import ImpactReport
        d = compare_versions("Manual sem artigos, versão um.", "Manual sem artigos, versão dois.")
        assert not d.structured
        md = format_report(ImpactReport("m", "Manual", [], structured=False))
        assert "não tem artigos" in md and "Art. documento" not in md

    def t_md_escape():
        from impact import md_escape
        s = md_escape("![x](http://mal.example/?d=1) [link](http://a) `code`")
        assert "\\!\\[x\\]\\(" in s and "\\`code\\`" in s, s

    def t_frases():
        s = split_sentences("Ver arts. 3º e 7º do Regulamento Interno para jornada e férias. "
                            "O expediente vai das 8h às 17h, de segunda a sexta.")
        assert len(s) == 2 and s[0].startswith("Ver arts. 3º e 7º"), s

    # ── Ponta a ponta com os exemplos ───────────────────────

    state = {}

    def t_base_exemplo():
        from eval.eval_versoes import montar_base
        gab, store, retriever, r2 = montar_base(tmp)
        state.update(gab=gab, store=store, retriever=retriever, r2=r2)
        assert r2.status == "created" and r2.version.version == 2
        assert sorted(c.key for c in r2.diff.changed if c.key != "preambulo") == ["12", "3", "7"]
        assert [a.key for a in r2.diff.removed] == ["15"]
        assert [a.key for a in r2.diff.added] == ["17"]

    def t_impacto_diretas():
        rep = analyze_impact(state["store"], state["retriever"], state["r2"].document.id,
                             state["r2"].diff, semantic=False)
        got = sorted((c.document_id, c.article) for c in rep.direct)
        exp = sorted((r["documento"], r["artigo"]) for r in state["gab"]["referencias_cruzadas"])
        assert got == exp, f"\n  obtido:   {got}\n  esperado: {exp}"
        assert all(c.document_id != "regulamento-interno" for c in rep.direct)

    def t_impacto_sugestoes():
        rep = analyze_impact(state["store"], state["retriever"], state["r2"].document.id,
                             state["r2"].diff)
        assert all(s.relevance >= 0.0 for s in rep.suggestions)
        for s in rep.suggestions:
            cites = [m.articles for m in find_citations(s.text, AL)]
            assert not any(s.article in a for a in cites), f"sugestão é citação direta: {s.text}"
        md = format_report(rep)
        assert "Certo (10)" in md, md[:300]

    # ── Documentos de outro tema e verificação por LLM ──────

    def t_outros_temas_sem_llm():
        from eval.eval_versoes import montar_outros_temas
        store, retriever, r2 = montar_outros_temas(tmp / "outros")
        state.update(o_store=store, o_retr=retriever, o_r2=r2)
        rep = analyze_impact(store, retriever, r2.document.id, r2.diff)
        assert rep.suggestions == [], [(s.document_name, s.text[:60]) for s in rep.suggestions]
        assert rep.direct == []
        assert len(rep.off_topic) == 3, rep.off_topic

    def t_outros_temas_com_llm():
        # Mesmo uma LLM que aprovasse tudo não cria sugestão: o filtro de tema
        # vem antes, e ela nem chega a ser chamada.
        chamadas = []

        def aprova_tudo(antes, depois, trecho):
            chamadas.append(trecho)
            return True, "trata do mesmo assunto"

        rep = analyze_impact(state["o_store"], state["o_retr"], state["o_r2"].document.id,
                             state["o_r2"].diff, verificar=aprova_tudo)
        assert rep.suggestions == [] and chamadas == [], chamadas

    def t_mesmo_tema_passa_no_filtro():
        rep = analyze_impact(state["store"], state["retriever"], state["r2"].document.id,
                             state["r2"].diff)
        assert rep.off_topic == {}, rep.off_topic
        assert rep.suggestions and all(s.topic >= 0.87 for s in rep.suggestions)

    def _com_verificador(fn):
        return analyze_impact(state["store"], state["retriever"], state["r2"].document.id,
                              state["r2"].diff, verificar=fn)

    def t_verificador_recusa_e_confirma():
        base = analyze_impact(state["store"], state["retriever"], state["r2"].document.id,
                              state["r2"].diff)
        rep = _com_verificador(lambda a, d, t: (False, "fala de outra coisa"))
        assert rep.suggestions == [] and len(rep.rejected) == len(base.suggestions)
        assert rep.llm_used
        vistos = []

        def confirma(antes, depois, trecho):
            vistos.append((antes, depois, trecho))
            return True, "o trecho repete a regra do artigo"

        rep = _com_verificador(confirma)
        assert len(rep.suggestions) == len(base.suggestions)
        assert all(s.verified and s.reason == "o trecho repete a regra do artigo"
                   for s in rep.suggestions)
        # O verificador recebe o artigo antes, o artigo depois (ou o aviso de que
        # saiu) e o trecho do outro documento.
        assert all(a.startswith("Regulamento Interno, art.") for a, _, _ in vistos)
        retirado = any(d == "(o artigo foi retirado do documento)" for _, d, _ in vistos)
        assert retirado == any(s.article == "15" for s in base.suggestions)
        md = format_report(rep)
        assert "o trecho repete a regra do artigo" in md, md

    def t_verificador_com_erro_ou_lento():
        import impact
        base = analyze_impact(state["store"], state["retriever"], state["r2"].document.id,
                              state["r2"].diff)

        def quebra(a, d, t):
            raise RuntimeError("sem conexão")

        rep = _com_verificador(quebra)
        assert len(rep.suggestions) == len(base.suggestions)
        assert all(s.verified is None for s in rep.suggestions)

        antigo = impact.VERIFY_TIMEOUT
        impact.VERIFY_TIMEOUT = 0.3
        try:
            t0 = time.perf_counter()
            rep = _com_verificador(lambda a, d, t: (time.sleep(2), (False, ""))[1])
            assert time.perf_counter() - t0 < 1.5, "a análise esperou a LLM lenta"
            assert len(rep.suggestions) == len(base.suggestions)
        finally:
            impact.VERIFY_TIMEOUT = antigo

    def t_relatorio_leigo():
        rep = analyze_impact(state["store"], state["retriever"], state["r2"].document.id,
                             state["r2"].diff)
        md = format_report(rep)
        baixo = md.lower()
        for jargao in ("chunk", "semelhança", "relevância", "limiar", "regex"):
            assert jargao not in baixo, f"jargão {jargao!r} no relatório"
        assert all(ord(ch) < 0x2190 or ch in "“”…" for ch in md), "emoji ou símbolo no relatório"
        assert "Mudaram os artigos 3, 7 e 12, e saiu o artigo 15." in md, md[:200]
        assert "cita diretamente" in md and "pode depender do art." in md
        assert "o texto trata do mesmo assunto" in md

    def t_busca_chat_versao_vigente():
        hits = state["retriever"].search("prazo de reembolso de despesas", top_k=5)
        reg = [h for h in hits if h["document_id"] == "regulamento-interno"]
        vigentes = {c["id"] for c in state["retriever"].get_chunks(document_id="regulamento-interno")}
        assert reg and all(h["id"] in vigentes for h in reg), "achou chunk não vigente"
        assert any("15 (quinze)" in h["text"] for h in reg), "deveria achar o texto novo"
        assert not any("30 (trinta) dias após" in h["text"] for h in hits), "achou o texto antigo"

    runner.run("Citação simples (art., artigo, Art.)",       t_cit_simples)
    runner.run("Citação com parágrafo e inciso",              t_cit_paragrafo)
    runner.run("Lista e intervalo de artigos",                t_cit_lista_e_intervalo)
    runner.run("Documento antes do artigo",                   t_cit_documento_antes)
    runner.run("Artigo com letra (12-A)",                     t_cit_sufixo)
    runner.run("Sem acento e em maiúsculas",                  t_cit_sem_acento_e_maiusculas)
    runner.run("Não confunde com outros documentos",          t_cit_negativos)
    runner.run("Cabeçalho do próprio artigo não é citação",   t_cabecalho_proprio_nao_e_citacao)
    runner.run("Palavra solta entre número e documento",      t_palavra_no_meio_desfaz)
    runner.run("Relatório de documento sem artigos",          t_relatorio_sem_artigos)
    runner.run("Escape de Markdown em texto de documento",    t_md_escape)
    runner.run("Divisão em frases não quebra em 'art.'",      t_frases)
    runner.run("Base de exemplo: 3 documentos, 2 versões",    t_base_exemplo)
    runner.run("Impacto: as 10 citações diretas, sem falsas", t_impacto_diretas)
    runner.run("Impacto: sugestões não repetem citação",      t_impacto_sugestoes)
    runner.run("Outro tema: nenhuma sugestão, sem LLM",       t_outros_temas_sem_llm)
    runner.run("Outro tema: nenhuma sugestão, com LLM",       t_outros_temas_com_llm)
    runner.run("Mesmo tema passa no filtro de tema",          t_mesmo_tema_passa_no_filtro)
    runner.run("Verificador por LLM recusa e confirma",       t_verificador_recusa_e_confirma)
    runner.run("Verificador com erro ou lento não trava",     t_verificador_com_erro_ou_lento)
    runner.run("Relatório em linguagem simples",              t_relatorio_leigo)
    runner.run("Chat busca só a versão vigente",              t_busca_chat_versao_vigente)

    shutil.rmtree(tmp, ignore_errors=True)
    return runner.summary()
