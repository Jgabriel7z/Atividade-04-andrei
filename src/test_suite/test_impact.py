"""
test_suite/test_impact.py
─────────────────────────
Testes de src/impact.py: reconhecimento de citações (regex) e análise de
impacto ponta a ponta com os documentos de exemplo (data/exemplos/),
em banco temporário.
"""

import shutil
import tempfile
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
        assert "Citações diretas (10)" in md, md[:300]

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
    runner.run("Chat busca só a versão vigente",              t_busca_chat_versao_vigente)

    shutil.rmtree(tmp, ignore_errors=True)
    return runner.summary()
