"""
test_suite/test_equipe.py
─────────────────────────
Conversa entre as pessoas, presença, leitura e contas de acesso (equipe.py).
Não precisa de chave nem de modelo.
"""

import shutil
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

from test_suite.conftest import _title
from test_suite.runner import TestRunner
from equipe import SALA_GERAL, Equipe, ler_admins, ler_contas, quando_humano
from versioning import VersionStore


def test_equipe() -> Tuple[int, int]:
    _title("TESTES: equipe.py (conversa, presença, contas)")
    runner = TestRunner()
    tmp = Path(tempfile.mkdtemp(prefix="eixoai_equipe_"))

    def t_contas():
        contas = ler_contas("joao:s1:João Gabriel, luiz:s2 ,ruim, :x, ana:")
        assert [(c.login, c.senha, c.nome) for c in contas] == [
            ("joao", "s1", "João Gabriel"), ("luiz", "s2", "Luiz")]
        assert ler_contas("") == []

    def t_conversa_por_sala():
        eq = Equipe(tmp / "a.db")
        eq.enviar(SALA_GERAL, "João", "bom dia")
        eq.enviar("regulamento", "Luiz", "o art. 12 mudou?")
        assert eq.enviar(SALA_GERAL, "João", "   ") is None, "mensagem vazia não entra"
        assert [m.texto for m in eq.conversa(SALA_GERAL)] == ["bom dia"]
        assert [m.autor for m in eq.conversa("regulamento")] == ["Luiz"]
        assert eq.ultima_mensagem() == 2

    def t_nao_lidas():
        eq = Equipe(tmp / "b.db")
        eq.enviar(SALA_GERAL, "Luiz", "oi")
        eq.enviar(SALA_GERAL, "Luiz", "viu a v2?")
        eq.enviar(SALA_GERAL, "João", "vi")
        assert eq.nao_lidas("João") == {SALA_GERAL: 2}, "as próprias mensagens não contam"
        eq.marcar_lida("João", SALA_GERAL)
        assert eq.nao_lidas("João") == {}
        eq.enviar("manual", "Luiz", "e o manual?")
        assert eq.nao_lidas("João") == {"manual": 1}

    def t_presenca():
        eq = Equipe(tmp / "c.db")
        eq.sinal("João", "Documentos: Regulamento Interno")
        eq.sinal("Luiz", "Perguntar")
        assert eq.online() == [("João", "Documentos: Regulamento Interno"), ("Luiz", "Perguntar")]
        depois = datetime.now() + timedelta(seconds=60)
        assert eq.online(agora=depois) == [], "sem sinal recente = saiu"

    def t_atividade_junta_versoes_e_mensagens():
        db = tmp / "d.db"
        store = VersionStore(db)                      # mesmo arquivo, como no app
        store.ensure_document("reg", "Regulamento Interno")
        store.add_version("reg", 1, filename="r1.pdf", sha256="a", text="x", author="João")
        store.add_version("reg", 2, filename="r2.pdf", sha256="b", text="y", author="Luiz",
                          reason="prazo menor")
        eq = Equipe(db)
        eq.enviar("reg", "João", "conferi o art. 12")
        textos = [(e.quem, e.tipo, e.texto) for e in eq.atividade()]
        assert ("Luiz", "versao", "publicou a versão 2 de Regulamento Interno: prazo menor") in textos
        assert ("João", "versao", "enviou Regulamento Interno") in textos
        assert ("João", "mensagem", "escreveu sobre Regulamento Interno: conferi o art. 12") in textos

    def t_conversas_ia():
        eq = Equipe(tmp / "e.db")
        eq.registrar_pergunta("c1", "Luiz", "qual o prazo de reembolso?", "15 dias (art. 12).",
                              escopo="Todos os documentos", modelo="openai/gpt-oss-120b",
                              fontes=[{"documento": "Regulamento", "pagina": 1, "trecho": "Art. 12..."}])
        eq.registrar_pergunta("c2", "João", "e as férias?", "30 dias de antecedência.")
        eq.registrar_pergunta("c1", "Luiz", "e se passar de 90 dias?", "Não é reembolsado.")
        convs = eq.conversas_ia()
        assert [c.id for c in convs] == ["c1", "c2"], "mais recente primeiro"
        assert [q.pergunta for q in convs[0].perguntas] == ["qual o prazo de reembolso?",
                                                             "e se passar de 90 dias?"]
        assert convs[0].perguntas[0].fontes[0]["documento"] == "Regulamento"
        assert [c.autor for c in eq.conversas_ia(autor="João")] == ["João"]
        assert eq.autores_ia() == ["Luiz", "João"]
        assert eq.ultima_pergunta_ia() == 3

    def t_ia_na_atividade_so_para_admin():
        eq = Equipe(tmp / "f.db")
        eq.registrar_pergunta("c1", "Luiz", "qual o prazo?", "15 dias.")
        assert [e.tipo for e in eq.atividade()] == [], "quem não é admin não vê perguntas à IA"
        assert [(e.quem, e.texto) for e in eq.atividade(incluir_ia=True)] == [
            ("Luiz", "perguntou à IA: qual o prazo?")]

    def t_admins():
        assert ler_admins(" joao, luiz ,,") == {"joao", "luiz"}
        assert ler_admins("") == set()

    def t_quando_humano():
        agora = datetime(2026, 11, 19, 15, 0, 0)
        assert quando_humano("2026-11-19T14:59:40", agora) == "agora"
        assert quando_humano("2026-11-19T14:50:00", agora) == "há 10 min"
        assert quando_humano("2026-11-19T09:05:00", agora) == "hoje às 09:05"
        assert quando_humano("2026-11-18T22:10:00", agora) == "ontem às 22:10"
        assert quando_humano("2026-11-10T08:00:00", agora) == "10/11 às 08:00"

    runner.run("Contas do .env (login:senha:Nome)",           t_contas)
    runner.run("Conversa geral e por documento",              t_conversa_por_sala)
    runner.run("Não lidas por sala, sem contar as próprias",  t_nao_lidas)
    runner.run("Quem está online e vendo o quê",              t_presenca)
    runner.run("Atividade junta versões e mensagens",         t_atividade_junta_versoes_e_mensagens)
    runner.run("Datas em linguagem simples",                  t_quando_humano)
    runner.run("Conversas com a IA: guarda, agrupa e filtra", t_conversas_ia)
    runner.run("Perguntas à IA na atividade só para admin",   t_ia_na_atividade_so_para_admin)
    runner.run("Admins do .env (EIXOAI_ADMINS)",              t_admins)

    shutil.rmtree(tmp, ignore_errors=True)
    return runner.summary()
