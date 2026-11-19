"""
test_suite/test_busca.py
────────────────────────
Testes de src/busca.py (busca híbrida densa + BM25 com fusão RRF e reescrita
opcional pela LLM) e de LLMChain.reescrever_pergunta, em banco temporário.
Nenhum teste precisa da GROQ_API_KEY: a LLM é simulada.
"""

import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

from test_suite.conftest import _title
from test_suite.runner import TestRunner
import busca
from busca import buscar, normalizar, palavras, IndiceLexical
from retriever import DocumentRetriever

_REG = (
    "REGULAMENTO DE TESTE\n\n"
    "Art. 1º A jornada de trabalho é de 8 (oito) horas diárias.\n\n"
    "Art. 2º O registro de ponto é obrigatório no início e no fim da jornada.\n\n"
    "Art. 3º O atestado médico deve ser entregue em até 48 (quarenta e oito) horas.\n\n"
    "Art. 4º O reembolso de despesas será pago em até 15 (quinze) dias."
)
_EDITAL = (
    "EDITAL DE SELEÇÃO\n\n"
    "1. Das inscrições. As inscrições serão feitas pelo sistema acadêmico, sem taxa.\n\n"
    "2. Da avaliação. O currículo Lattes será avaliado pela comissão de seleção.\n\n"
    "3. Do resultado. O resultado final será divulgado no site do programa."
)
CAMPOS = {"id", "text", "distance", "relevance", "source", "page", "chunk",
          "article", "document_id", "version", "rank"}


def test_busca() -> Tuple[int, int]:
    print(_title("MÓDULO: busca.py"))
    runner = TestRunner()
    tmp = Path(tempfile.mkdtemp(prefix="eixoai_busca_"))
    ctx = {}

    def t_base():
        r = DocumentRetriever(db_path=tmp / "chroma")
        # chunk pequeno: um artigo por trecho, para dar para conferir qual voltou
        r._chunk_text = lambda text, *a, **k: r._chunk_text_legal(text, chunk_size=120) \
            if text.startswith("REG") else r._chunk_text_semantic(text, 120, 0)
        r.add_version("reg", _REG, 1, source="reg.pdf")
        r.add_version("edital", _EDITAL, 1, source="edital.pdf")
        ctx["r"] = r
        assert r.collection.count() >= 6, r.collection.count()

    def t_normalizacao():
        assert normalizar("Férias INSCRIÇÃO") == "ferias inscricao"
        assert palavras("Qual é o prazo do reembolso?") == ["prazo", "reembolso"]
        assert palavras("Taxa de INSCRIÇÃO") == ["taxa", "inscricao"]

    def t_corrige_digitacao():
        idx = IndiceLexical([{"id": "a", "text": "Currículo Lattes e atestado"}])
        assert idx.corrigir("lates") == ["lattes"]
        assert idx.corrigir("atestado") == ["atestado"]
        assert idx.corrigir("xyzw") == ["xyzw"]

    def t_formato():
        res = buscar(ctx["r"], "qual o prazo do reembolso?", k=3)
        assert res, "sem resultado"
        assert len(res) <= 3
        for x in res:
            assert CAMPOS <= set(x), set(x) ^ CAMPOS
            assert 0.0 <= x["relevance"] <= 1.0001
            assert abs(x["relevance"] - (1 - x["distance"])) < 1e-6
        assert [x["rank"] for x in res] == list(range(1, len(res) + 1))
        # format_context da interface continua aceitando o resultado
        assert "CONTEXTO RELEVANTE" in DocumentRetriever.format_context(res)

    def t_sem_acento_e_curta():
        for q in ("reembolso?", "REEMBOLSO", "reembolço"):
            res = buscar(ctx["r"], q, k=2)
            assert any("15 (quinze)" in x["text"] for x in res), (q, [x["text"][:40] for x in res])

    def t_erro_digitacao():
        res = buscar(ctx["r"], "atestdo", k=2)
        assert any("atestado médico" in x["text"] for x in res), [x["text"][:40] for x in res]

    def t_filtro_documento():
        res = buscar(ctx["r"], "inscrições sem taxa", document_id="reg", k=5)
        assert res and all(x["document_id"] == "reg" for x in res)
        res = buscar(ctx["r"], "registro de ponto", document_id="edital", k=5)
        assert res and all(x["document_id"] == "edital" for x in res)

    def t_so_vigente():
        r = ctx["r"]
        novo = _REG.replace("15 (quinze) dias", "10 (dez) dias")
        r.add_version("reg", novo, 2, source="reg_v2.pdf")
        res = buscar(r, "prazo do reembolso", k=5)
        textos = " ".join(x["text"] for x in res)
        assert "10 (dez) dias" in textos and "15 (quinze)" not in textos, textos[:200]

    def t_indice_atualiza():
        r = ctx["r"]
        buscar(r, "qualquer coisa", k=1)            # garante índice em cache
        r.add_version("novo", "Art. 1º O uso de crachá é obrigatório na portaria.", 1, source="n.pdf")
        res = buscar(r, "cracha", k=1)
        assert res and res[0]["document_id"] == "novo", res

    def t_reescrita_usada():
        # a palavra "portaria" só existe no documento "novo"; a pergunta não a tem
        chamadas = []

        def reescrever(p):
            chamadas.append(p)
            return ["identificação na portaria", "crachá obrigatório"]
        res = buscar(ctx["r"], "preciso usar alguma identificação para entrar?", k=2,
                     reescrever=reescrever)
        assert chamadas == ["preciso usar alguma identificação para entrar?"]
        assert any(x["document_id"] == "novo" for x in res), res

    def t_reescrita_com_erro():
        def quebra(_):
            raise RuntimeError("sem GROQ_API_KEY")
        a = buscar(ctx["r"], "registro de ponto", k=3)
        b = buscar(ctx["r"], "registro de ponto", k=3, reescrever=quebra)
        assert [x["id"] for x in a] == [x["id"] for x in b]
        c = buscar(ctx["r"], "registro de ponto", k=3, reescrever=lambda _: None)
        assert [x["id"] for x in a] == [x["id"] for x in c]

    def t_reescrita_lenta():
        def lenta(_):
            time.sleep(3)
            return ["registro de ponto"]
        t0 = time.perf_counter()
        res = buscar(ctx["r"], "bater ponto", k=3, reescrever=lenta, timeout_reescrita=0.5)
        dt = time.perf_counter() - t0
        assert res, "a busca tem que seguir sem a reescrita"
        assert dt < 2.5, f"esperou a reescrita lenta ({dt:.1f} s)"

    def t_pergunta_vazia():
        assert buscar(ctx["r"], "   ") == []

    def t_falha_volta_para_densa():
        r = ctx["r"]
        original = busca._indice
        busca._indice = lambda _r: (_ for _ in ()).throw(RuntimeError("quebrou"))
        try:
            res = buscar(r, "registro de ponto", k=2)
        finally:
            busca._indice = original
        assert res and CAMPOS <= set(res[0])

    def t_llm_reescrever_pergunta():
        # LLMChain sem rede: só o tratamento da resposta.
        from llm_chain import LLMChain
        chain = LLMChain.__new__(LLMChain)
        chain.model = "qwen/teste"
        chain.conversation_history = []
        texto = ("<think>pensando</think>\n1) Qual o prazo para o reembolso de despesas?\n"
                 "em linguagem simples:\n- Em quantos dias recebo o dinheiro de volta?\n"
                 "3. reembolso prazo despesas\nextra que sobra")
        pedido = {}

        def create(**kw):
            pedido.update(kw)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=texto))])
        chain.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        v = chain.reescrever_pergunta("reembolso?", timeout=3)
        assert v == ["Qual o prazo para o reembolso de despesas?",
                     "Em quantos dias recebo o dinheiro de volta?",
                     "reembolso prazo despesas"], v
        assert pedido["timeout"] == 3 and "reasoning_effort" not in pedido
        assert chain.conversation_history == []

    runner.run("Base temporária com 2 documentos",             t_base)
    runner.run("Normalização: acento, maiúsculas, stopwords",  t_normalizacao)
    runner.run("Correção de erro de digitação no vocabulário", t_corrige_digitacao)
    runner.run("Formato igual ao retriever.search",            t_formato)
    runner.run("Pergunta curta, sem acento, maiúscula",        t_sem_acento_e_curta)
    runner.run("Pergunta com erro de digitação",               t_erro_digitacao)
    runner.run("Filtro por documento",                         t_filtro_documento)
    runner.run("Só a versão vigente",                          t_so_vigente)
    runner.run("Índice lexical acompanha documento novo",      t_indice_atualiza)
    runner.run("Variantes da reescrita entram na busca",       t_reescrita_usada)
    runner.run("Reescrita com erro não muda o resultado",      t_reescrita_com_erro)
    runner.run("Reescrita lenta respeita o tempo limite",      t_reescrita_lenta)
    runner.run("Pergunta vazia",                               t_pergunta_vazia)
    runner.run("Falha na busca híbrida volta para a densa",    t_falha_volta_para_densa)
    runner.run("LLMChain.reescrever_pergunta (sem rede)",      t_llm_reescrever_pergunta)

    shutil.rmtree(tmp, ignore_errors=True)
    return runner.summary()
