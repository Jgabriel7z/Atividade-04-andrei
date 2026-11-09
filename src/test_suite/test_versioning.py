"""
test_suite/test_versioning.py
─────────────────────────────
Testes de src/versioning.py, das mudanças de versão em src/retriever.py
e do LLMChain.summarize_change.

As partes de SQLite e comparação não carregam o modelo de embeddings.
Os testes com ChromaDB usam um banco temporário, sem tocar em ../vector_db.
"""

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Tuple

from test_suite.conftest import _title
from test_suite.runner import TestRunner
from versioning import (
    VersionStore, compare_versions, publish_version, sha256_text, slugify, split_articles,
)

V1 = """REGULAMENTO DE TESTE
Art. 1º Objeto do regulamento.
Art. 2º O prazo de entrega é de 30 (trinta) dias, nos termos do art. 1º.
Art. 3º O horário é das 8h às 17h.
Art. 4º Revogam-se as disposições em contrário.
"""
V2 = """REGULAMENTO DE TESTE
Art. 1º Objeto do regulamento.
Art. 2º O prazo de entrega é de 15 (quinze) dias, nos termos do art. 1º.
Art. 3º O horário é das 8h   às 17h.
Art. 5º Fica criado o auxílio mensal.
"""


class FakeRetriever:
    """Imita a parte do DocumentRetriever usada por publish_version."""

    def __init__(self):
        self.chunks = []

    def add_version(self, document_id, text, version, source="", metadata=None, **_):
        for c in self.chunks:
            if c["document_id"] == document_id and c["is_current"]:
                c["is_current"], c["version_to"] = False, version
        self.chunks.append({"id": f"{document_id}_v{version}_0", "document_id": document_id,
                            "version": version, "is_current": True})
        return {"count": 1, "added": 1, "kept": 0, "retired": 0, "ids": [self.chunks[-1]["id"]]}

    def rollback_version(self, document_id, version):
        self.chunks = [c for c in self.chunks
                       if not (c["document_id"] == document_id and c["version"] == version)]
        for c in self.chunks:
            if c.get("version_to") == version:
                c["is_current"] = True


def test_versioning() -> Tuple[int, int]:
    print(_title("MÓDULO: versioning.py"))
    runner = TestRunner()
    tmp = Path(tempfile.mkdtemp(prefix="eixoai_versioning_"))

    # ── Divisão e comparação ────────────────────────────────

    def t_split_articles():
        arts = split_articles(V1)
        assert [a.key for a in arts] == ["preambulo", "1", "2", "3", "4"], [a.key for a in arts]
        assert arts[2].text.startswith("Art. 2º")

    def t_split_ignora_citacao():
        # "art. 1º" no meio da frase é citação, não começo de artigo
        arts = split_articles("Art. 1º Texto.\nArt. 2º Conforme o art. 1º e o\nart. 9º do Código.")
        assert [a.key for a in arts] == ["1", "2"], [a.key for a in arts]

    def t_split_numeracao_repetida():
        arts = split_articles("Art. 1º A.\nArt. 2º B.\nATO DAS DISPOSIÇÕES TRANSITÓRIAS\nArt. 1º C.")
        assert [a.key for a in arts] == ["1", "2", "1#2"], [a.key for a in arts]

    def t_split_sufixo_letra():
        arts = split_articles("Art. 12. A.\nArt. 12-A. B.\nArt. 13. C.")
        assert [a.key for a in arts] == ["12", "12-A", "13"], [a.key for a in arts]

    def t_split_estilo_clt():
        # "Art. 5º - A todo trabalho…": o "A" é o caput, não o artigo 5-A
        arts = split_articles("Art. 3º - A jornada é de 8 horas.\nArt. 4º - O salário é mensal.")
        assert [a.key for a in arts] == ["3", "4"], [a.key for a in arts]

    def t_diff_texto_grande_sem_artigos():
        import random, time
        rnd = random.Random(1)
        vocab = [f"palavra{i}" for i in range(300)] + ["de", "a", "o", "que", "e", "do"] * 20
        old = " ".join(rnd.choice(vocab) for _ in range(12000))
        new = old.replace("palavra7 ", "palavra8 ", 50)
        t0 = time.perf_counter()
        d = compare_versions(old, new)
        dt = time.perf_counter() - t0
        assert not d.structured and d.changed, "deve achar a mudança"
        assert dt < 2.0, f"diff de 12 mil palavras levou {dt:.1f}s"

    def t_compare():
        d = compare_versions(V1, V2)
        assert [c.key for c in d.changed] == ["2"], [c.key for c in d.changed]
        assert [a.key for a in d.removed] == ["4"]
        assert [a.key for a in d.added] == ["5"]
        assert d.unchanged == 3, d.unchanged   # preâmbulo, 1 e 3 (só espaço mudou)
        assert ("30 (trinta)", "15 (quinze)") in d.changed[0].edits, d.changed[0].edits
        assert d.affected_keys == ["2", "4"]

    def t_compare_igual():
        d = compare_versions(V1, V1.replace("\n", "\n\n"))
        assert not d.has_changes

    def t_compare_sem_artigos():
        d = compare_versions("Texto livre um.", "Texto livre dois.")
        assert not d.structured and len(d.changed) == 1

    def t_as_text():
        txt = compare_versions(V1, V2).as_text()
        assert 'Art. 2 alterado: "30 (trinta)" → "15 (quinze)"' in txt, txt
        assert "Art. 4 removido." in txt and "Art. 5 incluído." in txt

    def t_slugify():
        assert slugify("Regulamento Interno (2026)") == "regulamento-interno-2026"
        assert slugify("Manual do Colaborador") == "manual-do-colaborador"

    # ── SQLite ──────────────────────────────────────────────

    def t_store_registro():
        store = VersionStore(tmp / "a.db")
        store.ensure_document("reg", "Regulamento", ["Regulamento", "Reg"])
        v1 = store.add_version("reg", 1, filename="r1.pdf", sha256="aa", text=V1,
                               author="Ana", reason="inicial", n_chunks=4)
        v2 = store.add_version("reg", 2, filename="r2.pdf", sha256="bb", text=V2,
                               author="Bruno", reason="prazo menor", n_chunks=4)
        assert v1.author == "Ana" and v2.reason == "prazo menor"
        assert store.get_current("reg").version == 2
        assert not store.get_version("reg", 1).is_current
        assert store.get_version("reg", 1).text == V1, "texto da versão antiga deve ficar guardado"
        assert [v.version for v in store.list_versions("reg")] == [1, 2]
        assert store.next_version("reg") == 3
        assert store.get_document("reg").aliases == ["Regulamento", "Reg"]
        assert len(v2.created_at) >= 19   # data ISO com hora

    def t_store_persiste():
        # Reabrir o banco mantém tudo (persistência entre execuções)
        store = VersionStore(tmp / "a.db")
        assert store.get_current("reg").sha256 == "bb"

    # ── publish_version ─────────────────────────────────────

    def t_publish():
        store, fr = VersionStore(tmp / "b.db"), FakeRetriever()
        r1 = publish_version(store, fr, name="Regulamento de Teste", text=V1, filename="v1.pdf",
                             sha256=sha256_text(V1), author="Ana", reason="inicial")
        assert r1.status == "created" and r1.version.version == 1 and r1.diff is None
        r2 = publish_version(store, fr, name="Regulamento de Teste", text=V2, filename="v2.pdf",
                             sha256=sha256_text(V2), author="Bruno", reason="revisão")
        assert r2.status == "created" and r2.version.version == 2
        assert r2.previous.version == 1
        assert [c.key for c in r2.diff.changed] == ["2"]
        cur = [c for c in fr.chunks if c["is_current"]]
        assert len(cur) == 1 and cur[0]["version"] == 2, cur
        assert len(fr.chunks) == 2, "a versão antiga continua no índice"

    def t_publish_mesmo_arquivo():
        store, fr = VersionStore(tmp / "c.db"), FakeRetriever()
        publish_version(store, fr, name="Doc", text=V1, filename="v1.pdf", sha256="x")
        r = publish_version(store, fr, name="Doc", text=V1, filename="v1-copia.pdf", sha256="x")
        assert r.status == "unchanged" and r.version.version == 1
        assert store.next_version("doc") == 2 and len(fr.chunks) == 1

    def t_publish_varios_documentos():
        store, fr = VersionStore(tmp / "d.db"), FakeRetriever()
        publish_version(store, fr, name="Doc A", text=V1, filename="a.pdf", sha256="a")
        publish_version(store, fr, name="Doc B", text=V2, filename="b.pdf", sha256="b")
        assert [d.id for d in store.list_documents()] == ["doc-a", "doc-b"]
        assert len(store.current_texts()) == 2
        assert [d.id for d, _ in store.current_texts(exclude_doc_id="doc-a")] == ["doc-b"]

    # ── LLM: resumo não polui o histórico ───────────────────

    def t_summarize_sem_historico():
        from llm_chain import LLMChain
        chain = LLMChain.__new__(LLMChain)          # sem Groq(): não precisa de chave
        chain.model, chain.temperature, chain.max_tokens, chain.top_p = "fake", 0.7, 256, 1.0
        chain.conversation_history = [{"role": "user", "content": "oi"}]
        sent = {}

        def create(**kw):
            sent.update(kw)
            msg = SimpleNamespace(content=" O prazo do art. 2 caiu de 30 para 15 dias. ")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

        chain.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        out = chain.summarize_change(V1, V2, "Regulamento", compare_versions(V1, V2).as_text())
        assert out == "O prazo do art. 2 caiu de 30 para 15 dias."
        assert chain.conversation_history == [{"role": "user", "content": "oi"}], "histórico mudou"
        assert len(sent["messages"]) == 2 and sent["messages"][0]["role"] == "system"
        assert "15 (quinze)" in sent["messages"][1]["content"]

    # ── ChromaDB real: versão vigente na busca ──────────────

    # Documento com vários chunks: 8 artigos de ~450 caracteres (2 por chunk).
    temas = ["férias", "jornada", "reembolso", "uniforme", "sigilo", "transporte",
             "treinamento", "advertência"]
    LONG_V1 = "\n".join(
        f"Art. {i + 1}º Sobre {t}: " + (f"regra número {i + 1} do tema {t}, aplicada a todos. " * 7)
        for i, t in enumerate(temas))
    LONG_V2 = LONG_V1.replace("Art. 3º Sobre reembolso: regra número 3",
                              "Art. 3º Sobre reembolso: prazo de 15 dias, regra número 3")

    def t_retriever_incremental():
        from retriever import DocumentRetriever
        r = DocumentRetriever(db_path=tmp / "chroma")
        a = r.add_version("reg", LONG_V1, version=1, source="v1.pdf")
        assert a["added"] == a["count"] > 2 and a["kept"] == 0, a
        b = r.add_version("reg", LONG_V2, version=2, source="v2.pdf")
        assert b["added"] == 1 and b["retired"] == 1, f"só o chunk do art. 3 muda: {b}"
        assert b["kept"] == a["count"] - 1, b
        assert not set(a["ids"]) & set(b["ids"]), "IDs de versões diferentes não podem colidir"
        total = r.get_stats()["total_chunks"]
        assert total == a["count"] + 1, f"sem duplicar chunks iguais (total {total})"
        cur = r.get_chunks(document_id="reg")
        assert len(cur) == b["count"], "vigentes = chunks da versão 2"
        hits = r.search("prazo de 15 dias para reembolso", top_k=3)
        assert "15 dias" in hits[0]["text"], hits[0]["text"][:80]
        old = r.search("reembolso regra número 3", top_k=10, only_current=False)
        assert any(h["version"] == 1 and "15 dias" not in h["text"] and "reembolso" in h["text"]
                   for h in old), "o texto antigo deve continuar no banco"
        r.add_version("reg", LONG_V2 + "\nArt. 9º Novo.", version=3)
        r.rollback_version("reg", 3)
        assert len(r.get_chunks(document_id="reg")) == b["count"], "rollback volta à v2"

    def t_publish_rollback_real():
        from retriever import DocumentRetriever
        r = DocumentRetriever(db_path=tmp / "chroma")
        store = VersionStore(tmp / "rb.db")
        publish_version(store, r, name="Doc RB", text=LONG_V1, filename="a.pdf", sha256="a")
        before = sorted(c["id"] for c in r.get_chunks(document_id="doc-rb"))

        def boom(*a, **k):
            raise RuntimeError("falha simulada no SQLite")
        store.add_version = boom
        try:
            publish_version(store, r, name="Doc RB", text=LONG_V2, filename="b.pdf", sha256="b")
            raise AssertionError("deveria ter levantado a falha")
        except RuntimeError:
            pass
        after = sorted(c["id"] for c in r.get_chunks(document_id="doc-rb"))
        assert after == before, "o índice deve voltar ao estado anterior"

    def t_publicacoes_simultaneas():
        from concurrent.futures import ThreadPoolExecutor
        store, fr = VersionStore(tmp / "conc.db"), FakeRetriever()
        texts = [V1 + f"\nArt. 9º Versão {i}." for i in range(6)]
        with ThreadPoolExecutor(6) as ex:
            list(ex.map(lambda t: publish_version(store, fr, name="Conc", text=t,
                                                  filename="c.pdf", sha256=sha256_text(t)), texts))
        assert [v.version for v in store.list_versions("conc")] == [1, 2, 3, 4, 5, 6]
        assert sum(1 for v in store.list_versions("conc") if v.is_current) == 1

    def t_retriever_filtros():
        from retriever import DocumentRetriever
        r = DocumentRetriever(db_path=tmp / "chroma")
        r.add_version("outro", "Texto de outro documento sobre prazo de entrega.", version=1)
        only = r.search("prazo", top_k=5, document_id="outro")
        assert only and all(h["document_id"] == "outro" for h in only)
        excl = r.search("prazo", top_k=5, exclude_document_id="outro")
        assert excl and all(h["document_id"] == "reg" for h in excl)

    runner.run("Divisão por artigo",                         t_split_articles)
    runner.run("Citação no meio da frase não vira artigo",   t_split_ignora_citacao)
    runner.run("Numeração repetida (ADCT) não colide",       t_split_numeracao_repetida)
    runner.run("Artigo com letra (12-A)",                    t_split_sufixo_letra)
    runner.run("Estilo CLT 'Art. 4º - O …' não vira 4-O",    t_split_estilo_clt)
    runner.run("Diff de 12 mil palavras sem artigos < 2 s",  t_diff_texto_grande_sem_artigos)
    runner.run("Comparação: alterado, removido, incluído",   t_compare)
    runner.run("Só quebra de linha não conta como mudança",  t_compare_igual)
    runner.run("Texto sem artigos é comparado inteiro",      t_compare_sem_artigos)
    runner.run("Resumo determinístico do diff",              t_as_text)
    runner.run("Identificador do documento (slug)",          t_slugify)
    runner.run("SQLite: autor, data, motivo, SHA-256",       t_store_registro)
    runner.run("SQLite: persiste ao reabrir",                t_store_persiste)
    runner.run("publish_version cria versões sem apagar",    t_publish)
    runner.run("Mesmo arquivo não gera versão nova",         t_publish_mesmo_arquivo)
    runner.run("Vários documentos na mesma base",            t_publish_varios_documentos)
    runner.run("summarize_change não mexe no histórico",     t_summarize_sem_historico)
    runner.run("ChromaDB: versão nova só troca o que mudou", t_retriever_incremental)
    runner.run("ChromaDB: filtro por documento",             t_retriever_filtros)
    runner.run("Falha no SQLite desfaz o índice",            t_publish_rollback_real)
    runner.run("6 publicações simultâneas: versões 1 a 6",   t_publicacoes_simultaneas)

    shutil.rmtree(tmp, ignore_errors=True)
    return runner.summary()
