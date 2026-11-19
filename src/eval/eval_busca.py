"""
eval/eval_busca.py
──────────────────
Mede a busca do chat com perguntas escritas de jeitos diferentes
(eval/parafrases.json): cada pergunta em 3 formas, técnica, leiga e
curta/com erro de digitação. Hit Rate@5 e MRR@5 por forma.

A base junta, como na interface, os exemplos de data/exemplos/ (Regulamento v1
e v2, Contrato Modelo, Manual do Colaborador) e o edital do PPGI, e a busca é
feita em todos os documentos ("Buscar em: todos").

Métodos comparados:
  atual       retriever.search(pergunta), o caminho da interface sem LLM
  atual+llm   pergunta + reescrita da _expand_query do app.py, depois retriever.search
  nova        busca.buscar(retriever, pergunta)            (sem LLM)
  nova+llm    busca.buscar(retriever, pergunta, reescrever=...)

Uso (de dentro de src/, com bancos temporários):
    python eval/eval_busca.py                         # sem LLM, textos .txt e .pdf
    python eval/eval_busca.py --llm                   # inclui os métodos com LLM (GROQ_API_KEY)
    python eval/eval_busca.py --variante pdf --detalhe
    python eval/eval_busca.py --edital caminho.db|.txt|.pdf
    python eval/eval_busca.py --cf88 CF.pdf           # Hit Rate/MRR da CF/88 (atual x nova)

As respostas da LLM ficam em cache (llm_cache.json na pasta da base) para a
avaliação ser repetível e não gastar cota à toa.
"""

import argparse
import json
import os
import sqlite3
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

# Bancos temporários antes de importar o projeto: nada encosta em vector_db/ nem em data/versions.db.
_TMP = Path(os.environ.get("EVAL_BUSCA_DIR") or tempfile.mkdtemp(prefix="eval_busca_"))
os.environ.setdefault("CHROMADB_PATH", str(_TMP / "chroma_padrao"))
os.environ.setdefault("EIXOAI_VERSIONS_DB", str(_TMP / "versions_padrao.db"))

from dotenv import load_dotenv                                     # noqa: E402
load_dotenv(ROOT / ".env")

import busca                                                       # noqa: E402
from ingester import extract_text_from_pdf                         # noqa: E402
from retriever import DocumentRetriever                            # noqa: E402
from versioning import VersionStore, publish_version               # noqa: E402

EXEMPLOS = ROOT / "data" / "exemplos"
PARAFRASES = Path(__file__).resolve().parent / "parafrases.json"
FORMAS = ("tecnica", "leiga", "curta")
EDITAL_NOME = "Edital_Processo_Seletivo_PPGI_2027_1"
# O edital do PPGI não vai no repositório: aponte EVAL_EDITAL_PATH (ou --edital) para
# um .pdf, .txt ou um versions.db que o tenha. Sem ele, as 14 perguntas do edital ficam de fora.
EDITAL_DB_PADRAO = os.environ.get("EVAL_EDITAL_PATH", "")


def norm(s: str) -> str:
    """Minúsculas, sem acento e com espaços normalizados (o edital vem do PDF com espaços duplos)."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return " ".join(s.split())


# ── Textos ───────────────────────────────────────────────────

def texto_edital(caminho: str) -> str | None:
    p = Path(caminho) if caminho else None
    if p is None or not p.is_file():
        return None
    if p.suffix.lower() == ".pdf":
        return extract_text_from_pdf(str(p))
    if p.suffix.lower() in (".txt", ".md"):
        return p.read_text(encoding="utf-8")
    # SQLite da interface: só leitura, sem tocar no arquivo.
    con = sqlite3.connect(f"file:{p.as_posix()}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT v.text FROM versions v JOIN documents d ON d.id = v.document_id "
            "WHERE d.name = ? ORDER BY v.version DESC LIMIT 1", (EDITAL_NOME,)).fetchone()
    finally:
        con.close()
    return row[0] if row else None


def textos_exemplos(variante: str) -> dict:
    """Texto de cada arquivo de exemplo, lido do .txt ou extraído do .pdf (como no upload)."""
    out = {}
    for stem in ("regulamento_v1", "regulamento_v2", "contrato_modelo", "manual"):
        if variante == "pdf":
            out[stem] = extract_text_from_pdf(str(EXEMPLOS / "pdf" / f"{stem}.pdf"))
        else:
            out[stem] = (EXEMPLOS / f"{stem}.txt").read_text(encoding="utf-8")
    return out


def montar_base(pasta: Path, variante: str, edital: str | None) -> DocumentRetriever:
    """Publica os documentos como a aba Versões faria. Reaproveita a base se já existir."""
    pasta.mkdir(parents=True, exist_ok=True)
    retriever = DocumentRetriever(db_path=pasta / "chroma")
    if retriever.collection.count():
        return retriever
    store = VersionStore(pasta / "versions.db")
    t = textos_exemplos(variante)

    def pub(nome, texto, arquivo):
        import hashlib
        publish_version(store, retriever, name=nome, text=texto, filename=arquivo,
                        sha256=hashlib.sha256(texto.encode()).hexdigest())

    pub("Regulamento Interno", t["regulamento_v1"], f"regulamento_v1.{variante}")
    pub("Contrato Modelo", t["contrato_modelo"], f"contrato_modelo.{variante}")
    pub("Manual do Colaborador", t["manual"], f"manual.{variante}")
    pub("Regulamento Interno", t["regulamento_v2"], f"regulamento_v2.{variante}")
    if edital:
        pub(EDITAL_NOME, edital, f"{EDITAL_NOME}.pdf")
    return retriever


# ── LLM (opcional, com cache) ────────────────────────────────

class CacheLLM:
    def __init__(self, arquivo: Path, modelo: str):
        self.arquivo, self.modelo = arquivo, modelo
        self.dados = json.loads(arquivo.read_text(encoding="utf-8")) if arquivo.exists() else {}
        self._chain = None
        self.falhas = 0

    def _llm(self):
        if self._chain is None:
            from llm_chain import LLMChain
            self._chain = LLMChain(model=self.modelo, stream=False)
        return self._chain

    def _get(self, tipo: str, pergunta: str, gerar):
        chave = f"{self.modelo}|{tipo}|{pergunta}"
        if chave not in self.dados:
            try:
                self.dados[chave] = gerar()
            except Exception as e:                       # noqa: BLE001
                self.falhas += 1
                print(f"    [llm] falhou ({type(e).__name__}): {e}")
                return None
            self.arquivo.write_text(json.dumps(self.dados, ensure_ascii=False, indent=1),
                                    encoding="utf-8")
        return self.dados[chave]

    def expand_query_app(self, pergunta: str) -> str:
        """Reproduz a _expand_query do app.py (mesmo prompt), sem importar a interface."""
        def gerar():
            chain = self._llm()
            prompt = (
                "Reescreva a pergunta abaixo adicionando sinônimos e termos jurídicos "
                "da Constituição Federal brasileira que possam aparecer nos trechos relevantes. "
                "Responda APENAS com a pergunta reescrita, sem explicações.\n\n"
                f"Pergunta: {pergunta}"
            )
            resp = chain.generate_response(prompt)
            chain.clear_history()
            return resp
        exp = self._get("expand_app", pergunta, gerar)
        return f"{pergunta} {exp}" if exp else pergunta

    def reescrever(self, pergunta: str):
        return self._get("reescrever", pergunta,
                         lambda: self._llm().reescrever_pergunta(pergunta))


# ── Métricas ─────────────────────────────────────────────────

def relevante(res: dict, esperado: list) -> bool:
    t = norm(res["text"])
    return any(res.get("document_id") == e["documento"] and norm(e["contem"]) in t
               for e in esperado)


def avaliar(metodos: dict, perguntas: list, k: int, detalhe: bool) -> dict:
    """{metodo: {forma: {"hit": [...], "rr": [...], "ms": [...]}}} e as falhas."""
    out = {m: {f: {"hit": [], "rr": [], "ms": [], "grupo": []} for f in FORMAS} for m in metodos}
    falhas = {m: [] for m in metodos}
    for p in perguntas:
        grupo = "edital" if p["id"].startswith("edi-") else "exemplos"
        for forma in FORMAS:
            q = p["formas"][forma]
            for nome, fn in metodos.items():
                t0 = time.perf_counter()
                res = fn(q)[:k]
                ms = (time.perf_counter() - t0) * 1000
                rel = [relevante(r, p["esperado"]) for r in res]
                rr = next((1 / (i + 1) for i, x in enumerate(rel) if x), 0.0)
                d = out[nome][forma]
                d["hit"].append(any(rel)); d["rr"].append(rr); d["ms"].append(ms)
                d["grupo"].append(grupo)
                if not any(rel):
                    falhas[nome].append((p["id"], forma, q,
                                         [f"{r['document_id']}#{r['chunk']}" for r in res]))
    if detalhe:
        for nome, fs in falhas.items():
            print(f"\n  Falhas de {nome} ({len(fs)}):")
            for pid, forma, q, got in fs:
                print(f"    {pid:24s} {forma:8s} {q!r} -> {got}")
    return out


def _media(xs):
    return sum(xs) / len(xs) if xs else 0.0


def tabela(out: dict, titulo: str, grupo: str | None = None) -> dict:
    print(f"\n{titulo}")
    print(f"  {'método':10s} | " + " | ".join(f"{f:^15s}" for f in FORMAS) + f" | {'total':^15s} | ms/consulta")
    print(f"  {'':10s} | " + " | ".join(f"{'Hit@5':>7s} {'MRR':>7s}" for _ in range(4)) + " |")
    resumo = {}
    for nome, por_forma in out.items():
        cel, H, R, MS = [], [], [], []
        resumo[nome] = {}
        for f in FORMAS:
            d = por_forma[f]
            idx = [i for i, g in enumerate(d["grupo"]) if grupo is None or g == grupo]
            h = [d["hit"][i] for i in idx]; r = [d["rr"][i] for i in idx]
            H += h; R += r; MS += [d["ms"][i] for i in idx]
            cel.append(f"{_media(h):7.1%} {_media(r):7.3f}")
            resumo[nome][f] = {"hit_rate": _media(h), "mrr": _media(r), "n": len(h)}
        resumo[nome]["total"] = {"hit_rate": _media(H), "mrr": _media(R), "n": len(H),
                                 "ms": _media(MS)}
        print(f"  {nome:10s} | " + " | ".join(cel) + f" | {_media(H):7.1%} {_media(R):7.3f} | {_media(MS):6.0f}")
    return resumo


def checar_gabarito(retriever: DocumentRetriever, perguntas: list) -> list:
    """Cada 'contem' precisa existir em algum trecho vigente; senão a pergunta está mal escrita."""
    chunks = retriever.get_chunks()
    faltando = []
    for p in perguntas:
        if not any(relevante({"text": c["text"], "document_id": c.get("document_id")}, p["esperado"])
                   for c in chunks):
            faltando.append(p["id"])
    return faltando


# ── CF/88 ────────────────────────────────────────────────────

def avaliar_cf88(pdf: str, pasta: Path, k: int, llm: "CacheLLM | None" = None) -> dict:
    """Mesma regra do RAGEvaluator (keywords, limiar 0,3), para o caminho atual e o novo."""
    from eval.qa_pairs import CF88_QA_PAIRS
    retriever = DocumentRetriever(db_path=pasta / "chroma_cf88")
    doc_id = "cf88"
    if not retriever.collection.count():
        retriever.add_document(doc_id, extract_text_from_pdf(pdf), source=Path(pdf).name)
    n_chunks = retriever.collection.count()

    def rel(text, kws):
        tl = text.lower()
        return sum(1 for kw in kws if kw.lower() in tl) / len(kws) >= 0.3 if kws else False

    metodos = {
        "atual": lambda q: retriever.search(q, top_k=k, rerank=True, document_id=doc_id),
        "nova": lambda q: busca.buscar(retriever, q, document_id=doc_id, k=k),
    }
    if llm:
        metodos["nova+llm"] = lambda q: busca.buscar(retriever, q, document_id=doc_id, k=k,
                                                      reescrever=llm.reescrever)
    out = {}
    for nome, fn in metodos.items():
        hits, rrs, por_q = [], [], []
        for p in CF88_QA_PAIRS:
            res = fn(p["query"])[:k]
            r = [rel(x["text"], p.get("keywords", [])) for x in res]
            hits.append(any(r)); rrs.append(next((1 / (i + 1) for i, x in enumerate(r) if x), 0.0))
            por_q.append(any(r))
        out[nome] = {"hit_rate": _media(hits), "mrr": _media(rrs), "n": len(hits), "por_pergunta": por_q}
    a, b = out["atual"]["por_pergunta"], out["nova"]["por_pergunta"]
    print(f"\nCF/88 ({len(a)} perguntas, {n_chunks} chunks, Hit@{k} pela regra do RAGEvaluator):")
    for nome in metodos:
        print(f"  {nome:8s} Hit Rate {out[nome]['hit_rate']:.1%}  MRR {out[nome]['mrr']:.3f}")
    ganhou = [i for i in range(len(a)) if b[i] and not a[i]]
    perdeu = [i for i in range(len(a)) if a[i] and not b[i]]
    print(f"  perguntas que passaram a acertar: {ganhou}; que deixaram de acertar: {perdeu}")
    out["chunks"] = n_chunks
    return out


# ── CLI ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Avalia a busca com paráfrases")
    ap.add_argument("--variante", choices=["txt", "pdf", "ambas"], default="ambas")
    ap.add_argument("--llm", action="store_true", help="inclui os métodos com LLM (GROQ_API_KEY)")
    ap.add_argument("--modelo", default=None, help="modelo do Groq (padrão: DEFAULT_MODEL)")
    ap.add_argument("--edital", default=EDITAL_DB_PADRAO, help="edital do PPGI: .db da interface, .txt ou .pdf")
    ap.add_argument("--base", default=None, help="pasta das bases (reaproveitada entre execuções)")
    ap.add_argument("--metodos", default=None, help="lista separada por vírgula")
    ap.add_argument("--cf88", default=None, help="PDF da CF/88 para comparar atual x nova")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--detalhe", action="store_true", help="lista as perguntas que falharam")
    ap.add_argument("--saida", default=None, help="salva o resumo em JSON")
    args = ap.parse_args()

    pasta = Path(args.base) if args.base else _TMP
    dados = json.loads(PARAFRASES.read_text(encoding="utf-8"))
    edital = texto_edital(args.edital)
    perguntas = dados["perguntas"]
    if not edital:
        print(f"Edital não encontrado em {args.edital}: as perguntas do edital ficam de fora.")
        perguntas = [p for p in perguntas if not p["id"].startswith("edi-")]

    llm = None
    if args.llm:
        from llm_chain import DEFAULT_MODEL
        llm = CacheLLM(pasta / "llm_cache.json", args.modelo or DEFAULT_MODEL)

    resumo = {"k": args.k, "perguntas": len(perguntas), "variantes": {}}
    variantes = ["txt", "pdf"] if args.variante == "ambas" else [args.variante]
    for var in variantes:
        retriever = montar_base(pasta / f"base_{var}", var, edital)
        falt = checar_gabarito(retriever, perguntas)
        if falt:
            print(f"AVISO: trecho esperado não existe na base ({var}): {falt}")
        k = args.k
        metodos = {
            "atual": lambda q, r=retriever: r.search(q, top_k=k, rerank=True),
            "nova": lambda q, r=retriever: busca.buscar(r, q, k=k),
        }
        if llm:
            metodos["atual+llm"] = lambda q, r=retriever: r.search(llm.expand_query_app(q), top_k=k, rerank=True)
            metodos["nova+llm"] = lambda q, r=retriever: busca.buscar(r, q, k=k, reescrever=llm.reescrever)
        if args.metodos:
            metodos = {m: metodos[m] for m in args.metodos.split(",") if m in metodos}

        n = len(perguntas)
        print(f"\n=== Base com textos .{var}: {retriever.collection.count()} chunks no índice, "
              f"{n} perguntas x 3 formas ===")
        out = avaliar(metodos, perguntas, k, args.detalhe)
        r_all = tabela(out, f"Todas as perguntas ({n}), busca em todos os documentos")
        r_ex = tabela(out, "Só exemplos (Regulamento, Contrato, Manual)", "exemplos")
        r_ed = tabela(out, "Só edital do PPGI", "edital") if edital else {}
        resumo["variantes"][var] = {"todas": r_all, "exemplos": r_ex, "edital": r_ed}

    if args.cf88:
        resumo["cf88"] = avaliar_cf88(args.cf88, pasta, args.k, llm)
    if llm and llm.falhas:
        print(f"\nChamadas à LLM que falharam: {llm.falhas}")
    if args.saida:
        Path(args.saida).write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResumo salvo em {args.saida}")


if __name__ == "__main__":
    main()
