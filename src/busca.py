"""
busca.py
────────
Busca do chat que entende a mesma pergunta escrita de jeitos diferentes:
técnica ou leiga, curta ("reembolso?") ou longa, com ou sem acento, com
erro de digitação.

Por que existe (números em eval/eval_busca.py): com a busca só densa, a mesma
pergunta acertava 88-90% na forma técnica e só 63-65% nas formas leiga e curta.
O e5 dá notas muito próximas (0,80-0,85) para quase todo trecho, e um documento
grande (o edital, 163 trechos) ocupa o top-5 de perguntas sobre documentos
pequenos. Uma palavra exata da pergunta ("ponto", "atestado", "estagiário") não
pesava nada.

Como funciona:
  1. Busca densa (e5, prefixo "query: ") e busca lexical BM25 em todos os trechos
     vigentes. O BM25 compara palavras sem acento e sem maiúsculas, com um
     radical simples (as 6 primeiras letras), e corrige erro de digitação
     trocando a palavra que não existe na base pela mais parecida que existe.
     O peso da lista lexical na fusão cai quando boa parte da pergunta não
     existe na base (ex.: 'folhas' num edital que só fala em 'páginas').
  2. Opcional: uma função `reescrever(pergunta)` (ex.: LLMChain.reescrever_pergunta)
     devolve variantes da pergunta (técnica, leiga, palavras-chave). Cada
     variante também é buscada. Ela roda em paralelo, com tempo limite, e
     qualquer erro é ignorado: sem chave do Groq, a busca funciona igual.
  3. As listas são fundidas por RRF (Reciprocal Rank Fusion): cada trecho soma
     peso / (60 + posição) em cada lista em que aparece.

Uso:
    from busca import buscar
    results = buscar(retriever, pergunta, document_id=None, k=5,
                     reescrever=llm.reescrever_pergunta if llm else None)

O resultado tem o mesmo formato de DocumentRetriever.search (id, text,
distance, relevance, source, page, chunk, article, document_id, version, rank),
então format_context e o rastreamento da interface continuam iguais.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import math
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Callable, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)

# ── Parâmetros ───────────────────────────────────────────────

CANDIDATOS = 30          # quantos trechos cada lista (densa e lexical) leva para a fusão
RRF_K = 60               # constante do RRF (valor usual da literatura)
PESO_DENSA = 1.0
PESO_LEXICAL = 1.0
PESO_VARIANTE = 0.5      # variantes da LLM pesam menos que a pergunta original
TEMPO_REESCRITA = 8.0    # segundos de espera pela reescrita antes de seguir sem ela
MAX_VARIANTES = 3
RADICAL = 6              # letras mantidas no radical (truncamento simples para o português)
BM25_K1, BM25_B = 1.2, 0.75
CORRECAO_MIN = 0.85     # semelhança mínima (difflib) para trocar palavra inexistente pela da base
PESO_POR_COBERTURA = True  # lista lexical pesa a parte da pergunta que existe na base (ver cobertura)

Reescrever = Callable[[str], Union[str, Sequence[str], None]]

_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="reescrita")


# ── Texto ────────────────────────────────────────────────────

_STOP = set("""
a ao aos as ate com como da das de dela dele deles demais depois do dos e ela elas ele eles em
entao entre era essa essas esse esses esta estao estas este estes eu foi for ha isso isto ja la
lhe lhes mais mas me mesmo meu meus minha minhas muito na nas nao nem no nos nossa nosso num numa
o os ou para pela pelas pelo pelos por porque qual quais quando quanto quantos quantas quanta que
quem se sem ser seu seus sua suas so tambem te tem tenho ter teu tu um uma umas uns voce voces
vai vou sobre preciso posso pode podem devo deve existe existem algum alguma alguns algumas
onde coisa coisas
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


def normalizar(texto: str) -> str:
    """Minúsculas e sem acento: 'Férias' → 'ferias', 'inscriçao' → 'inscricao'."""
    texto = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in texto if not unicodedata.combining(c)).lower()


def palavras(texto: str) -> List[str]:
    """Palavras normalizadas, sem stopwords."""
    return [t for t in _TOKEN.findall(normalizar(texto)) if t not in _STOP]


def radical(palavra: str) -> str:
    """Truncamento: 'reembolsadas' e 'reembolso' → 'reembo'. Números ficam inteiros."""
    return palavra if palavra.isdigit() else palavra[:RADICAL]


# ── Índice lexical (BM25) ────────────────────────────────────

class IndiceLexical:
    """BM25 sobre os trechos vigentes. Reconstruído quando o conjunto de trechos muda."""

    def __init__(self, chunks: List[Dict]):
        self.ids = [c["id"] for c in chunks]
        self.meta = {c["id"]: c for c in chunks}
        self.tf: List[Counter] = []
        self.len: List[int] = []
        df: Counter = Counter()
        vocab: set = set()
        for c in chunks:
            ws = palavras(c["text"])
            vocab.update(ws)
            rs = Counter(radical(w) for w in ws)
            self.tf.append(rs)
            self.len.append(sum(rs.values()))
            df.update(rs.keys())
        n = len(chunks)
        self.n = n
        self.media_len = (sum(self.len) / n) if n else 1.0
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.vocab = vocab
        # Para corrigir erro de digitação sem varrer o vocabulário inteiro.
        self.por_inicial: Dict[str, List[str]] = defaultdict(list)
        for w in vocab:
            if len(w) >= 4 and not w.isdigit():
                self.por_inicial[w[0]].append(w)

    def corrigir(self, palavra: str) -> List[str]:
        """Palavras da base mais parecidas com uma que não existe nela ('lates' → 'lattes')."""
        if palavra in self.vocab or len(palavra) < 4 or palavra.isdigit():
            return [palavra]
        cands = [w for w in self.por_inicial.get(palavra[0], ()) if abs(len(w) - len(palavra)) <= 2]
        perto = difflib.get_close_matches(palavra, cands, n=2, cutoff=CORRECAO_MIN)
        return perto or [palavra]

    def termos(self, consulta: str) -> List[str]:
        out: List[str] = []
        for w in palavras(consulta):
            for c in self.corrigir(w):
                r = radical(c)
                if r not in out:
                    out.append(r)
        return out

    def cobertura(self, consulta: str) -> float:
        """
        Parte do 'peso' (IDF) da pergunta que existe na base, de 0 a 1. Palavra
        que não existe conta com o maior IDF. Em 'meu projeto pode ter quantas
        folhas?' só 'projeto' existe, e é comum no edital: a lista lexical vira
        ruído, então pesa menos na fusão.
        """
        termos = self.termos(consulta)
        if not termos or not self.idf:
            return 0.0
        teto = max(self.idf.values())
        total = sum(self.idf.get(t, teto) for t in termos)
        achado = sum(self.idf[t] for t in termos if t in self.idf)
        return achado / total if total else 0.0

    def buscar(self, consulta: str, n: int, filtro: Optional[Callable[[Dict], bool]] = None) -> List[str]:
        termos = [t for t in self.termos(consulta) if t in self.idf]
        if not termos:
            return []
        notas = []
        for i, tf in enumerate(self.tf):
            s = 0.0
            for t in termos:
                f = tf.get(t)
                if f:
                    s += self.idf[t] * f * (BM25_K1 + 1) / (
                        f + BM25_K1 * (1 - BM25_B + BM25_B * self.len[i] / self.media_len))
            if s > 0 and (filtro is None or filtro(self.meta[self.ids[i]])):
                notas.append((s, i))
        notas.sort(reverse=True)
        return [self.ids[i] for _, i in notas[:n]]


_CACHE: Dict[int, tuple] = {}
_CACHE_LOCK = threading.Lock()


def _indice(retriever) -> IndiceLexical:
    """Índice do retriever, refeito só quando os trechos vigentes mudam (versão nova, upload)."""
    atuais = retriever.collection.get(where={"is_current": {"$eq": True}}, include=[])["ids"]
    assinatura = hashlib.md5("|".join(sorted(atuais)).encode()).hexdigest()
    with _CACHE_LOCK:
        hit = _CACHE.get(id(retriever))
        if hit and hit[0] is retriever and hit[1] == assinatura:
            return hit[2]
        chunks = retriever.get_chunks(only_current=True)
        idx = IndiceLexical(chunks)
        _CACHE[id(retriever)] = (retriever, assinatura, idx)
        return idx


# ── Reescrita (opcional) ─────────────────────────────────────

def _variantes(resp: Union[str, Sequence[str], None], pergunta: str) -> List[str]:
    if not resp:
        return []
    itens = resp.splitlines() if isinstance(resp, str) else list(resp)
    out, vistos = [], {normalizar(pergunta).strip()}
    for v in itens:
        v = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", str(v)).strip().strip('"').strip()
        if 2 <= len(v) <= 300 and normalizar(v) not in vistos:
            vistos.add(normalizar(v))
            out.append(v)
    return out[:MAX_VARIANTES]


def _iniciar_reescrita(reescrever: Optional[Reescrever], pergunta: str):
    if reescrever is None:
        return None
    try:
        return _POOL.submit(reescrever, pergunta)
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"Reescrita não iniciada: {e}")
        return None


def _colher_reescrita(fut, pergunta: str, timeout: float) -> List[str]:
    if fut is None:
        return []
    try:
        return _variantes(fut.result(timeout=timeout), pergunta)
    except FuturesTimeout:
        logger.warning(f"Reescrita passou de {timeout:.0f} s; seguindo só com a pergunta original")
    except Exception as e:                                   # noqa: BLE001
        logger.warning(f"Reescrita falhou, seguindo só com a pergunta original: {e}")
    return []


# ── Busca ────────────────────────────────────────────────────

def _densa(retriever, consultas: List[str], where, n: int):
    """Uma lista de ids por consulta, mais os embeddings (para calcular a relevância exibida)."""
    embs = retriever.model.encode(
        [f"query: {retriever._normalize_text(q)}" for q in consultas],
        normalize_embeddings=True, show_progress_bar=False,
    )
    total = retriever.collection.count()
    if not total:
        return [[] for _ in consultas], embs, {}
    kw = {"where": where} if where else {}
    res = retriever.collection.query(query_embeddings=[e.tolist() for e in embs],
                                     n_results=min(n, total), **kw)
    listas, info = [], {}
    for qi in range(len(consultas)):
        ids = res["ids"][qi] if res["ids"] else []
        listas.append(list(ids))
        for cid, doc, dist, meta in zip(ids, res["documents"][qi], res["distances"][qi],
                                        res["metadatas"][qi]):
            if qi == 0 or cid not in info:
                info.setdefault(cid, (doc, meta, float(dist) if qi == 0 else None))
    return listas, embs, info


def _formatar(cid: str, texto: str, meta: Dict, dist: float, rank: int) -> Dict:
    meta = meta or {}
    return {
        "id": cid,
        "text": texto,
        "distance": float(dist),
        "relevance": 1 - float(dist),
        "source": meta.get("source"),
        "page": meta.get("page"),
        "chunk": meta.get("chunk"),
        "article": meta.get("article", "N/A"),
        "document_id": meta.get("document_id"),
        "version": meta.get("version"),
        "rank": rank,
    }


def buscar(
    retriever,
    pergunta: str,
    *,
    document_id: Optional[str] = None,
    k: int = 5,
    reescrever: Optional[Reescrever] = None,
    timeout_reescrita: float = TEMPO_REESCRITA,
) -> List[Dict]:
    """
    Busca híbrida (densa + BM25, fusão RRF) na versão vigente dos documentos.

    `document_id` restringe a um documento (None = todos). `reescrever` é
    opcional: recebe a pergunta e devolve variantes (lista ou texto com uma
    por linha); falha ou demora acima de `timeout_reescrita` são ignoradas.
    Devolve até `k` resultados no formato de DocumentRetriever.search.
    """
    pergunta = (pergunta or "").strip()
    if not pergunta:
        return []
    fut = _iniciar_reescrita(reescrever, pergunta)
    try:
        where = retriever._build_where(True, document_id, None)
        filtro = (lambda m: m.get("document_id") == document_id) if document_id else None

        # A pergunta original é buscada enquanto a LLM reescreve.
        listas_d, embs, info = _densa(retriever, [pergunta], where, CANDIDATOS)
        idx = _indice(retriever)
        def peso_lex(consulta: str) -> float:
            return PESO_LEXICAL * (idx.cobertura(consulta) if PESO_POR_COBERTURA else 1.0)

        listas = [(PESO_DENSA, listas_d[0]),
                  (peso_lex(pergunta), idx.buscar(pergunta, CANDIDATOS, filtro))]

        variantes = _colher_reescrita(fut, pergunta, timeout_reescrita)
        if variantes:
            logger.info(f"Variantes da pergunta: {variantes}")
            ld, _, info_v = _densa(retriever, variantes, where, CANDIDATOS)
            for cid, v in info_v.items():
                info.setdefault(cid, (v[0], v[1], None))
            for v, lista in zip(variantes, ld):
                listas.append((PESO_DENSA * PESO_VARIANTE, lista))
                listas.append((peso_lex(v) * PESO_VARIANTE, idx.buscar(v, CANDIDATOS, filtro)))

        notas: Dict[str, float] = defaultdict(float)
        for peso, lista in listas:
            for pos, cid in enumerate(lista):
                notas[cid] += peso / (RRF_K + pos + 1)
        ordem = sorted(notas, key=lambda c: -notas[c])

        # Relevância exibida = semelhança do e5 com a pergunta original, também
        # para os trechos que vieram só do BM25 ou das variantes.
        sem_dist = [c for c in ordem[: k * 3] if info.get(c, (None, None, None))[2] is None]
        if sem_dist:
            got = retriever.collection.get(ids=sem_dist, include=["embeddings", "documents", "metadatas"])
            q = embs[0]
            for cid, emb, doc, meta in zip(got["ids"], got["embeddings"], got["documents"],
                                           got["metadatas"]):
                dist = 1 - float(sum(a * b for a, b in zip(q, emb)))
                info[cid] = (doc, meta, dist)

        out, vistos = [], set()
        for cid in ordem:
            if cid not in info or info[cid][2] is None:
                continue
            doc, meta, dist = info[cid]
            h = hashlib.md5(doc[:100].encode()).hexdigest()   # mesmo critério do rerank do search
            if h in vistos:
                continue
            vistos.add(h)
            out.append(_formatar(cid, doc, meta, dist, len(out) + 1))
            if len(out) == k:
                break
        return out
    except Exception as e:                                   # noqa: BLE001
        # Nunca derruba o chat: na pior hipótese, volta à busca densa original.
        logger.warning(f"Busca híbrida falhou ({e}); usando a busca densa")
        return retriever.search(pergunta, top_k=k, rerank=True, document_id=document_id)
