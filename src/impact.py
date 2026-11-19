"""
impact.py
─────────
Análise de impacto: quando um documento muda, quais trechos de OUTROS
documentos da base dependem dos artigos alterados ou removidos.

Dois níveis de confiança:
  • citação direta (certo)  — o trecho cita o artigo: "art. 12 do Regulamento",
                               "arts. 3º e 9º do Regulamento Interno", "Regulamento, art. 12"
  • sugestão (possível)     — o trecho não cita o artigo, mas trata do mesmo assunto.
                               Passa por três filtros (ver analyze_impact):
                                 1. o documento do trecho é do mesmo tema do documento alterado;
                                 2. o artigo antigo mais parecido com a frase é um dos que mudaram;
                                 3. (opcional) uma LLM confirma que o trecho depende do artigo.

As citações são procuradas no texto inteiro da versão vigente de cada documento
(guardado no SQLite); as sugestões, nos chunks vigentes do ChromaDB.
"""

from __future__ import annotations

import bisect
import hashlib
import logging
import math
import os
import re
import threading
import unicodedata
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from versioning import VersionDiff, VersionStore

logger = logging.getLogger(__name__)

# Piso de semelhança frase × artigo. O multilingual-e5 põe quase qualquer frase
# jurídica entre 0,85 e 0,92 de qualquer artigo, então este piso sozinho não
# separa nada; quem separa são o filtro de tema (MIN_DOC_SIMILARITY) e a regra
# do "artigo mais parecido" em analyze_impact.
MIN_RELEVANCE = float(os.environ.get("IMPACT_MIN_RELEVANCE", "0.85"))

# Filtro de tema, no nível do documento. A nota de tema é a média de duas vistas:
#   • o começo: os primeiros TOPIC_HEAD_CHARS caracteres de cada documento
#     (título e primeiros parágrafos, onde o documento diz do que trata);
#   • as partes: cada chunk do documento candidato fica com o chunk mais parecido
#     do documento alterado, e conta a média do quarto mais parecido (o quarto,
#     e não todos, para um manual grande que só em parte trata do mesmo assunto
#     não ser descartado). Usa os embeddings já guardados no ChromaDB.
# Medido em 2026-11-19 (ver docs/VERSIONAMENTO.md): documentos do mesmo tema do
# Regulamento ficaram entre 0,880 e 0,940; de outro tema, entre 0,818 e 0,862.
MIN_DOC_SIMILARITY = float(os.environ.get("IMPACT_MIN_DOC_SIMILARITY", "0.87"))
TOPIC_HEAD_CHARS = 2000

SUGGESTIONS_PER_ARTICLE = 3

# Verificação opcional por LLM: no máximo VERIFY_MAX chamadas, VERIFY_WORKERS
# por vez (o plano gratuito do Groq dá 8 mil tokens por minuto, e cada chamada
# gasta perto de 600), com VERIFY_TIMEOUT segundos para todas. Quem não
# responder a tempo (ou der erro) fica como sugestão sem confirmação, igual a
# quando não há LLM.
VERIFY_MAX = 8
VERIFY_WORKERS = 2
VERIFY_TIMEOUT = float(os.environ.get("IMPACT_VERIFY_TIMEOUT", "30"))

# verificar(artigo_antes, artigo_depois, trecho) -> (depende, motivo)
Verifier = Callable[[str, str, str], Tuple[bool, str]]


# ── Normalização com mapeamento 1:1 de posição ────────────────

def _fold(text: str) -> str:
    """
    Minúsculas e sem acento, mantendo o mesmo comprimento do original
    (cada caractere vira exatamente um), para recortar o trecho certo depois.
    'º' vira 'o'.
    """
    out = []
    for ch in text:
        base = "".join(c for c in unicodedata.normalize("NFKD", ch) if not unicodedata.combining(c))
        out.append((base[:1] or ch).lower())
    return "".join(out)


_CIT_HEAD = re.compile(r"\b(?:arts?\.|artigos?\b)\s*")
_NUM = re.compile(r"(\d+)(?:\s*o(?![a-z])|\s*°)?(?:-([a-z])(?![a-z]))?")
_SEP = re.compile(r"\s*(,|\be\b|\ba\b)\s*")
_CONNECTOR = re.compile(r"\b(?:do|da|no|na)\s+")
# O que pode vir entre o número e o "do Regulamento": "§ 2º", "inciso II",
# "parágrafo único", "alínea b", "caput". Qualquer outra palavra ("30 dias do
# Regulamento", "Os prazos previstos no Regulamento") desfaz a citação.
_QUALIFIERS = re.compile(
    r"(?:[\s,]|§+\s*\d+o?|paragrafos?\s+(?:unico|\d+o?)|incisos?\s+[ivxlc]+(?![a-z])"
    r"|alineas?\s+[a-z](?![a-z])|caput)*"
)
# "Art." maiúsculo no começo da linha é o cabeçalho de um artigo do próprio
# documento, não uma citação (mesma âncora de retriever.ARTICLE_START).
_OWN_HEADER = re.compile(r"(?m)^[ \t]*Art\.")


@dataclass
class CitationMatch:
    articles: List[str]   # chaves de artigo citadas ("12", "12-A")
    start: int
    end: int


def _parse_numbers(s: str, pos: int) -> Tuple[List[str], int]:
    """Lê '12', '3º e 9º', '3º a 5º', '12-A' a partir de pos."""
    keys: List[str] = []
    m = _NUM.match(s, pos)
    if not m:
        return keys, pos
    keys.append(m.group(1) + (f"-{m.group(2).upper()}" if m.group(2) else ""))
    end = m.end()
    while True:
        sep = _SEP.match(s, end)
        if not sep:
            break
        nxt = _NUM.match(s, sep.end())
        if not nxt:
            break
        key = nxt.group(1) + (f"-{nxt.group(2).upper()}" if nxt.group(2) else "")
        if sep.group(1) == "a" and keys[-1].isdigit() and key.isdigit():
            keys.extend(str(n) for n in range(int(keys[-1]) + 1, int(key) + 1))
        else:
            keys.append(key)
        end = nxt.end()
    return keys, end


def find_citations(text: str, aliases: Iterable[str]) -> List[CitationMatch]:
    """
    Citações de artigos de um documento identificado pelos `aliases`
    ("Regulamento Interno", "Regulamento"). Aceita:
      art. 12 do Regulamento | artigo 12, § 2º, do Regulamento Interno
      arts. 3º e 9º do Regulamento | Regulamento Interno, art. 12
    Recusa "art. 12 do Código Civil e o Regulamento": vale o primeiro
    documento nomeado depois do número.
    """
    folded = _fold(text)
    al = sorted({_fold(a.strip()) for a in aliases if a and a.strip()}, key=len, reverse=True)
    if not al:
        return []

    own_headers = {m.end() - 4 for m in _OWN_HEADER.finditer(text)}

    out: List[CitationMatch] = []
    for head in _CIT_HEAD.finditer(folded):
        if head.start() in own_headers:
            continue
        keys, end = _parse_numbers(folded, head.end())
        if not keys:
            continue

        # Depois do número: até o próximo "art", ";" ou fim de frase.
        tail = folded[end:end + 120]
        cut = re.search(r"\bart(?:igo)?s?\b|;|\.\s|\n\s*\n", tail)
        if cut:
            tail = tail[:cut.start()]
        conn = _CONNECTOR.search(tail)
        after_ok = (
            bool(conn)
            and _QUALIFIERS.fullmatch(tail[:conn.start()]) is not None
            and any(re.match(re.escape(a) + r"(?![a-z])", tail[conn.end():]) for a in al)
        )

        # Antes do "art.": "Regulamento Interno, art. 12".
        before = folded[max(0, head.start() - 60):head.start()]
        before_ok = any(re.search(re.escape(a) + r"\s*,?\s*$", before) for a in al)

        if after_ok or before_ok:
            out.append(CitationMatch(keys, head.start(), end))
    return out


def _snippet(text: str, start: int, end: int, width: int = 110) -> str:
    a = max(0, start - width)
    b = min(len(text), end + width)
    s = " ".join(text[a:b].split())
    return ("…" if a > 0 else "") + s + ("…" if b < len(text) else "")


# Fim de frase: pontuação seguida de espaço, menos a de "art." e "arts.", ou
# linha em branco. Quebra de linha simples não conta: no texto extraído do PDF
# ela aparece no meio da frase.
_SENT_END = re.compile(r"(?<![Aa]rt)(?<![Aa]rts)[.;!?](?=\s)|\n\s*\n")


def _sentence(text: str, start: int, end: int, reach: int = 240) -> str:
    """A frase em que está a citação (até `reach` caracteres para cada lado)."""
    a0 = max(0, start - reach)
    ends = [m.end() for m in _SENT_END.finditer(text, a0, start)]
    a = ends[-1] if ends else a0
    m = _SENT_END.search(text, end, min(len(text), end + reach))
    b = m.end() if m else min(len(text), end + reach)
    out = " ".join(text[a:b].split())
    return ("…" if a == a0 and a > 0 else "") + out + ("…" if not m and b < len(text) else "")


def _base_key(key: str) -> str:
    return key.split("#", 1)[0]


# ── Relatório ─────────────────────────────────────────────────

@dataclass
class DirectCitation:
    document_id: str
    document_name: str
    article: str          # artigo do documento alterado que é citado
    snippet: str
    location: str         # artigo/trecho do documento que cita
    sentence: str = ""    # só a frase da citação, para mostrar na tela


@dataclass
class Suggestion:
    document_id: str
    document_name: str
    article: str
    chunk_id: str
    text: str
    relevance: float               # semelhança frase × artigo antigo
    topic: float = 0.0             # semelhança de tema entre os documentos
    verified: Optional[bool] = None  # True: a LLM confirmou; None: não foi verificada
    reason: str = ""               # motivo dado pela LLM


@dataclass
class ImpactReport:
    document_id: str
    document_name: str
    affected_articles: List[str]
    direct: List[DirectCitation] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)
    structured: bool = True
    removed_articles: List[str] = field(default_factory=list)
    # Documentos descartados no filtro de tema (nome -> semelhança), para conferência.
    off_topic: Dict[str, float] = field(default_factory=dict)
    # Sugestões que a LLM descartou (com o motivo), para conferência.
    rejected: List[Suggestion] = field(default_factory=list)
    llm_used: bool = False

    @property
    def total(self) -> int:
        return len(self.direct) + len(self.suggestions)


_HEADING = re.compile(
    r"(?m)^[ \t]*(?:Art\.\s*\d+[^\s]*|CL[ÁA]USULA\s+\S+|\d+(?:\.\d+)*[.)]?\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ][^\n]{0,60})"
)


class _Locator:
    """
    Título mais próximo antes de uma posição: 'Art. 4º', 'CLÁUSULA QUINTA',
    '3. Férias'. Os títulos são lidos uma vez por texto e achados com bisect
    (antes, cada citação refazia a varredura do texto inteiro).
    """

    def __init__(self, text: str):
        self.text = text
        found = [(m.start(), " ".join(m.group(0).split())[:60]) for m in _HEADING.finditer(text)]
        self.starts = [s for s, _ in found]
        self.labels = [lbl for _, lbl in found]

    def __call__(self, pos: int) -> str:
        i = bisect.bisect_right(self.starts, pos) - 1
        if i >= 0:
            return self.labels[i]
        return f"linha {self.text.count(chr(10), 0, pos) + 1}"


def analyze_impact(
    store: VersionStore,
    retriever,
    document_id: str,
    diff: VersionDiff,
    min_relevance: float = MIN_RELEVANCE,
    per_article: int = SUGGESTIONS_PER_ARTICLE,
    semantic: bool = True,
    verificar: Optional[Verifier] = None,
    min_doc_similarity: float = MIN_DOC_SIMILARITY,
) -> ImpactReport:
    """
    `verificar`, opcional: função (artigo_antes, artigo_depois, trecho) -> (depende, motivo),
    por exemplo LLMChain.verify_dependency. É chamada só para as sugestões que
    passaram pelos filtros locais; a sugestão que ela recusa sai do relatório
    (fica em report.rejected). Erro ou demora na função não derruba a análise:
    a sugestão fica, sem confirmação.
    """
    doc = store.get_document(document_id)
    # Documento sem artigos: não há "art. N" para outros documentos citarem.
    affected = [k for k in diff.affected_keys if k != "preambulo"] if diff.structured else []
    report = ImpactReport(document_id, doc.name if doc else document_id, affected,
                          structured=diff.structured,
                          removed_articles=[a.key for a in diff.removed] if diff.structured else [])
    if not affected:
        return report

    affected_base: Set[str] = {_base_key(k) for k in affected}
    aliases = doc.aliases if doc else [document_id]

    # 1. Citações diretas no texto vigente dos outros documentos
    other_texts: Dict[str, str] = {}
    for other, version in store.current_texts(exclude_doc_id=document_id):
        other_texts[other.id] = version.text
        locate = None
        for m in find_citations(version.text, aliases):
            for key in m.articles:
                if key in affected_base:
                    locate = locate or _Locator(version.text)
                    report.direct.append(DirectCitation(
                        other.id, other.name, key,
                        _snippet(version.text, m.start, m.end),
                        locate(m.start),
                        _sentence(version.text, m.start, m.end),
                    ))

    if not semantic or retriever is None:
        return report

    # 2. Sugestões.
    #    a) O ChromaDB acha os chunks candidatos (vigentes, de outros documentos)
    #       parecidos com o texto ANTIGO de cada artigo afetado, que é o texto em
    #       que os dependentes se basearam.
    #    b) Filtro de tema: o documento de cada candidato tem de ser do mesmo
    #       assunto do documento alterado (MIN_DOC_SIMILARITY). Sem isso, um
    #       edital de mestrado virava sugestão de um regulamento de cooperativa:
    #       a frase "o candidato deverá estar conectado com antecedência mínima
    #       de 5 minutos" fica a 0,888 do artigo de férias ("antecedência mínima
    #       de 60 dias"), mais do que frases que dependem de verdade.
    #    c) Cada frase dos candidatos que sobram é comparada com TODOS os artigos
    #       da versão antiga e fica com o mais parecido. Só vira sugestão se esse
    #       artigo foi alterado ou removido.
    #    d) Opcional: a LLM confirma ou recusa cada sugestão.
    names = {d.id: d.name for d in store.list_documents()}
    old_text: Dict[str, str] = {c.key: c.before for c in diff.changed}
    old_text.update({a.key: a.text for a in diff.removed})
    new_text: Dict[str, str] = {c.key: c.after for c in diff.changed}
    all_old = [a for a in diff.old_articles if a.key != "preambulo"]
    if not all_old:
        return report
    art_keys = [_base_key(a.key) for a in all_old]
    art_embs = _article_embeddings(retriever, all_old)

    # Uma busca em lote para todos os artigos afetados (um encode, uma query).
    queries = [old_text[k] for k in affected if k in old_text]
    candidates: Dict[str, Dict] = {}
    for hits in retriever.search_many(queries, top_k=per_article * 2,
                                      exclude_document_id=document_id, passage=True):
        for h in hits:
            candidates.setdefault(h["id"], h)

    topic = _topic_similarity(
        retriever, document_id, "\n".join(a.text for a in diff.old_articles),
        {d: other_texts.get(d, "") for d in {h.get("document_id") or "" for h in candidates.values()}},
        art_embs,
    )
    for d, sim in topic.items():
        if sim < min_doc_similarity:
            report.off_topic[names.get(d, d)] = sim
    candidates = {cid: h for cid, h in candidates.items()
                  if topic.get(h.get("document_id") or "", 0.0) >= min_doc_similarity}

    # Todas as frases candidatas numa chamada só ao modelo.
    pairs = [(h, s) for h in candidates.values() for s in split_sentences(h["text"])]
    if not pairs:
        return report
    sent_embs = retriever.model.encode([f"passage: {s}" for _, s in pairs],
                                       normalize_embeddings=True, show_progress_bar=False,
                                       batch_size=64)
    sims_all = sent_embs @ art_embs.T

    best: Dict[Tuple[str, str], Suggestion] = {}
    for (h, s), row in zip(pairs, sims_all):
        i = int(row.argmax())
        key, sim = art_keys[i], float(row[i])
        if key not in affected_base or sim < min_relevance:
            continue
        # Frase que já cita este artigo é citação direta, não sugestão.
        if any(key in m.articles for m in find_citations(s, aliases)):
            continue
        d = h.get("document_id") or ""
        best[(h["id"], s)] = Suggestion(
            d, names.get(d, h.get("source") or ""),
            key, h["id"], s, sim, topic.get(d, 0.0),
        )
    ranked = sorted(best.values(), key=lambda s: s.relevance, reverse=True)
    # Até `per_article` sugestões por artigo alterado.
    count: Dict[str, int] = {}
    chosen: List[Suggestion] = []
    for s in ranked:
        if count.get(s.article, 0) < per_article:
            chosen.append(s)
            count[s.article] = count.get(s.article, 0) + 1

    if verificar is not None and chosen:
        chosen = _verify(report, chosen, verificar, report.document_name, old_text, new_text)
    report.suggestions = chosen
    return report


def _current_embeddings(retriever, document_id: str) -> Optional[np.ndarray]:
    """Embeddings já guardados no ChromaDB dos chunks vigentes de um documento."""
    got = retriever.collection.get(
        where={"$and": [{"document_id": {"$eq": document_id}}, {"is_current": {"$eq": True}}]},
        include=["embeddings"],
    )
    embs = got.get("embeddings")
    if embs is None or len(embs) == 0:
        return None
    return np.asarray(embs, dtype=np.float32)


def _topic_similarity(retriever, document_id: str, document_text: str,
                      others: Dict[str, str], article_embs) -> Dict[str, float]:
    """
    Semelhança de tema entre o documento alterado e cada um dos `others`
    ({id: texto vigente}): média de "o começo" e "as partes" (ver MIN_DOC_SIMILARITY).
    Sem como medir nenhuma das duas, a nota é 0 e o documento não recebe sugestão.
    """
    out: Dict[str, float] = {}
    try:
        ref = _current_embeddings(retriever, document_id)
    except Exception as e:                      # retriever sem ChromaDB
        logger.warning(f"Tema do documento alterado indisponível: {e}")
        ref = None
    if ref is None:
        ref = np.asarray(article_embs, dtype=np.float32)
    ids = [d for d in others if d]
    heads = _head_embeddings(retriever, [document_text] + [others[d] for d in ids])
    for n, d in enumerate(ids):
        views: List[float] = []
        if heads is not None and others[d].strip():
            views.append(float(heads[n + 1] @ heads[0]))
        try:
            embs = _current_embeddings(retriever, d)
        except Exception as e:
            logger.warning(f"Tema de {d} indisponível: {e}")
            embs = None
        if embs is not None:
            per_chunk = np.sort((embs @ ref.T).max(axis=1))[::-1]
            k = max(1, math.ceil(len(per_chunk) / 4))
            views.append(float(per_chunk[:k].mean()))
        out[d] = sum(views) / len(views) if views else 0.0
    return out


_HEAD_CACHE: "OrderedDict[str, object]" = OrderedDict()
_HEAD_CACHE_MAX = 256


def _head_embeddings(retriever, texts: List[str]) -> Optional[np.ndarray]:
    """Embedding do começo de cada texto, com cache por conteúdo."""
    heads = [" ".join(t.split())[:TOPIC_HEAD_CHARS] for t in texts]
    keys = [hashlib.sha1(h.encode("utf-8")).hexdigest() for h in heads]
    with _ART_CACHE_LOCK:
        missing = [i for i, k in enumerate(keys) if k not in _HEAD_CACHE]
    if missing:
        try:
            embs = retriever.model.encode([f"passage: {heads[i]}" for i in missing],
                                          normalize_embeddings=True, show_progress_bar=False,
                                          batch_size=16)
        except Exception as e:
            logger.warning(f"Começo dos documentos indisponível: {e}")
            return None
        with _ART_CACHE_LOCK:
            for i, e in zip(missing, embs):
                _HEAD_CACHE[keys[i]] = e
            while len(_HEAD_CACHE) > _HEAD_CACHE_MAX:
                _HEAD_CACHE.popitem(last=False)
    with _ART_CACHE_LOCK:
        return np.asarray([_HEAD_CACHE.get(k) for k in keys], dtype=np.float32)


def _verify(report: ImpactReport, chosen: List[Suggestion], verificar: Verifier,
            doc_name: str, old_text: Dict[str, str], new_text: Dict[str, str]) -> List[Suggestion]:
    """
    Pergunta à LLM, em paralelo, se cada sugestão depende mesmo do artigo.
    Recusada: sai (vai para report.rejected). Confirmada: fica, com o motivo.
    Erro ou fora do prazo: fica, sem confirmação.
    """
    report.llm_used = True
    todo = chosen[:VERIFY_MAX]
    pool = ThreadPoolExecutor(max_workers=min(VERIFY_WORKERS, len(todo)))
    futures = {}
    for s in todo:
        before = old_text.get(s.article) or next(
            (v for k, v in old_text.items() if _base_key(k) == s.article), "")
        after = new_text.get(s.article) or next(
            (v for k, v in new_text.items() if _base_key(k) == s.article), "")
        futures[pool.submit(
            verificar,
            f"{doc_name}, art. {s.article}: {before}",
            f"{doc_name}, art. {s.article}: {after}" if after else "(o artigo foi retirado do documento)",
            f"{s.document_name}: {s.text}",
        )] = s
    done, _ = wait(futures, timeout=VERIFY_TIMEOUT)
    pool.shutdown(wait=False, cancel_futures=True)

    kept: List[Suggestion] = []
    for f, s in futures.items():
        if f not in done:
            logger.warning("Verificação por LLM passou do prazo; a sugestão fica sem confirmação.")
            kept.append(s)
            continue
        try:
            ok, reason = f.result()
        except Exception as e:
            logger.warning(f"Verificação por LLM falhou ({e}); a sugestão fica sem confirmação.")
            kept.append(s)
            continue
        s.verified, s.reason = bool(ok), " ".join(str(reason or "").split())
        (kept if ok else report.rejected).append(s)
    kept += chosen[VERIFY_MAX:]
    kept.sort(key=lambda s: s.relevance, reverse=True)
    return kept


# Embeddings dos artigos da versão antiga. O caput diz do que o artigo trata;
# cortar em ARTICLE_EMBED_CHARS evita passar artigos enormes (o art. 5º da
# CF/88 tem ~15 mil caracteres) pelo modelo inteiro. O cache evita refazer
# a conta ao comparar as mesmas versões de novo.
ARTICLE_EMBED_CHARS = 600
_ART_CACHE: "OrderedDict[str, object]" = OrderedDict()
_ART_CACHE_MAX = 8
_ART_CACHE_LOCK = threading.Lock()     # handlers do Gradio rodam em threads


def _article_embeddings(retriever, articles) -> "object":
    texts = [a.text[:ARTICLE_EMBED_CHARS] for a in articles]
    key = hashlib.sha1("\x00".join(texts).encode("utf-8")).hexdigest()
    with _ART_CACHE_LOCK:
        if key in _ART_CACHE:
            _ART_CACHE.move_to_end(key)
            return _ART_CACHE[key]
    embs = retriever.model.encode([f"passage: {t}" for t in texts], normalize_embeddings=True,
                                  show_progress_bar=False, batch_size=64)
    with _ART_CACHE_LOCK:
        _ART_CACHE[key] = embs
        while len(_ART_CACHE) > _ART_CACHE_MAX:
            _ART_CACHE.popitem(last=False)
    return embs


_SENT_SPLIT = re.compile(
    r"(?<=[.;:!?])(?<!art\.)(?<!Art\.)(?<!rts\.)\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ])"
)


def split_sentences(text: str, min_len: int = 40) -> List[str]:
    """
    Frases de um chunk. Não quebra em "art. 12" (abreviação seguida de número)
    e descarta pedaços curtos como títulos ("2. Horário de trabalho").
    """
    flat = " ".join(text.split())
    return [s for s in _SENT_SPLIT.split(flat) if len(s) >= min_len]


_MD_SPECIAL = re.compile(r"([\\`*_\[\]()!<>#|~])")


def md_escape(text: str) -> str:
    """
    Escapa texto vindo de documento (ou de nome digitado) antes de pôr em
    gr.Markdown: sem isso, um PDF podia injetar imagem externa ou link.
    """
    return _MD_SPECIAL.sub(r"\\\1", text)


def _join(items: List[str]) -> str:
    """'3' | '3 e 7' | '3, 7 e 12'"""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " e " + items[-1]


def _arts_label(keys: List[str]) -> str:
    return ("o artigo " if len(keys) == 1 else "os artigos ") + _join(keys)


def _short(text: str, limit: int = 280) -> str:
    t = " ".join(text.split())
    return t[:limit].rstrip() + ("…" if len(t) > limit else "")


def format_report(report: ImpactReport) -> str:
    """
    Markdown para a interface, para quem não é da área: o que é certo (o outro
    documento cita o artigo) e o que é possível (trata do mesmo assunto).
    """
    if not report.structured:
        return ("Este documento não tem artigos numerados. A comparação foi feita no "
                "texto inteiro, e não há artigo para procurar em outros documentos.")
    if not report.affected_articles:
        return "Nenhum artigo mudou ou saiu. Nada a conferir em outros documentos."

    removed = [k for k in report.affected_articles if k in set(report.removed_articles)]
    changed = [k for k in report.affected_articles if k not in set(removed)]
    head = []
    if changed:
        head.append(f"Mudou {_arts_label(changed)}" if len(changed) == 1
                    else f"Mudaram {_arts_label(changed)}")
    if removed:
        head.append(f"saiu {_arts_label(removed)}" if len(removed) == 1
                    else f"saíram {_arts_label(removed)}")
    first = ", e ".join(head) + "."
    lines = [first[0].upper() + first[1:], ""]

    # Certo: o outro documento cita o artigo.
    lines.append(f"### Certo ({len(report.direct)})")
    if not report.direct:
        lines.append("Nenhum outro documento cita esses artigos.")
    by_doc: "OrderedDict[str, List[DirectCitation]]" = OrderedDict()
    for c in report.direct:
        by_doc.setdefault(c.document_name, []).append(c)
    for name, cits in by_doc.items():
        arts = list(dict.fromkeys(c.article for c in cits))
        lines.append(f"**{md_escape(name)}** cita diretamente "
                     + ("o art. " if len(arts) == 1 else "os arts. ") + _join(arts) + ".")
        # A mesma frase pode citar mais de um artigo ("arts. 3º e 7º"): uma linha só.
        by_sentence: "OrderedDict[Tuple[str, str], List[str]]" = OrderedDict()
        for c in cits:
            keys = by_sentence.setdefault((c.location, c.sentence or c.snippet), [])
            if c.article not in keys:
                keys.append(c.article)
        for (where, text), keys in by_sentence.items():
            label = ("Art. " if len(keys) == 1 else "Arts. ") + _join(keys)
            lines.append(f"- {label}, em {md_escape(where)}: “{md_escape(_short(text))}”")
        lines.append("")

    # Possível: trata do mesmo assunto, sem citar o artigo.
    lines += ["", f"### Possível ({len(report.suggestions)})"]
    if not report.suggestions:
        lines.append("Nenhum trecho de outro documento parece depender desses artigos.")
    else:
        lines.append("Trechos que não citam o artigo, mas tratam do mesmo assunto. Vale conferir.")
        lines.append("")
    for s in report.suggestions:
        why = md_escape(s.reason) if s.verified and s.reason else "o texto trata do mesmo assunto."
        if why and why[-1] not in ".!?":
            why += "."
        lines.append(f"**{md_escape(s.document_name)}** pode depender do art. {s.article}: {why}")
        lines.append(f"- “{md_escape(_short(s.text))}”")
        lines.append("")
    return "\n".join(lines).rstrip()
