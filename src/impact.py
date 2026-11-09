"""
impact.py
─────────
Análise de impacto: quando um documento muda, quais trechos de OUTROS
documentos da base dependem dos artigos alterados ou removidos.

Dois níveis de confiança:
  • citação direta (certo)  — regex: "art. 12 do Regulamento", "arts. 3º e 9º do
                               Regulamento Interno", "Regulamento, art. 12"
  • sugestão (semântica)    — trecho parecido com o artigo antigo, acima de um
                               limiar mínimo de relevância, que não cita o artigo

As citações são procuradas no texto inteiro da versão vigente de cada documento
(guardado no SQLite); as sugestões, nos chunks vigentes do ChromaDB.
"""

from __future__ import annotations

import bisect
import hashlib
import os
import re
import threading
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple

from versioning import VersionDiff, VersionStore

# Piso de semelhança das sugestões. O multilingual-e5 põe quase qualquer frase
# jurídica entre 0,85 e 0,92 de qualquer artigo, então o limiar sozinho não
# separa nada; quem separa é a regra do "artigo mais parecido" em analyze_impact.
# No conjunto de exemplo o resultado é o mesmo de 0,80 a 0,87 (eval_versoes.py --sweep).
MIN_RELEVANCE = float(os.environ.get("IMPACT_MIN_RELEVANCE", "0.85"))
SUGGESTIONS_PER_ARTICLE = 3


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


@dataclass
class Suggestion:
    document_id: str
    document_name: str
    article: str
    chunk_id: str
    text: str
    relevance: float


@dataclass
class ImpactReport:
    document_id: str
    document_name: str
    affected_articles: List[str]
    direct: List[DirectCitation] = field(default_factory=list)
    suggestions: List[Suggestion] = field(default_factory=list)
    structured: bool = True

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
) -> ImpactReport:
    doc = store.get_document(document_id)
    # Documento sem artigos: não há "art. N" para outros documentos citarem.
    affected = [k for k in diff.affected_keys if k != "preambulo"] if diff.structured else []
    report = ImpactReport(document_id, doc.name if doc else document_id, affected,
                          structured=diff.structured)
    if not affected:
        return report

    affected_base: Set[str] = {_base_key(k) for k in affected}
    aliases = doc.aliases if doc else [document_id]

    # 1. Citações diretas no texto vigente dos outros documentos
    for other, version in store.current_texts(exclude_doc_id=document_id):
        locate = None
        for m in find_citations(version.text, aliases):
            for key in m.articles:
                if key in affected_base:
                    locate = locate or _Locator(version.text)
                    report.direct.append(DirectCitation(
                        other.id, other.name, key,
                        _snippet(version.text, m.start, m.end),
                        locate(m.start),
                    ))

    if not semantic or retriever is None:
        return report

    # 2. Sugestões semânticas.
    #    a) O ChromaDB acha os chunks candidatos (vigentes, de outros documentos)
    #       parecidos com o texto ANTIGO de cada artigo afetado, que é o texto em
    #       que os dependentes se basearam.
    #    b) Cada frase desses chunks é comparada com TODOS os artigos da versão
    #       antiga e fica com o mais parecido. Só vira sugestão se esse artigo
    #       foi alterado ou removido. Só o limiar não bastava: o e5 põe quase
    #       toda frase jurídica entre 0,87 e 0,90 de qualquer artigo, e a
    #       precisão ficava em 50% para qualquer limiar (ver docs/VERSIONAMENTO.md).
    names = {d.id: d.name for d in store.list_documents()}
    old_text: Dict[str, str] = {c.key: c.before for c in diff.changed}
    old_text.update({a.key: a.text for a in diff.removed})
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
        best[(h["id"], s)] = Suggestion(
            h.get("document_id") or "",
            names.get(h.get("document_id"), h.get("source") or ""),
            key, h["id"], s, sim,
        )
    ranked = sorted(best.values(), key=lambda s: s.relevance, reverse=True)
    # Até `per_article` sugestões por artigo alterado.
    count: Dict[str, int] = {}
    for s in ranked:
        if count.get(s.article, 0) < per_article:
            report.suggestions.append(s)
            count[s.article] = count.get(s.article, 0) + 1
    return report


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


def format_report(report: ImpactReport) -> str:
    """Markdown para a interface."""
    if not report.structured:
        return ("O documento não tem artigos (\"Art. N\" no começo da linha): a comparação "
                "é do texto inteiro e não há citação de artigo a procurar em outros documentos.")
    if not report.affected_articles:
        return "Nenhum artigo alterado ou removido: nada a verificar em outros documentos."
    lines = [
        f"**Artigos alterados ou removidos:** "
        + ", ".join(f"Art. {k}" for k in report.affected_articles),
        "",
        f"### ✅ Citações diretas ({len(report.direct)})",
    ]
    if not report.direct:
        lines.append("_Nenhum outro documento cita esses artigos explicitamente._")
    for c in report.direct:
        lines.append(f"- **{md_escape(c.document_name)}** ({md_escape(c.location)}) cita o "
                     f"**Art. {c.article}**: {md_escape(c.snippet)}")
    lines += ["", f"### 💡 Sugestões por semelhança ({len(report.suggestions)})"]
    if not report.suggestions:
        lines.append("_Nenhum trecho parecido acima do limiar._")
    for s in report.suggestions:
        txt = " ".join(s.text.split())
        txt = txt[:280] + ("…" if len(txt) > 280 else "")
        lines.append(f"- **{md_escape(s.document_name)}** — parecido com o Art. {s.article} "
                     f"(relevância {s.relevance:.0%}): {md_escape(txt)}")
    return "\n".join(lines)
