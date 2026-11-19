"""
versioning.py
─────────────
Histórico de versões de documentos.

  • VersionStore      — registro das versões em SQLite (autor, data, motivo, SHA-256)
  • split_articles    — divide o texto em artigos ("Art. N" no começo da linha)
  • compare_versions  — artigos adicionados, removidos e alterados (difflib)
  • publish_version   — registra + indexa no ChromaDB + troca a versão vigente

O texto de cada versão fica guardado no SQLite: a comparação não precisa
reabrir o PDF, e o ChromaDB continua só com os chunks.
"""

from __future__ import annotations

import difflib
import hashlib
import logging
import os
import re
import sqlite3
import threading
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_DB = Path(
    os.environ.get(
        "EIXOAI_VERSIONS_DB",
        str(Path(__file__).resolve().parent.parent / "data" / "versions.db"),
    )
)

# "Art. 5º", "Art. 12.", "Art. 12-A", "Art. 1o" no começo da linha.
# O sufixo de letra exige hífen colado ("12-A"): no estilo antigo da CLT,
# "Art. 5º - A todo trabalho…", o "A" é a primeira palavra do caput.
_ARTICLE_HEADER = re.compile(
    r"(?m)^[ \t]*Art\.\s*(\d+)\s*(?:º|°|o(?![A-Za-zÀ-ÿ]))?(?:-([A-Z])(?![A-Za-zÀ-ÿ]))?"
)
_PUBLISH_LOCK = threading.Lock()
# Acima disto, o diff palavra a palavra usa a heurística "autojunk" do difflib:
# sem ela, um texto de 12 mil palavras sem artigos levava 6 s.
_AUTOJUNK_WORDS = 3000
PREAMBLE_KEY = "preambulo"


# ── Utilidades ────────────────────────────────────────────────

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def slugify(name: str) -> str:
    """'Regulamento Interno (2026)' → 'regulamento-interno-2026'."""
    s = unicodedata.normalize("NFKD", name)
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "documento"


def _norm_ws(text: str) -> str:
    """Compara sem ligar para quebra de linha ou espaço duplo do PDF."""
    return " ".join(text.split())


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ── Divisão por artigo ────────────────────────────────────────

@dataclass
class Article:
    key: str      # "12", "12-A", "1#2" (segunda ocorrência do Art. 1, como no ADCT)
    label: str    # "Art. 12"
    text: str


def split_articles(text: str) -> List[Article]:
    """
    Divide o texto em artigos. O trecho antes do primeiro artigo vira o
    'preâmbulo'. Usa a mesma âncora do chunker jurídico (has_article_structure):
    "Art." no começo da linha, para não confundir artigo com citação.
    """
    matches = list(_ARTICLE_HEADER.finditer(text))
    if not matches:
        return []

    articles: List[Article] = []
    head = text[: matches[0].start()].strip()
    if head:
        articles.append(Article(PREAMBLE_KEY, "Preâmbulo", head))

    seen: Dict[str, int] = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        base = m.group(1) + (f"-{m.group(2)}" if m.group(2) else "")
        seen[base] = seen.get(base, 0) + 1
        key = base if seen[base] == 1 else f"{base}#{seen[base]}"
        label = f"Art. {base}" + (f" ({seen[base]}ª ocorrência)" if seen[base] > 1 else "")
        articles.append(Article(key, label, text[m.start():end].strip()))
    return articles


# ── Comparação ────────────────────────────────────────────────

@dataclass
class ArticleChange:
    key: str
    label: str
    before: str
    after: str
    similarity: float
    edits: List[Tuple[str, str]]   # (trecho antes, trecho depois), por palavra

    def edits_text(self, max_items: int = 4) -> str:
        parts = []
        for old, new in self.edits[:max_items]:
            if old and new:
                parts.append(f'"{old}" → "{new}"')
            elif old:
                parts.append(f'removido "{old}"')
            else:
                parts.append(f'incluído "{new}"')
        return "; ".join(parts)


@dataclass
class VersionDiff:
    added: List[Article] = field(default_factory=list)
    removed: List[Article] = field(default_factory=list)
    changed: List[ArticleChange] = field(default_factory=list)
    unchanged: int = 0
    structured: bool = True        # False quando o texto não tem artigos
    old_articles: List[Article] = field(default_factory=list)   # todos os artigos da versão antiga

    @property
    def has_changes(self) -> bool:
        return bool(self.added or self.removed or self.changed)

    @property
    def affected_keys(self) -> List[str]:
        """Artigos que existiam antes e mudaram ou sumiram: são os que outros citam."""
        return [c.key for c in self.changed] + [a.key for a in self.removed]

    def as_text(self) -> str:
        """Resumo determinístico, usado sem LLM e como entrada para a LLM."""
        lines = []
        for c in self.changed:
            lines.append(f"{c.label} alterado: {c.edits_text()}")
        for a in self.removed:
            lines.append(f"{a.label} removido.")
        for a in self.added:
            lines.append(f"{a.label} incluído.")
        if not lines:
            lines.append("Nenhuma diferença de conteúdo entre as versões.")
        return "\n".join(lines)


def _word_edits(before: str, after: str) -> Tuple[float, List[Tuple[str, str]]]:
    a, b = before.split(), after.split()
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=len(a) + len(b) > _AUTOJUNK_WORDS)
    edits = [
        (" ".join(a[i1:i2]), " ".join(b[j1:j2]))
        for tag, i1, i2, j1, j2 in sm.get_opcodes()
        if tag != "equal"
    ]
    return sm.ratio(), edits


def compare_versions(old_text: str, new_text: str) -> VersionDiff:
    """Compara duas versões artigo por artigo."""
    old_arts = split_articles(old_text)
    new_arts = split_articles(new_text)

    if not old_arts and not new_arts:
        # Documento sem artigos: compara o texto inteiro como uma unidade.
        diff = VersionDiff(structured=False)
        if _norm_ws(old_text) != _norm_ws(new_text):
            ratio, edits = _word_edits(_norm_ws(old_text), _norm_ws(new_text))
            diff.changed.append(
                ArticleChange("documento", "Documento", old_text, new_text, ratio, edits)
            )
        else:
            diff.unchanged = 1
        return diff

    old_map = {a.key: a for a in old_arts}
    new_map = {a.key: a for a in new_arts}
    diff = VersionDiff(old_articles=old_arts)

    for a in old_arts:
        if a.key not in new_map:
            diff.removed.append(a)
            continue
        b = new_map[a.key]
        if _norm_ws(a.text) == _norm_ws(b.text):
            diff.unchanged += 1
        else:
            ratio, edits = _word_edits(_norm_ws(a.text), _norm_ws(b.text))
            diff.changed.append(ArticleChange(a.key, a.label, a.text, b.text, ratio, edits))

    diff.added = [b for b in new_arts if b.key not in old_map]
    return diff


def unified_diff(before: str, after: str, label: str = "") -> str:
    """Diff de linhas para mostrar na interface."""
    a = [_norm_ws(s) for s in re.split(r"(?<=[.;:])\s+", before) if s.strip()]
    b = [_norm_ws(s) for s in re.split(r"(?<=[.;:])\s+", after) if s.strip()]
    return "\n".join(
        difflib.unified_diff(a, b, fromfile=f"{label} (antes)", tofile=f"{label} (depois)",
                             lineterm="", n=1)
    )


# ── Registro em SQLite ────────────────────────────────────────

@dataclass
class DocumentInfo:
    id: str
    name: str
    aliases: List[str]
    created_at: str


@dataclass
class VersionInfo:
    document_id: str
    version: int
    filename: str
    sha256: str
    author: str
    reason: str
    created_at: str
    n_chunks: int
    is_current: bool
    text: str = ""
    change_summary: str = ""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    aliases     TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS versions (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id     TEXT NOT NULL REFERENCES documents(id),
    version         INTEGER NOT NULL,
    filename        TEXT NOT NULL,
    sha256          TEXT NOT NULL,
    author          TEXT NOT NULL DEFAULT '',
    reason          TEXT NOT NULL DEFAULT '',
    created_at      TEXT NOT NULL,
    text            TEXT NOT NULL,
    n_chunks        INTEGER NOT NULL DEFAULT 0,
    is_current      INTEGER NOT NULL DEFAULT 0,
    change_summary  TEXT NOT NULL DEFAULT '',
    UNIQUE (document_id, version)
);
"""


class VersionStore:
    def __init__(self, db_path: str | Path = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        # Uma conexão por operação: o Gradio chama de threads diferentes.
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # documentos
    def ensure_document(self, doc_id: str, name: str, aliases: Optional[List[str]] = None) -> DocumentInfo:
        aliases = [a.strip() for a in (aliases or []) if a.strip()]
        with self._conn() as c:
            row = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
            if row is None:
                c.execute(
                    "INSERT INTO documents (id, name, aliases, created_at) VALUES (?,?,?,?)",
                    (doc_id, name, "|".join(aliases), _now()),
                )
            elif aliases:
                old = [a for a in row["aliases"].split("|") if a]
                merged = old + [a for a in aliases if a not in old]
                c.execute("UPDATE documents SET aliases=? WHERE id=?", ("|".join(merged), doc_id))
        return self.get_document(doc_id)

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> DocumentInfo:
        return DocumentInfo(row["id"], row["name"], [a for a in row["aliases"].split("|") if a],
                            row["created_at"])

    def get_document(self, doc_id: str) -> Optional[DocumentInfo]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(self) -> List[DocumentInfo]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM documents ORDER BY name").fetchall()
        return [self._row_to_document(r) for r in rows]

    # versões
    @staticmethod
    def _row_to_version(row: sqlite3.Row, with_text: bool = True) -> VersionInfo:
        return VersionInfo(
            document_id=row["document_id"], version=row["version"], filename=row["filename"],
            sha256=row["sha256"], author=row["author"], reason=row["reason"],
            created_at=row["created_at"], n_chunks=row["n_chunks"],
            is_current=bool(row["is_current"]),
            text=row["text"] if with_text else "", change_summary=row["change_summary"],
        )

    def next_version(self, doc_id: str) -> int:
        with self._conn() as c:
            row = c.execute("SELECT MAX(version) AS v FROM versions WHERE document_id=?",
                            (doc_id,)).fetchone()
        return (row["v"] or 0) + 1

    def add_version(self, doc_id: str, version: int, *, filename: str, sha256: str, text: str,
                    author: str = "", reason: str = "", n_chunks: int = 0) -> VersionInfo:
        with self._conn() as c:
            c.execute("UPDATE versions SET is_current=0 WHERE document_id=?", (doc_id,))
            c.execute(
                """INSERT INTO versions (document_id, version, filename, sha256, author, reason,
                                         created_at, text, n_chunks, is_current)
                   VALUES (?,?,?,?,?,?,?,?,?,1)""",
                (doc_id, version, filename, sha256, author, reason, _now(), text, n_chunks),
            )
        return self.get_version(doc_id, version)

    def set_summary(self, doc_id: str, version: int, summary: str) -> None:
        with self._conn() as c:
            c.execute("UPDATE versions SET change_summary=? WHERE document_id=? AND version=?",
                      (summary, doc_id, version))

    def get_version(self, doc_id: str, version: int) -> Optional[VersionInfo]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM versions WHERE document_id=? AND version=?",
                            (doc_id, version)).fetchone()
        return self._row_to_version(row) if row else None

    def get_current(self, doc_id: str) -> Optional[VersionInfo]:
        with self._conn() as c:
            row = c.execute("SELECT * FROM versions WHERE document_id=? AND is_current=1",
                            (doc_id,)).fetchone()
        return self._row_to_version(row) if row else None

    def list_versions(self, doc_id: str) -> List[VersionInfo]:
        # Sem a coluna `text`: o histórico não precisa do documento inteiro.
        with self._conn() as c:
            rows = c.execute(
                "SELECT document_id, version, filename, sha256, author, reason, created_at, "
                "n_chunks, is_current, change_summary FROM versions "
                "WHERE document_id=? ORDER BY version", (doc_id,)).fetchall()
        return [self._row_to_version(r, with_text=False) for r in rows]

    def current_texts(self, exclude_doc_id: Optional[str] = None) -> List[Tuple[DocumentInfo, VersionInfo]]:
        """Versão vigente de cada documento (para a análise de impacto), numa consulta só."""
        with self._conn() as c:
            rows = c.execute(
                """SELECT d.id AS d_id, d.name AS d_name, d.aliases AS d_aliases,
                          d.created_at AS d_created, v.*
                   FROM documents d JOIN versions v ON v.document_id = d.id AND v.is_current = 1
                   WHERE d.id <> ? ORDER BY d.name""",
                (exclude_doc_id or "",),
            ).fetchall()
        return [(DocumentInfo(r["d_id"], r["d_name"], [a for a in r["d_aliases"].split("|") if a],
                              r["d_created"]), self._row_to_version(r)) for r in rows]

    def current_totals(self) -> Tuple[int, int]:
        """(documentos com versão vigente, total de chunks vigentes)."""
        with self._conn() as c:
            row = c.execute("SELECT COUNT(*) AS n, COALESCE(SUM(n_chunks), 0) AS ch "
                            "FROM versions WHERE is_current = 1").fetchone()
        return row["n"], row["ch"]

    def has_documents(self) -> bool:
        with self._conn() as c:
            return c.execute("SELECT 1 FROM documents LIMIT 1").fetchone() is not None

    # tempo real: cada tela pergunta "mudou algo desde a última vez?"
    def last_change(self) -> int:
        """Id da versão mais recente da base (0 se vazia). Consulta barata, feita a cada poucos segundos."""
        with self._conn() as c:
            row = c.execute("SELECT COALESCE(MAX(id), 0) AS m FROM versions").fetchone()
        return row["m"]

    def versions_since(self, last_id: int) -> List[Tuple[int, str, VersionInfo]]:
        """Versões registradas depois de `last_id`: (id, nome do documento, versão), da mais antiga à mais nova."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT v.id AS row_id, d.name AS d_name, v.document_id, v.version, v.filename, "
                "v.sha256, v.author, v.reason, v.created_at, v.n_chunks, v.is_current, "
                "v.change_summary FROM versions v JOIN documents d ON d.id = v.document_id "
                "WHERE v.id > ? ORDER BY v.id", (last_id,)).fetchall()
        return [(r["row_id"], r["d_name"], self._row_to_version(r, with_text=False)) for r in rows]


# ── Publicação: registro + índice ─────────────────────────────

@dataclass
class PublishResult:
    status: str                      # "created" | "unchanged"
    document: DocumentInfo
    version: VersionInfo
    previous: Optional[VersionInfo] = None
    diff: Optional[VersionDiff] = None


def publish_version(
    store: VersionStore,
    retriever,
    *,
    name: str,
    text: str,
    filename: str,
    sha256: str,
    author: str = "",
    reason: str = "",
    aliases: Optional[List[str]] = None,
    document_id: Optional[str] = None,
) -> PublishResult:
    """
    Registra uma nova versão. Se o arquivo for idêntico à versão vigente
    (mesmo SHA-256), não cria versão nova. Nada é apagado: o texto de cada
    versão fica no SQLite, e os chunks que mudaram ficam no ChromaDB marcados
    como não vigentes.

    Uma publicação por vez: o Gradio roda as abas Chat e Versões em threads
    diferentes, e duas publicações do mesmo documento pegariam o mesmo número
    de versão, deixando o SQLite e o ChromaDB fora de sincronia.
    """
    with _PUBLISH_LOCK:
        doc_id = document_id or slugify(name)
        doc = store.ensure_document(doc_id, name, aliases or [name])
        previous = store.get_current(doc_id)

        if previous and previous.sha256 == sha256:
            return PublishResult("unchanged", doc, previous, previous=None, diff=None)

        version = store.next_version(doc_id)
        try:
            # Só os chunks que mudaram entram ou saem do índice (ver add_version).
            result = retriever.add_version(
                document_id=doc_id,
                text=text,
                version=version,
                source=filename,
                metadata={"document_name": doc.name},
            )
            info = store.add_version(doc_id, version, filename=filename, sha256=sha256,
                                     text=text, author=author, reason=reason,
                                     n_chunks=result["count"])
        except Exception:
            # Desfaz o índice se algo falhar no meio, para os dois não divergirem.
            retriever.rollback_version(doc_id, version)
            raise

    diff = compare_versions(previous.text, text) if previous else None
    logger.info(f"📚 {doc.name}: versão {version} registrada ({result['count']} chunks)")
    return PublishResult("created", doc, info, previous=previous, diff=diff)
