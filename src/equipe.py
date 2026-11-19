"""
equipe.py
─────────
O que permite várias pessoas usarem a mesma base ao mesmo tempo:

- contas de acesso (login), lidas do .env;
- conversa entre as pessoas: uma sala geral e uma sala por documento;
- presença: quem está com o app aberto agora e o que está vendo;
- conversas com a IA: cada pergunta e resposta fica guardada, para a equipe ver;
- atividade recente: versões publicadas, mensagens e perguntas à IA, numa linha do tempo só.

Fica no mesmo SQLite do registro de versões (data/versions.db). Cada tela
pergunta a cada poucos segundos se há novidade (ver `novidade`), em vez de o
servidor empurrar: é o que o Gradio permite sem servidor de mensagens.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SALA_GERAL = "geral"
ONLINE_SEGUNDOS = 20          # sem sinal há mais que isso = saiu
MAX_TEXTO = 2000              # mensagem maior é cortada

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mensagens (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    sala       TEXT NOT NULL,
    autor      TEXT NOT NULL,
    texto      TEXT NOT NULL,
    criado_em  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS mensagens_sala ON mensagens (sala, id);
CREATE TABLE IF NOT EXISTS presenca (
    usuario    TEXT PRIMARY KEY,
    visto_em   TEXT NOT NULL,
    vendo      TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS ia_perguntas (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    conversa   TEXT NOT NULL,
    autor      TEXT NOT NULL,
    pergunta   TEXT NOT NULL,
    resposta   TEXT NOT NULL,
    escopo     TEXT NOT NULL DEFAULT '',
    modelo     TEXT NOT NULL DEFAULT '',
    fontes     TEXT NOT NULL DEFAULT '[]',
    criado_em  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ia_perguntas_conversa ON ia_perguntas (conversa, id);
CREATE TABLE IF NOT EXISTS leitura (
    usuario    TEXT NOT NULL,
    sala       TEXT NOT NULL,
    ultimo_id  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (usuario, sala)
);
"""


def _agora() -> datetime:
    return datetime.now().replace(microsecond=0)


# ── Contas ────────────────────────────────────────────────────

@dataclass
class Conta:
    login: str
    senha: str
    nome: str


def ler_contas(texto: Optional[str] = None) -> List[Conta]:
    """
    Contas do .env, no formato `login:senha:Nome para mostrar`, separadas por
    vírgula. O nome é opcional (sem ele, vale o login com a inicial maiúscula).
    Ex.: EIXOAI_USUARIOS=joao:senha1:João Gabriel,luiz:senha2:Luiz Alberto
    Sem a variável, o app abre sem login.
    """
    texto = os.environ.get("EIXOAI_USUARIOS", "") if texto is None else texto
    contas = []
    for item in texto.split(","):
        partes = [p.strip() for p in item.split(":", 2)]
        if len(partes) < 2 or not partes[0] or not partes[1]:
            continue
        nome = partes[2] if len(partes) == 3 and partes[2] else partes[0].capitalize()
        contas.append(Conta(partes[0], partes[1], nome))
    return contas


def ler_admins(texto: Optional[str] = None) -> set:
    """
    Logins com cargo de admin, do .env: EIXOAI_ADMINS=joao,luiz.
    Só admin vê as conversas das outras pessoas com a IA.
    """
    texto = os.environ.get("EIXOAI_ADMINS", "") if texto is None else texto
    return {t.strip() for t in texto.split(",") if t.strip()}


# ── Registro ──────────────────────────────────────────────────

@dataclass
class Mensagem:
    id: int
    sala: str
    autor: str
    texto: str
    criado_em: str


@dataclass
class PerguntaIA:
    id: int
    pergunta: str
    resposta: str
    escopo: str
    modelo: str
    fontes: List[Dict]        # [{"documento", "pagina", "trecho"}]
    criado_em: str


@dataclass
class ConversaIA:
    """Uma conversa de uma pessoa com a IA (de abrir a página até 'Nova conversa')."""
    id: str
    autor: str
    perguntas: List[PerguntaIA]

    @property
    def inicio(self) -> str:
        return self.perguntas[0].criado_em

    @property
    def fim(self) -> str:
        return self.perguntas[-1].criado_em


@dataclass
class Evento:
    """Uma linha da atividade recente."""
    quando: str
    quem: str
    tipo: str          # "versao" | "mensagem" | "ia"
    texto: str
    sala: str = ""     # documento ou sala da mensagem


class Equipe:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        # Uma conexão por operação: o Gradio chama de threads diferentes.
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # mensagens
    def enviar(self, sala: str, autor: str, texto: str) -> Optional[Mensagem]:
        texto = (texto or "").strip()[:MAX_TEXTO]
        if not texto:
            return None
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO mensagens (sala, autor, texto, criado_em) VALUES (?,?,?,?)",
                (sala or SALA_GERAL, autor or "Visitante", texto, _agora().isoformat()))
            novo = cur.lastrowid
        return self.mensagem(novo)

    def mensagem(self, msg_id: int) -> Optional[Mensagem]:
        with self._conn() as c:
            r = c.execute("SELECT * FROM mensagens WHERE id=?", (msg_id,)).fetchone()
        return Mensagem(r["id"], r["sala"], r["autor"], r["texto"], r["criado_em"]) if r else None

    def conversa(self, sala: str, limite: int = 100) -> List[Mensagem]:
        """Últimas mensagens da sala, da mais antiga para a mais nova."""
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM mensagens WHERE sala=? ORDER BY id DESC LIMIT ?",
                (sala, limite)).fetchall()
        return [Mensagem(r["id"], r["sala"], r["autor"], r["texto"], r["criado_em"])
                for r in reversed(rows)]

    def ultima_mensagem(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COALESCE(MAX(id), 0) AS m FROM mensagens").fetchone()["m"]

    # conversas com a IA
    def registrar_pergunta(self, conversa: str, autor: str, pergunta: str, resposta: str,
                           escopo: str = "", modelo: str = "",
                           fontes: Optional[List[Dict]] = None) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO ia_perguntas (conversa, autor, pergunta, resposta, escopo, modelo, "
                "fontes, criado_em) VALUES (?,?,?,?,?,?,?,?)",
                (conversa, autor or "Visitante", pergunta[:MAX_TEXTO], resposta, escopo, modelo,
                 json.dumps(fontes or [], ensure_ascii=False), _agora().isoformat()))
            return cur.lastrowid

    def ultima_pergunta_ia(self) -> int:
        with self._conn() as c:
            return c.execute("SELECT COALESCE(MAX(id), 0) AS m FROM ia_perguntas").fetchone()["m"]

    def autores_ia(self) -> List[str]:
        with self._conn() as c:
            return [r["autor"] for r in c.execute(
                "SELECT autor, MAX(id) AS u FROM ia_perguntas GROUP BY autor ORDER BY u DESC")]

    def conversas_ia(self, autor: Optional[str] = None, limite: int = 30) -> List[ConversaIA]:
        """Conversas com a IA, da mais recente para a mais antiga; `autor` filtra por pessoa."""
        filtro, args = ("WHERE autor = ?", (autor,)) if autor else ("", ())
        with self._conn() as c:
            ids = [r["conversa"] for r in c.execute(
                f"SELECT conversa, MAX(id) AS u FROM ia_perguntas {filtro} "
                f"GROUP BY conversa ORDER BY u DESC LIMIT ?", (*args, limite))]
            if not ids:
                return []
            marcas = ",".join("?" * len(ids))
            rows = c.execute(f"SELECT * FROM ia_perguntas WHERE conversa IN ({marcas}) ORDER BY id",
                             ids).fetchall()
        por_id: Dict[str, ConversaIA] = {}
        for r in rows:
            conv = por_id.setdefault(r["conversa"], ConversaIA(r["conversa"], r["autor"], []))
            try:
                fontes = json.loads(r["fontes"])
            except ValueError:
                fontes = []
            conv.perguntas.append(PerguntaIA(r["id"], r["pergunta"], r["resposta"], r["escopo"],
                                             r["modelo"], fontes, r["criado_em"]))
        return [por_id[i] for i in ids if i in por_id]

    # leitura: o que cada pessoa já viu de cada sala
    def marcar_lida(self, usuario: str, sala: str) -> None:
        with self._conn() as c:
            ultimo = c.execute("SELECT COALESCE(MAX(id), 0) AS m FROM mensagens WHERE sala=?",
                               (sala,)).fetchone()["m"]
            c.execute(
                "INSERT INTO leitura (usuario, sala, ultimo_id) VALUES (?,?,?) "
                "ON CONFLICT(usuario, sala) DO UPDATE SET ultimo_id=excluded.ultimo_id",
                (usuario, sala, ultimo))

    def nao_lidas(self, usuario: str) -> Dict[str, int]:
        """Mensagens de outras pessoas ainda não vistas, por sala."""
        with self._conn() as c:
            rows = c.execute(
                """SELECT m.sala, COUNT(*) AS n FROM mensagens m
                   LEFT JOIN leitura l ON l.usuario = ? AND l.sala = m.sala
                   WHERE m.autor <> ? AND m.id > COALESCE(l.ultimo_id, 0)
                   GROUP BY m.sala""", (usuario, usuario)).fetchall()
        return {r["sala"]: r["n"] for r in rows}

    # presença
    def sinal(self, usuario: str, vendo: str = "") -> None:
        """Chamado pela tela a cada poucos segundos: 'estou aqui, vendo X'."""
        with self._conn() as c:
            c.execute(
                "INSERT INTO presenca (usuario, visto_em, vendo) VALUES (?,?,?) "
                "ON CONFLICT(usuario) DO UPDATE SET visto_em=excluded.visto_em, vendo=excluded.vendo",
                (usuario, _agora().isoformat(), vendo or ""))

    def online(self, agora: Optional[datetime] = None) -> List[Tuple[str, str]]:
        """(usuário, o que está vendo) de quem deu sinal nos últimos ONLINE_SEGUNDOS."""
        limite = ((agora or _agora()) - timedelta(seconds=ONLINE_SEGUNDOS)).isoformat()
        with self._conn() as c:
            rows = c.execute("SELECT usuario, vendo FROM presenca WHERE visto_em >= ? "
                             "ORDER BY usuario", (limite,)).fetchall()
        return [(r["usuario"], r["vendo"]) for r in rows]

    # atividade: versões (do registro de versões) + mensagens
    def atividade(self, limite: int = 30, incluir_ia: bool = False) -> List[Evento]:
        """Linha do tempo da equipe. Perguntas à IA só entram com incluir_ia (para admin)."""
        with self._conn() as c:
            tem_versoes = c.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='versions'").fetchone()
            eventos: List[Evento] = []
            if tem_versoes:
                for r in c.execute(
                        """SELECT v.created_at, v.author, v.version, v.reason, d.name
                           FROM versions v JOIN documents d ON d.id = v.document_id
                           ORDER BY v.id DESC LIMIT ?""", (limite,)):
                    acao = (f"enviou {r['name']}" if r["version"] == 1
                            else f"publicou a versão {r['version']} de {r['name']}")
                    if r["reason"] and r["reason"] != "enviado pelo chat":
                        acao += f": {r['reason']}"
                    eventos.append(Evento(r["created_at"], r["author"] or "Alguém", "versao",
                                          acao, r["name"]))
            nomes = {}
            if tem_versoes:
                nomes = {r["id"]: r["name"] for r in c.execute("SELECT id, name FROM documents")}
            for r in c.execute("SELECT * FROM mensagens ORDER BY id DESC LIMIT ?", (limite,)):
                onde = "na conversa geral" if r["sala"] == SALA_GERAL else \
                       f"sobre {nomes.get(r['sala'], r['sala'])}"
                trecho = r["texto"] if len(r["texto"]) <= 80 else r["texto"][:77] + "..."
                eventos.append(Evento(r["criado_em"], r["autor"], "mensagem",
                                      f"escreveu {onde}: {trecho}", r["sala"]))
            consulta_ia = "SELECT * FROM ia_perguntas ORDER BY id DESC LIMIT ?" if incluir_ia else                 "SELECT * FROM ia_perguntas WHERE 0 LIMIT ?"
            for r in c.execute(consulta_ia, (limite,)):
                trecho = r["pergunta"] if len(r["pergunta"]) <= 80 else r["pergunta"][:77] + "..."
                eventos.append(Evento(r["criado_em"], r["autor"], "ia",
                                      f"perguntou à IA: {trecho}", r["conversa"]))
        eventos.sort(key=lambda e: e.quando, reverse=True)
        return eventos[:limite]


def quando_humano(iso: str, agora: Optional[datetime] = None) -> str:
    """'agora', 'há 5 min', 'hoje às 14:03', 'ontem às 09:10', '17/11 às 10:00'."""
    try:
        t = datetime.fromisoformat(iso)
    except (TypeError, ValueError):
        return iso or ""
    agora = agora or _agora()
    seg = (agora - t).total_seconds()
    if seg < 60:
        return "agora"
    if seg < 3600:
        return f"há {int(seg // 60)} min"
    if t.date() == agora.date():
        return f"hoje às {t:%H:%M}"
    if t.date() == (agora - timedelta(days=1)).date():
        return f"ontem às {t:%H:%M}"
    return f"{t:%d/%m} às {t:%H:%M}"
