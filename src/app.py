"""
EixoAI — documentos e versões de uma equipe, com perguntas respondidas pela IA.

Três abas:
  Perguntar   chat com a IA sobre a versão atual dos documentos;
  Documentos  lista para navegar, texto atual, mudanças, histórico, conversa
              sobre cada documento e envio de arquivos (documento novo ou
              versão nova);
  Equipe      conversa geral, quem está aqui e atividade recente.

Várias pessoas usam a mesma base ao mesmo tempo. Cada tela pergunta a cada
REFRESH_SECONDS se surgiu versão ou mensagem nova (gr.Timer) e atualiza só a
aba aberta: no Gradio 6.3, componente de aba oculta que recebe atualização
fica preso com o "carregando" por cima quando a aba aparece.
"""

import gradio as gr
import html
import inspect
import logging
import os
import re
import shutil
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

from ingester import extract_text_from_pdf
from llm_chain import DEFAULT_MODEL, LLMChain
from retriever import DocumentRetriever
from versioning import (
    VersionStore, VersionDiff, compare_versions, publish_version, sha256_file, split_articles,
)
from impact import ImpactReport, analyze_impact, format_report, md_escape
from busca import buscar
from equipe import SALA_GERAL, Equipe, ler_admins, ler_contas, quando_humano
from markdown_it import MarkdownIt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("../data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALL_DOCS = "__todos__"
NOVO_DOC = "__novo__"
REFRESH_SECONDS = 3
SECOES = ["Documento", "Mudanças", "Histórico", "Conversa"]

CONTAS = ler_contas()
ADMINS = ler_admins()
NOMES = {c.login: c.nome for c in CONTAS}

# ─── Recursos compartilhados entre sessões ────────────────────────────────────
# O modelo de embeddings, o ChromaDB, o registro de versões e a conversa da
# equipe são de todos. O que é de cada pessoa (histórico do chat com a IA,
# modelo escolhido, aba aberta) fica em gr.State.

_retriever: Optional[DocumentRetriever] = None
_store: Optional[VersionStore] = None
_equipe: Optional[Equipe] = None
_retriever_lock = threading.Lock()


def get_retriever() -> DocumentRetriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:          # o warm_up e um envio podem chegar juntos
            if _retriever is None:
                _retriever = DocumentRetriever()
    return _retriever


def get_store() -> VersionStore:
    global _store
    if _store is None:
        _store = VersionStore()
    return _store


def get_equipe() -> Equipe:
    global _equipe
    if _equipe is None:
        _equipe = Equipe(get_store().db_path)     # mesmo SQLite das versões
    return _equipe


def warm_up() -> None:
    """Carrega o modelo de embeddings em segundo plano assim que o app sobe."""
    threading.Thread(target=get_retriever, name="warm-up", daemon=True).start()


def eh_admin(request: Optional[gr.Request]) -> bool:
    """Admin (EIXOAI_ADMINS no .env) vê as conversas das outras pessoas com a IA."""
    login = getattr(request, "username", None) if request else None
    return bool(login) and login in ADMINS


def exibir(nome: str) -> str:
    """Nome para mostrar: 'Edital_Processo_PPGI' aparece como 'Edital Processo PPGI'."""
    return " ".join(nome.replace("_", " ").split()) if nome else nome


def quem(request: Optional[gr.Request]) -> str:
    """Nome de quem está usando: vem do login. Sem login configurado, 'Visitante'."""
    login = getattr(request, "username", None) if request else None
    return NOMES.get(login, login) if login else "Visitante"

# ─── Modelos do Groq ──────────────────────────────────────────────────────────

_MODELOS_CONHECIDOS = {
    "openai/gpt-oss-120b": "GPT-OSS 120B · mais completo",
    "openai/gpt-oss-20b": "GPT-OSS 20B · mais rápido",
    "qwen/qwen3.8-27b": "Qwen 3.8 27B · equilibrado",
    "llama-3.3-70b-versatile": "Llama 3.3 70B",
    "llama-3.1-8b-instant": "Llama 3.1 8B · instantâneo",
}
# Modelos de áudio e de moderação não respondem perguntas.
_NAO_CONVERSAM = ("whisper", "guard", "orpheus", "tts", "playai", "safeguard", "distil")
_modelos_cache: Tuple[float, List[Tuple[str, str]]] = (0.0, [])


def modelos_disponiveis() -> List[Tuple[str, str]]:
    """(rótulo, id) dos modelos de conversa da conta do Groq; consulta a cada 10 min."""
    global _modelos_cache
    quando, lista = _modelos_cache
    if lista and time.time() - quando < 600:
        return lista
    ids: List[str] = []
    try:
        from groq import Groq
        ids = [m.id for m in Groq(timeout=8).models.list().data
               if not any(p in m.id.lower() for p in _NAO_CONVERSAM)]
    except Exception as e:
        logger.warning(f"Não deu para listar os modelos do Groq: {e}")
    if DEFAULT_MODEL not in ids:
        ids.insert(0, DEFAULT_MODEL)
    conhecidos = [i for i in _MODELOS_CONHECIDOS if i in ids]
    outros = sorted(i for i in ids if i not in _MODELOS_CONHECIDOS)
    lista = [(_MODELOS_CONHECIDOS.get(i, i), i) for i in conhecidos + outros]
    _modelos_cache = (time.time(), lista)
    return lista


def _chain(llm: Optional[LLMChain], modelo: Optional[str]) -> Optional[LLMChain]:
    """LLMChain da sessão no modelo escolhido; troca de modelo mantém a conversa."""
    modelo = modelo or DEFAULT_MODEL
    if llm is not None and llm.model == modelo:
        return llm
    try:
        novo = LLMChain(model=modelo, stream=False)
        # Sem as novas tentativas automáticas do cliente (esperavam 10 s ou mais
        # no limite por minuto): no limite, troca de modelo (ver _com_reserva).
        novo.client = novo.client.with_options(max_retries=0, timeout=60)
    except Exception as e:
        logger.warning(f"IA indisponível: {e}")
        return llm
    if llm is not None:
        novo.conversation_history = list(llm.conversation_history)
    return novo


def _rotulo_modelo(modelo_id: str) -> str:
    return dict((i, r) for r, i in modelos_disponiveis()).get(modelo_id, modelo_id).split(" · ")[0]


def _no_limite(e: Exception) -> bool:
    t = str(e).lower()
    return "rate_limit" in t or "rate limit" in t or "error code: 429" in t


class IAOcupada(Exception):
    """Todos os modelos tentados estão no limite de uso por minuto do Groq."""


def _com_reserva(llm: LLMChain, tarefa):
    """
    Roda tarefa(chain) no modelo escolhido. Se ele bateu no limite de uso por
    minuto da conta do Groq (comum na conta gratuita, com várias pessoas ao
    mesmo tempo), tenta até dois outros modelos. Devolve (resultado, aviso).
    """
    tentados = []
    ids = [llm.model] + [i for _, i in modelos_disponiveis() if i != llm.model]
    for m in ids[:3]:
        chain = llm if m == llm.model else _chain(llm, m)
        if chain is None or chain.model != m:
            continue
        antes = len(chain.conversation_history)
        try:
            r = tarefa(chain)
        except Exception as e:
            # generate_response põe a pergunta no histórico antes de chamar a API;
            # se a chamada falha, ela sai, senão a próxima tentativa a duplicaria.
            del chain.conversation_history[antes:]
            if not _no_limite(e):
                raise
            tentados.append(m)
            continue
        if chain is not llm:
            llm.conversation_history = chain.conversation_history
        aviso = "" if not tentados else (
            f"O modelo {_rotulo_modelo(tentados[0])} chegou ao limite de uso por minuto; "
            f"esta resposta veio do {_rotulo_modelo(m)}.")
        return r, aviso
    raise IAOcupada()

# ─── Paleta ───────────────────────────────────────────────────────────────────
# Azul petróleo bem escuro, texto quase branco, verde-água nos destaques e
# amarelo para pedir atenção. Linhas finas, cantos de 10 px, sem emoji.

FUNDO = "#081C25"
PAINEL = "#0B2530"
PAINEL_2 = "#0E2C38"
LINHA = "rgba(238, 243, 240, .14)"
LINHA_FORTE = "rgba(238, 243, 240, .32)"
TEXTO = "#EEF3F0"
SUAVE = "#9FB4B9"
DESTAQUE = "#3DDBC3"
DESTAQUE_TINTA = "#04141A"
ATENCAO = "#F4C542"
ERRO = "#FF6B6B"


def _caixa(corpo: str, cor: str = LINHA, fundo: str = PAINEL) -> str:
    return (f"<div class='caixa' style='border-color:{cor};background:{fundo}'>{corpo}</div>")


def _aviso_erro(msg: str) -> str:
    return _caixa(f"<b style='color:{ERRO}'>Não foi possível concluir.</b> {html.escape(msg)}",
                  cor=ERRO)


def _aviso_ok(msg: str) -> str:
    return _caixa(html.escape(msg), cor=DESTAQUE)


def _alerta_mudanca(diff: Optional[VersionDiff], impact: Optional[ImpactReport], nome: str,
                    v_antes: Optional[int], v_depois: int) -> str:
    """Faixa que aparece quando um documento muda: o que mudou e quem é afetado."""
    nome = html.escape(nome)
    if v_antes is None or diff is None:
        return _caixa(f"<b>{nome}</b> foi adicionado à base (versão {v_depois}).", cor=DESTAQUE)
    n_alt = len([c for c in diff.changed if c.key != "preambulo"])
    partes = []
    if not diff.structured:
        partes.append("o texto mudou (documento sem artigos)")
    elif n_alt:
        partes.append(f"{n_alt} artigo{'s' if n_alt > 1 else ''} mudou" if n_alt == 1
                      else f"{n_alt} artigos mudaram")
    if diff.removed:
        partes.append(f"{len(diff.removed)} saiu" if len(diff.removed) == 1
                      else f"{len(diff.removed)} saíram")
    if diff.added:
        partes.append(f"{len(diff.added)} entrou" if len(diff.added) == 1
                      else f"{len(diff.added)} entraram")
    oque = ", ".join(partes) or "só mudou o cabeçalho"
    n_cita = len(impact.direct) if impact else 0
    n_pode = len(impact.suggestions) if impact else 0
    docs = sorted({c.document_name for c in impact.direct}) if impact else []
    topo = f"<div class='alerta-titulo'><b>{nome}</b> mudou da versão {v_antes} para a {v_depois}: {oque}.</div>"
    if n_cita or n_pode:
        linhas = []
        if n_cita:
            linhas.append(f"<b>{n_cita}</b> trecho{'s' if n_cita > 1 else ''} em "
                          f"{html.escape(', '.join(docs))} cita{'m' if n_cita > 1 else ''} "
                          f"diretamente o que mudou.")
        if n_pode:
            linhas.append(f"<b>{n_pode}</b> trecho{'s' if n_pode > 1 else ''} "
                          f"pode{'m' if n_pode > 1 else ''} depender do que mudou.")
        corpo = topo + "".join(f"<div>{l}</div>" for l in linhas) + \
            "<div class='suave'>Confira a lista abaixo antes de usar esses documentos.</div>"
        return _caixa(corpo, cor=ATENCAO, fundo="rgba(244,197,66,.07)")
    return _caixa(topo + "<div class='suave'>Nenhum outro documento da base depende do que mudou.</div>",
                  cor=DESTAQUE)


def _tema() -> gr.themes.Base:
    t = gr.themes.Base(
        primary_hue=gr.themes.colors.teal,
        secondary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.slate,
        font=["General Sans", "Segoe UI", "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
        radius_size=gr.themes.sizes.radius_md,
    )
    valores = dict(
        body_background_fill=FUNDO, body_text_color=TEXTO, body_text_color_subdued=SUAVE,
        background_fill_primary=FUNDO, background_fill_secondary=PAINEL,
        block_background_fill=PAINEL, block_border_color=LINHA, block_border_width="1px",
        block_label_background_fill=PAINEL, block_label_text_color=SUAVE,
        block_label_border_color=LINHA, block_title_text_color=TEXTO, block_info_text_color=SUAVE,
        block_shadow="none", border_color_primary=LINHA, border_color_accent=DESTAQUE,
        color_accent=DESTAQUE, color_accent_soft="rgba(61,219,195,.14)",
        input_background_fill=FUNDO, input_border_color=LINHA,
        input_border_color_focus=DESTAQUE, input_placeholder_color="#6E8E94",
        input_shadow="none", input_shadow_focus="none",
        button_primary_background_fill=DESTAQUE, button_primary_background_fill_hover="#5FE6D1",
        button_primary_text_color=DESTAQUE_TINTA, button_primary_border_color=DESTAQUE,
        button_secondary_background_fill="transparent",
        button_secondary_background_fill_hover="rgba(238,243,240,.06)",
        button_secondary_text_color=TEXTO, button_secondary_border_color=LINHA_FORTE,
        button_shadow="none", button_shadow_hover="none",
        link_text_color=DESTAQUE, link_text_color_hover=TEXTO,
        table_even_background_fill=PAINEL, table_odd_background_fill=PAINEL_2,
        table_border_color=LINHA, checkbox_label_background_fill=PAINEL,
        checkbox_background_color_selected=DESTAQUE, checkbox_border_color_selected=DESTAQUE,
        panel_background_fill=PAINEL, panel_border_color=LINHA,
        code_background_fill=FUNDO,
    )
    aceitos = inspect.signature(t.set).parameters
    t.set(**{k: v for k, v in valores.items() if k in aceitos},
          **{f"{k}_dark": v for k, v in valores.items() if f"{k}_dark" in aceitos})
    return t


THEME = _tema()

# ─── Documentos e versões ─────────────────────────────────────────────────────

def _nome_doc(doc_id: Optional[str]) -> str:
    d = get_store().get_document(doc_id) if doc_id else None
    return d.name if d else ""


def _lista_docs() -> List[Tuple[str, str]]:
    """Itens da lista de documentos: 'Nome · versão N'."""
    out = []
    for d in get_store().list_documents():
        atual = get_store().get_current(d.id)
        out.append((f"{exibir(d.name)}  ·  v{atual.version}" if atual else exibir(d.name), d.id))
    return out


def _escopos() -> List[Tuple[str, str]]:
    return [("Todos os documentos", ALL_DOCS)] + [(exibir(d.name), d.id)
                                                  for d in get_store().list_documents()]


def _destinos_envio() -> List[Tuple[str, str]]:
    return [("Um documento novo", NOVO_DOC)] + \
           [(f"Nova versão de {exibir(d.name)}", d.id) for d in get_store().list_documents()]


_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _fence(text: str, lang: str = "") -> str:
    """Bloco de código que o próprio texto não consegue fechar (crases vindas do PDF)."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    f = "`" * max(3, longest + 1)
    return f"{f}{lang}\n{text}\n{f}"


def _resumir(diff: VersionDiff, nome: str, llm: Optional[LLMChain]) -> Tuple[str, bool]:
    """
    Resumo pela IA só com os artigos que mudaram; sem IA, o resumo automático.
    Devolve (texto, veio_da_ia): só o resumo da IA é gravado no histórico, para
    não ficar preso ao resumo reserva depois que a chave for configurada.
    """
    if not diff.has_changes:
        return "As duas versões têm o mesmo conteúdo.", False
    antes = "\n\n".join([c.before for c in diff.changed] + [a.text for a in diff.removed])
    depois = "\n\n".join([c.after for c in diff.changed] + [a.text for a in diff.added])
    try:
        chain = llm or _chain(None, DEFAULT_MODEL)
        texto, _ = _com_reserva(chain, lambda c: c.summarize_change(antes, depois, nome, diff.as_text()))
        # Um PDF pode tentar induzir a IA a pôr imagem externa no resumo.
        return _MD_IMAGE.sub("", texto), True
    except Exception as e:
        logger.warning(f"Resumo pela IA falhou, usando o automático: {e}")
        return ("_Resumo automático (a IA não respondeu):_\n\n"
                + "\n".join(f"- {md_escape(l)}" for l in diff.as_text().splitlines()), False)


def _mudancas_md(diff: Optional[VersionDiff]) -> str:
    """O que mudou, artigo por artigo, com o texto de antes e de depois."""
    if diff is None:
        return "_Primeira versão: ainda não há com o que comparar._"
    if not diff.has_changes:
        return "_As duas versões têm o mesmo conteúdo._"
    out = []
    for c in diff.changed:
        out.append(f"**{c.label}: mudou**")
        trocas = c.edits_text()
        if trocas:
            out.append(md_escape(trocas))
        out.append("Antes:\n\n> " + md_escape(" ".join(c.before.split())))
        out.append("Depois:\n\n> " + md_escape(" ".join(c.after.split())))
    for a in diff.removed:
        out.append(f"**{a.label}: saiu**")
        out.append("> " + md_escape(" ".join(a.text.split())))
    for a in diff.added:
        out.append(f"**{a.label}: entrou**")
        out.append("> " + md_escape(" ".join(a.text.split())))
    return "\n\n".join(out)


def _verificador(llm: Optional[LLMChain]):
    """
    Verificação por IA das sugestões da análise de impacto ("este trecho
    depende mesmo do artigo que mudou?"). Sem IA, None: vale só a regra local.
    """
    if llm is None or not hasattr(llm, "verify_dependency"):
        return None
    return llm.verify_dependency


def _analisar_impacto(doc_id: str, diff: VersionDiff, llm: Optional[LLMChain]) -> ImpactReport:
    verificar = _verificador(llm)
    params = inspect.signature(analyze_impact).parameters
    if verificar is not None and "verificar" in params:
        return analyze_impact(get_store(), get_retriever(), doc_id, diff, verificar=verificar)
    return analyze_impact(get_store(), get_retriever(), doc_id, diff)


def _nome_legivel(arquivo_nome: str) -> str:
    """'regulamento_v1.pdf' -> 'Regulamento'; 'Edital_PPGI_2027.pdf' -> 'Edital PPGI 2027'."""
    base = re.sub(r"[_\-]+", " ", Path(arquivo_nome).stem)
    base = re.sub(r"\s+(v|versao|versão|rev)\s*\d+$", "", base, flags=re.I).strip()
    return (base[:1].upper() + base[1:]) if base else Path(arquivo_nome).stem


def _registrar(arquivo, destino: str, nome_novo: str, autor: str, motivo: str, apelidos: str,
               llm: Optional[LLMChain]):
    """Registra o PDF como versão nova e calcula as mudanças, o resumo e o impacto."""
    caminho = arquivo.get("name") if isinstance(arquivo, dict) else arquivo
    texto = extract_text_from_pdf(str(caminho))
    if not texto:
        raise ValueError("o PDF não tem texto que dê para ler (pode ser uma imagem escaneada).")
    arquivo_nome = Path(caminho).name
    store = get_store()
    existente = store.get_document(destino) if destino and destino != NOVO_DOC else None
    nome = existente.name if existente else ((nome_novo or "").strip() or _nome_legivel(arquivo_nome))
    lista_apelidos = [a.strip() for a in (apelidos or "").split(",") if a.strip()]

    res = publish_version(
        store, get_retriever(), name=nome, text=texto, filename=arquivo_nome,
        sha256=sha256_file(caminho), author=autor or "", reason=(motivo or "").strip(),
        aliases=[nome] + lista_apelidos, document_id=existente.id if existente else None,
    )
    if res.status == "created":
        _guardar_pdf(caminho, res.document.id, res.version.version)
    impacto = resumo = None
    if res.status == "created" and res.diff is not None:
        impacto = _analisar_impacto(res.document.id, res.diff, llm)
        resumo, da_ia = _resumir(res.diff, res.document.name, llm)
        if da_ia:
            store.set_summary(res.document.id, res.version.version, resumo)
    return res, impacto, resumo


# ─── PDFs guardados ───────────────────────────────────────────────────────────
# Cada versão guarda o PDF original ao lado do registro de versões
# (data/pdfs/<documento>/v<N>.pdf), para a tela mostrar o arquivo de verdade.

def pasta_pdfs() -> Path:
    return (get_store().db_path.parent / "pdfs").resolve()


def _pdf_da_versao(doc_id: str, versao: int) -> Path:
    return pasta_pdfs() / doc_id / f"v{versao}.pdf"


def _guardar_pdf(origem, doc_id: str, versao: int) -> None:
    try:
        destino = _pdf_da_versao(doc_id, versao)
        destino.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(origem), destino)
    except OSError as e:                # sem o PDF, a tela mostra o texto extraído
        logger.warning(f"Não deu para guardar o PDF de {doc_id} v{versao}: {e}")


def _url_arquivo(caminho: Path) -> str:
    """Endereço que o Gradio serve (a pasta dos PDFs entra em allowed_paths no launch)."""
    return "/gradio_api/file=" + quote(caminho.as_posix(), safe="/:")


def _visualizacao(doc_id: Optional[str]) -> str:
    """O PDF da versão atual dentro da página, com o leitor do próprio navegador."""
    if not doc_id:
        return ""
    atual = get_store().get_current(doc_id)
    if atual is None:
        return ""
    pdf = _pdf_da_versao(doc_id, atual.version)
    if not pdf.exists():
        return ("<div class='suave aviso-pdf'>A pré-visualização não está disponível: este "
                "arquivo foi enviado antes de o EixoAI guardar os PDFs. Abaixo, o texto extraído "
                "dele. Envie o PDF de novo como nova versão para ver o arquivo aqui.</div>")
    url = _url_arquivo(pdf)
    return (f"<iframe class='visor-pdf' src='{url}#view=FitH' title='PDF'></iframe>"
            f"<div class='suave'><a href='{url}' target='_blank' rel='noopener'>Abrir o PDF "
            f"em outra aba</a></div>")


def _paragrafos(texto: str) -> List[str]:
    """
    Junta as linhas do texto extraído em parágrafos. Alguns PDFs (o edital do
    PPGI, por exemplo) saem com uma quebra de linha depois de cada palavra.
    O mesmo edital põe uma linha só com espaço entre as palavras, então linha em
    branco só fecha o parágrafo se a anterior termina em pontuação de fim de
    frase. Também começa parágrafo novo quando a linha anterior termina em ponto
    e a atual começa com maiúscula, número, "Art." ou "§".
    """
    paragrafos, atual = [], []
    for linha in texto.splitlines():
        linha = linha.strip()
        if not linha:
            if atual and atual[-1].endswith((".", ":", ";", "!", "?")):
                paragrafos.append(" ".join(atual))
                atual = []
            continue
        novo = bool(atual) and atual[-1].endswith((".", ":", ";", "!", "?")) and \
            (linha[:1].isupper() or linha[:1].isdigit() or linha.startswith(("Art.", "§")))
        if novo:
            paragrafos.append(" ".join(atual))
            atual = []
        atual.append(linha)
    if atual:
        paragrafos.append(" ".join(atual))
    return [" ".join(p.split()) for p in paragrafos]


def _texto_atual_md(doc_id: Optional[str], so_sem_pdf: bool = False) -> str:
    """Texto da versão atual, com cada artigo em destaque, para ler na tela."""
    if not doc_id:
        return ""
    atual = get_store().get_current(doc_id)
    if atual is None:
        return "_Sem versão registrada._"
    if so_sem_pdf and _pdf_da_versao(doc_id, atual.version).exists():
        return ""
    texto = get_store().get_version(doc_id, atual.version).text
    artigos = split_articles(texto)
    if len(artigos) <= 1:
        return "\n\n".join(md_escape(p) for p in _paragrafos(texto))
    out = []
    for a in artigos:
        corpo = " ".join(a.text.split())
        rotulo = a.label if a.key != "preambulo" else ""
        if rotulo and corpo.startswith("Art."):
            # "Art. 12. O reembolso..." -> rótulo em negrito, resto normal
            cab, _, resto = corpo.partition(" ")
            num, _, resto = resto.partition(" ")
            out.append(f"**{md_escape(cab + ' ' + num)}** {md_escape(resto)}")
        else:
            out.append(md_escape(corpo))
    return "\n\n".join(out)


def _cabecalho_doc(doc_id: Optional[str]) -> str:
    if not doc_id:
        return ("<div class='doc-cab vazio'>Escolha um documento na lista, ou envie o primeiro "
                "arquivo em <b>Enviar arquivo</b>.</div>")
    store = get_store()
    d = store.get_document(doc_id)
    vs = store.list_versions(doc_id)
    if d is None or not vs:
        return "<div class='doc-cab vazio'>Documento não encontrado.</div>"
    atual = next((v for v in vs if v.is_current), vs[-1])
    por = html.escape(atual.author or "alguém")
    return (f"<div class='doc-cab'><div class='doc-nome'>{html.escape(exibir(d.name))}</div>"
            f"<div class='suave'>Versão {atual.version} de {len(vs)} · atualizada "
            f"{quando_humano(atual.created_at)} por {por}"
            + (f" · {html.escape(atual.reason)}" if atual.reason else "") + "</div></div>")


def _historico(doc_id: Optional[str]) -> str:
    """Tabela das versões (a mais nova em cima). Texto longo quebra em vez de vazar."""
    if not doc_id:
        return ""
    vs = list(reversed(get_store().list_versions(doc_id)))
    linhas = []
    for v in vs:
        pdf = _pdf_da_versao(doc_id, v.version)
        nome = html.escape(v.filename)
        arquivo = (f"<a href='{_url_arquivo(pdf)}' target='_blank' rel='noopener'>{nome}</a>"
                   if pdf.exists() else nome)
        atual = " <span class='selo'>atual</span>" if v.is_current else ""
        linhas.append(
            f"<tr><td><b>{v.version}</b>{atual}</td><td>{quando_humano(v.created_at)}</td>"
            f"<td>{html.escape(v.author or '—')}</td><td>{html.escape(v.reason or '—')}</td>"
            f"<td class='arquivo'>{arquivo}<div class='codigo' title='Código de verificação "
            f"(SHA-256): muda se o arquivo mudar'>{v.sha256[:12]}</div></td></tr>")
    nota = ("" if len(vs) > 1 else "<div class='suave nota-hist'>Só há uma versão. Quando alguém "
            "enviar outra, dá para comparar as duas aqui embaixo.</div>")
    return ("<table class='historico'><colgroup><col style='width:13%'><col style='width:16%'>"
            "<col style='width:17%'><col style='width:24%'><col style='width:30%'></colgroup>"
            "<thead><tr><th>Versão</th><th>Quando</th><th>Quem enviou</th><th>O que mudou</th>"
            f"<th>Arquivo</th></tr></thead><tbody>{''.join(linhas)}</tbody></table>{nota}")


def _versoes_escolha(doc_id: Optional[str]) -> List[Tuple[str, int]]:
    if not doc_id:
        return []
    return [(f"Versão {v.version} · {quando_humano(v.created_at)} · {v.author or 'sem autor'}",
             v.version) for v in get_store().list_versions(doc_id)]


def _comparar(doc_id: str, va: int, vb: int, llm: Optional[LLMChain]):
    """(alerta, resumo, impacto, mudanças) entre duas versões já registradas."""
    store = get_store()
    a, b = store.get_version(doc_id, int(va)), store.get_version(doc_id, int(vb))
    if a is None or b is None:
        return _aviso_erro("versão não encontrada."), "", "", ""
    diff = compare_versions(a.text, b.text)
    if b.change_summary and a.version == b.version - 1:
        resumo = b.change_summary                       # já gerado no envio
    else:
        resumo, da_ia = _resumir(diff, store.get_document(doc_id).name, llm)
        if da_ia and a.version == b.version - 1:
            store.set_summary(doc_id, b.version, resumo)
    impacto = _analisar_impacto(doc_id, diff, llm)
    alerta = _alerta_mudanca(diff, impacto, store.get_document(doc_id).name, a.version, b.version)
    return alerta, resumo, format_report(impacto), _mudancas_md(diff)

# ─── Conversa da equipe ───────────────────────────────────────────────────────

def _iniciais(nome: str) -> str:
    partes = [p for p in re.split(r"\s+", nome.strip()) if p]
    return (partes[0][0] + (partes[1][0] if len(partes) > 1 else "")).upper() if partes else "?"


def _conversa_html(sala: str, eu: str, vazio: str) -> str:
    msgs = get_equipe().conversa(sala)
    if not msgs:
        return f"<div class='conversa'><div class='conversa-vazia'>{html.escape(vazio)}</div></div>"
    itens = []
    for m in reversed(msgs):          # a lista é column-reverse: fica rolada no fim
        minha = " minha" if m.autor == eu else ""
        itens.append(
            f"<div class='msg{minha}'><div class='msg-topo'><span class='avatar'>"
            f"{html.escape(_iniciais(m.autor))}</span><b>{html.escape(m.autor)}</b>"
            f"<span class='suave'>{quando_humano(m.criado_em)}</span></div>"
            f"<div class='msg-texto'>{html.escape(m.texto)}</div></div>")
    return f"<div class='conversa'>{''.join(itens)}</div>"


def _online_html(eu: str) -> str:
    pessoas = get_equipe().online()
    if not pessoas:
        return "<div class='suave'>Ninguém por aqui agora.</div>"
    linhas = []
    for nome, vendo in pessoas:
        voce = " (você)" if nome == eu else ""
        linhas.append(f"<div class='pessoa'><span class='ponto'></span><span class='avatar'>"
                      f"{html.escape(_iniciais(nome))}</span><div><b>{html.escape(nome)}{voce}</b>"
                      f"<div class='suave'>{html.escape(vendo or 'com o app aberto')}</div></div></div>")
    return "".join(linhas)


def _atividade_html(admin: bool = False) -> str:
    eventos = get_equipe().atividade(25, incluir_ia=admin)
    if not eventos:
        return "<div class='suave'>Nada por enquanto. Envie um documento ou escreva na conversa.</div>"
    return "".join(
        f"<div class='evento {e.tipo}'><span class='suave'>{quando_humano(e.quando)}</span>"
        f"<div><b>{html.escape(e.quem)}</b> {html.escape(e.texto)}</div></div>" for e in eventos)


def _topo_html(eu: str) -> str:
    pessoas = get_equipe().online()
    chips = "".join(
        f"<span class='chip' title='{html.escape(n)}: {html.escape(v or 'com o app aberto')}'>"
        f"<span class='ponto'></span>{html.escape(n.split()[0])}</span>"
        for n, v in pessoas if n != eu)
    nao_lidas = sum(get_equipe().nao_lidas(eu).values())
    aviso = (f"<span class='chip atencao'>{nao_lidas} mensage{'ns' if nao_lidas > 1 else 'm'} "
             f"nova{'s' if nao_lidas > 1 else ''}</span>") if nao_lidas else ""
    sair = " · <a href='/logout'>sair</a>" if CONTAS else ""
    return (f"<div class='topo'><div><div class='marca'>EixoAI</div>"
            f"<div class='suave'>Documentos da equipe, versões e perguntas</div></div>"
            f"<div class='topo-dir'>{aviso}{chips}<span class='eu'>{html.escape(eu)}{sair}</span></div></div>")

# ─── Perguntar ────────────────────────────────────────────────────────────────

# 8 trechos e não 5: no edital do PPGI, "ate que dia eh a inscricao do mestrado" deixava
# o item 1.1 (o período de inscrição) em 7º lugar, e a IA respondia que não havia a data.
TRECHOS_POR_PERGUNTA = 8


def _buscar(pergunta: str, llm: Optional[LLMChain], doc_filtro: Optional[str]) -> List[Dict]:
    """
    Busca híbrida (significado + palavras exatas, com correção de erro de
    digitação) em busca.py. A IA, se houver, só acrescenta variantes da
    pergunta; sem ela, ou se ela demorar, a busca segue igual.
    Antes, a pergunta era reescrita com "termos da Constituição Federal" e
    isso piorava a busca (Hit@5 de 71,7% para 51,7% nas paráfrases).
    """
    reescrever = getattr(llm, "reescrever_pergunta", None) if llm else None
    return buscar(get_retriever(), pergunta, document_id=doc_filtro, k=TRECHOS_POR_PERGUNTA,
                  reescrever=reescrever)


def _fontes_html(resultados: List[Dict]) -> str:
    if not resultados:
        return ("<div class='suave'>Nenhum trecho dos documentos parece responder a essa "
                "pergunta. Tente com outras palavras ou escolha um documento em "
                "<b>Procurar em</b>.</div>")
    nomes = {d.id: d.name for d in get_store().list_documents()}
    itens = []
    for r in resultados:
        rel = r.get("relevance", 0) or 0
        nivel = "alta" if rel >= 0.6 else ("média" if rel >= 0.35 else "baixa")
        texto = " ".join(str(r.get("text", "")).split())
        if len(texto) > 420:
            texto = texto[:417] + "..."
        doc = nomes.get(r.get("document_id"), r.get("source", ""))
        pag = r.get("page")
        onde = f" · página {pag}" if pag not in (None, "", 0, "0") else ""
        itens.append(f"<div class='fonte'><div class='fonte-topo'><b>{html.escape(str(doc))}</b>"
                     f"<span class='suave'>{onde} · relação {nivel}</span></div>"
                     f"<div class='fonte-texto'>{html.escape(texto)}</div></div>")
    return "".join(itens)


SYSTEM_PROMPT = (
    "Você responde perguntas sobre os documentos de uma equipe (regulamentos, contratos, "
    "manuais, editais) para pessoas que não são da área jurídica. "
    "Use só o contexto fornecido. Escreva em português do Brasil, em frases curtas e "
    "palavras simples, sem emoji. Diga de qual documento e de qual artigo ou item veio a "
    "informação (ex.: 'Pelo Regulamento Interno, art. 12, ...'). "
    "Entenda a pergunta mesmo escrita de modo informal, sem acento ou com erro de digitação "
    "('ate que dia eh a inscricao' = 'qual o prazo final das inscrições'). "
    "Se o contexto não tiver a resposta, responda SOMENTE com duas linhas: na primeira, "
    "exatamente [SEM_RESPOSTA]; na segunda, 'BUSCAR: ' e de 3 a 5 termos curtos separados "
    "por ponto e vírgula, com as palavras que provavelmente aparecem no trecho do documento "
    "que responde (sinônimos, nome da seção, por exemplo 'período de inscrição; cronograma')."
)
SEM_RESPOSTA = "[SEM_RESPOSTA]"
MAX_CONTEXT_CHARS = 12_000


def _sem_resposta(texto: str) -> bool:
    return SEM_RESPOSTA in (texto or "")


def _termos_sugeridos(texto: str) -> List[str]:
    m = re.search(r"BUSCAR:\s*(.+)", texto or "", flags=re.S)
    if not m:
        return []
    termos = [t.strip(" .\"'`") for t in re.split(r"[;\n]", m.group(1))]
    return [t for t in termos if 2 < len(t) < 80][:5]


def _desfazer_ultima_troca(llm: LLMChain) -> None:
    """Tira do histórico a última pergunta e resposta (a tentativa que não achou)."""
    h = llm.conversation_history
    if len(h) >= 2 and h[-1]["role"] == "assistant" and h[-2]["role"] == "user":
        del h[-2:]


def _responder(llm: LLMChain, pergunta: str, contexto: str):
    return _com_reserva(llm, lambda c: c.generate_response(
        user_message=pergunta, system_prompt=SYSTEM_PROMPT, context=contexto[:MAX_CONTEXT_CHARS]))


def _resumo_fontes(resultados: List[Dict]) -> List[Dict]:
    nomes = {d.id: exibir(d.name) for d in get_store().list_documents()}
    out = []
    for r in resultados:
        texto = " ".join(str(r.get("text", "")).split())
        out.append({"documento": nomes.get(r.get("document_id"), str(r.get("source", ""))),
                    "pagina": r.get("page"), "trecho": texto[:300]})
    return out


def perguntar(pergunta: str, historico: List[Dict], llm: Optional[LLMChain], modelo: str,
              escopo: str, conversa_id: Optional[str], request: gr.Request):
    """
    Busca, responde e, se a IA não achou a resposta nos trechos, tenta de novo:
    ela devolve os termos que procuraria, o app busca esses termos, junta os
    trechos novos e pergunta outra vez. Cada pergunta e resposta fica guardada
    (equipe.py) para os admins verem as conversas da equipe com a IA.
    """
    pergunta = (pergunta or "").strip()
    if not pergunta:
        return "", historico, gr.skip(), llm, conversa_id
    conversa_id = conversa_id or uuid.uuid4().hex
    llm = _chain(llm, modelo)
    doc_filtro = None if (not escopo or escopo == ALL_DOCS) else escopo
    fontes = ""
    resultados: List[Dict] = []
    if not get_store().has_documents():
        resposta = ("A base ainda está vazia. Envie um documento na aba Documentos e "
                    "depois pergunte aqui.")
        fontes = "<div class='suave'>Sem documentos na base.</div>"
    else:
        try:
            resultados = _buscar(pergunta, llm, doc_filtro)
        except Exception as e:
            logger.warning(f"Erro na busca: {e}")
            fontes = _aviso_erro(f"a busca falhou ({e}).")
        contexto = get_retriever().format_context(resultados) if resultados else ""
        extra = ""
        if llm is None:
            resposta = ("A IA não está disponível agora (falta a chave do Groq no arquivo .env). "
                        "Os trechos que respondem à pergunta estão ao lado.")
        else:
            try:
                resposta, aviso = _responder(llm, pergunta, contexto)
                if _sem_resposta(resposta):
                    termos = _termos_sugeridos(resposta)
                    _desfazer_ultima_troca(llm)
                    vistos = {r.get("id") for r in resultados}
                    novos: List[Dict] = []
                    for t in termos:
                        for r in buscar(get_retriever(), t, document_id=doc_filtro, k=4):
                            if r.get("id") not in vistos:
                                vistos.add(r.get("id"))
                                novos.append(r)
                    if novos:
                        resultados = resultados + novos[:6]
                        contexto = get_retriever().format_context(resultados)
                        resposta, aviso = _responder(llm, pergunta, contexto)
                    if termos:
                        extra = ("<div class='suave busca-extra'>Não achei de primeira; procurei "
                                 "também por: " + html.escape("; ".join(termos)) + ".</div>")
                    if _sem_resposta(resposta):
                        resposta = ("Não encontrei essa informação nos documentos da base"
                                    + (f" (procurei também por: {'; '.join(termos)})" if termos else "")
                                    + ". Tente perguntar de outro jeito ou escolha um documento em "
                                    "Procurar em.")
                        if llm.conversation_history and llm.conversation_history[-1]["role"] == "assistant":
                            llm.conversation_history[-1]["content"] = resposta
                if aviso:
                    resposta += f"\n\n_{aviso}_"
            except IAOcupada:
                resposta = ("Os modelos de IA estão no limite de uso por minuto da conta gratuita "
                            "do Groq. Espere alguns segundos e pergunte de novo. Enquanto isso, os "
                            "trechos que respondem à pergunta estão ao lado.")
            except Exception as e:
                logger.error(f"Erro na IA: {e}")
                resposta = ("A IA não conseguiu responder agora. Os trechos encontrados estão ao "
                            "lado; tente de novo em instantes ou escolha outro modelo no topo.")
        if not fontes:
            fontes = extra + _fontes_html(resultados)
    try:
        nome_escopo = "Todos os documentos" if not doc_filtro else exibir(_nome_doc(doc_filtro))
        get_equipe().registrar_pergunta(conversa_id, quem(request), pergunta, resposta,
                                        escopo=nome_escopo, modelo=llm.model if llm else "",
                                        fontes=_resumo_fontes(resultados))
    except Exception as e:
        logger.warning(f"Não deu para guardar a pergunta: {e}")
    historico = historico + [{"role": "user", "content": pergunta},
                             {"role": "assistant", "content": resposta}]
    return "", historico, fontes, llm, conversa_id


def nova_conversa(llm):
    if llm is not None:
        llm.clear_history()
    return ([], "<div class='suave'>As fontes de cada resposta aparecem aqui.</div>", llm,
            uuid.uuid4().hex)


# ─── Conversas da equipe com a IA (só admin) ──────────────────────────────────

_MD = MarkdownIt("commonmark", {"html": False}).enable("table").disable("image")
VER_MINHA, VER_EQUIPE = "Minha conversa", "Conversas da equipe"


def _conversas_ia_html(autor: str, eu: str) -> str:
    convs = get_equipe().conversas_ia(autor or None)
    if not convs:
        return "<div class='suave'>Ninguém conversou com a IA ainda.</div>"
    blocos = []
    for c in convs:
        n = len(c.perguntas)
        primeira = c.perguntas[0].pergunta
        primeira = primeira if len(primeira) <= 110 else primeira[:107] + "..."
        voce = " (você)" if c.autor == eu else ""
        itens = []
        for q in c.perguntas:
            fontes = "; ".join(dict.fromkeys(      # sem repetir o mesmo documento e página
                f"{f.get('documento', '')}" + (f", p. {f['pagina']}" if f.get("pagina") not in (None, "", 0, "0") else "")
                for f in q.fontes))
            itens.append(
                f"<div class='ia-item'><div class='suave'>{quando_humano(q.criado_em)} · "
                f"{html.escape(q.escopo or 'Todos os documentos')} · {html.escape(_rotulo_modelo(q.modelo) if q.modelo else '')}</div>"
                f"<div class='ia-pergunta'>{html.escape(q.pergunta)}</div>"
                f"<div class='ia-resposta'>{_MD.render(q.resposta)}</div>"
                + (f"<div class='suave ia-fontes'>Fontes: {html.escape(fontes)}</div>" if fontes else "")
                + "</div>")
        blocos.append(
            f"<details class='conv-ia'><summary><span class='avatar'>{html.escape(_iniciais(c.autor))}</span>"
            f"<div class='conv-topo'><div><b>{html.escape(c.autor)}{voce}</b> "
            f"<span class='suave'>· {quando_humano(c.fim)} · {n} pergunta{'s' if n > 1 else ''}</span></div>"
            f"<div class='conv-titulo'>{html.escape(primeira)}</div></div></summary>"
            f"{''.join(itens)}</details>")
    return "".join(blocos)


def _filtro_pessoas() -> List[Tuple[str, str]]:
    return [("Todas as pessoas", "")] + [(a, a) for a in get_equipe().autores_ia()]


def mostrar_ver_ia(ver):
    return gr.update(visible=ver != VER_EQUIPE), gr.update(visible=ver == VER_EQUIPE)


def preencher_ia(ver, autor, request: gr.Request):
    if ver != VER_EQUIPE:
        return gr.skip(), gr.skip(), gr.skip()
    if not eh_admin(request):       # confere no servidor, não só esconde o botão
        return ("<div class='suave'>Só quem tem cargo de admin vê as conversas da equipe.</div>",
                gr.skip(), gr.skip())
    opcoes = _filtro_pessoas()
    autor = autor if autor in [v for _, v in opcoes] else ""
    return (_conversas_ia_html(autor, quem(request)), gr.update(choices=opcoes, value=autor),
            get_equipe().ultima_pergunta_ia())

# ─── Documentos: navegação, envio e comparação ────────────────────────────────

def _painel_doc(doc_id: Optional[str], secao: str, eu: str) -> Dict:
    """Atualizações do painel do documento: cabeçalho e a seção visível."""
    out = {doc_cab: _cabecalho_doc(doc_id)}
    if secao == "Documento":
        out[doc_pdf] = _visualizacao(doc_id)
        out[doc_texto] = _texto_atual_md(doc_id, so_sem_pdf=True)
    elif secao == "Histórico":
        out[doc_hist] = _historico(doc_id)
        vs = _versoes_escolha(doc_id)
        out[cmp_a] = gr.update(choices=vs, value=vs[-2][1] if len(vs) > 1 else (vs[0][1] if vs else None))
        out[cmp_b] = gr.update(choices=vs, value=vs[-1][1] if vs else None)
    elif secao == "Conversa":
        if doc_id:
            get_equipe().marcar_lida(eu, doc_id)
        out[doc_conversa] = _conversa_html(doc_id, eu, "Ninguém escreveu sobre este documento ainda.") \
            if doc_id else ""
    elif secao == "Mudanças":
        pass                           # preenchida por envio ou comparação
    return out


def _visibilidade(secao: str) -> Dict:
    return {sec_texto: gr.update(visible=secao == "Documento"),
            sec_mud: gr.update(visible=secao == "Mudanças"),
            sec_hist: gr.update(visible=secao == "Histórico"),
            sec_conv: gr.update(visible=secao == "Conversa")}


# A seção é trocada em dois passos: primeiro aparece (mostrar_secao), depois
# recebe o conteúdo (preencher). Conteúdo mandado para uma coluna escondida no
# mesmo passo em que ela aparece se perdia no Gradio 6.3 (o histórico vinha vazio).
# O que já foi calculado num envio ou numa comparação vai pelo `pendente`.

def mostrar_secao(secao):
    return _visibilidade(secao)


def _ultima_mudanca(doc_id: Optional[str], llm: Optional[LLMChain]) -> Dict:
    """Seção Mudanças sem comparação escolhida: mostra a última mudança do documento."""
    if not doc_id:
        return {mud_alerta: "", mud_resumo: "", mud_impacto: "", mud_detalhe: ""}
    vs = get_store().list_versions(doc_id)
    if len(vs) < 2:
        return {mud_alerta: _caixa("Este documento só tem uma versão. Quando alguém enviar "
                                   "uma versão nova, as mudanças aparecem aqui."),
                mud_resumo: "", mud_impacto: "", mud_detalhe: ""}
    alerta, resumo, impacto, detalhe = _comparar(doc_id, vs[-2].version, vs[-1].version, llm)
    return {mud_alerta: alerta, mud_resumo: resumo, mud_impacto: impacto, mud_detalhe: detalhe}


def preencher(secao, doc_id, pendente, modelo, llm, request: gr.Request):
    eu = quem(request)
    out = _painel_doc(doc_id, secao, eu)
    if secao == "Mudanças":
        if pendente and pendente.get("doc") == doc_id:
            out.update({mud_alerta: pendente["alerta"], mud_resumo: pendente["resumo"],
                        mud_impacto: pendente["impacto"], mud_detalhe: pendente["detalhe"]})
        else:
            llm = _chain(llm, modelo)
            out.update(_ultima_mudanca(doc_id, llm))
    out[pendente_state] = None
    out[llm_state] = llm
    return out


def comparar(doc_id, va, vb, modelo, llm):
    llm = _chain(llm, modelo)
    if not doc_id or va is None or vb is None:
        return {pendente_state: None, llm_state: llm}
    alerta, resumo, impacto, detalhe = _comparar(doc_id, va, vb, llm)
    out = {pendente_state: {"doc": doc_id, "alerta": alerta, "resumo": resumo, "impacto": impacto,
                            "detalhe": detalhe},
           llm_state: llm, secao: "Mudanças"}
    out.update(_visibilidade("Mudanças"))
    return out


def enviar_arquivo(arquivo, destino, nome_novo, motivo, apelidos, modelo, llm, mine,
                   request: gr.Request):
    eu = quem(request)
    if arquivo is None:
        return {envio_status: _aviso_erro("escolha um arquivo PDF.")}
    llm = _chain(llm, modelo)
    try:
        res, impacto, resumo = _registrar(arquivo, destino, nome_novo, eu, motivo, apelidos, llm)
    except Exception as e:
        logger.error(f"Erro ao registrar versão: {e}")
        return {envio_status: _aviso_erro(str(e)), llm_state: llm}

    doc = res.document
    out = {envio_status: "", envio_arquivo: None, envio_motivo: "", envio_nome: "",
           envio_destino: gr.update(choices=_destinos_envio(), value=NOVO_DOC),
           envio_nome_box: gr.update(visible=True),
           doc_lista: gr.update(choices=_lista_docs(), value=doc.id),
           llm_state: llm, mine_state: mine + [f"{doc.id}:{res.version.version}"]}
    if res.status == "unchanged":
        out[envio_status] = _caixa(
            f"Esse arquivo é igual à versão atual de {html.escape(doc.name)} "
            f"(versão {res.version.version}). Nada foi criado.")
        return out
    v_antes = res.previous.version if res.previous else None
    nova_secao = "Mudanças" if v_antes else "Documento"
    out[secao] = nova_secao
    out.update(_visibilidade(nova_secao))
    if v_antes:
        out[pendente_state] = {
            "doc": doc.id,
            "alerta": _alerta_mudanca(res.diff, impacto, doc.name, v_antes, res.version.version),
            "resumo": resumo or "", "impacto": format_report(impacto) if impacto else "",
            "detalhe": _mudancas_md(res.diff) if res.diff else ""}
    else:
        out[envio_status] = _aviso_ok(f"{doc.name} foi adicionado à base.")
    return out


def mudar_destino(destino):
    return gr.update(visible=destino == NOVO_DOC)

# ─── Equipe ───────────────────────────────────────────────────────────────────

def escrever_geral(texto, request: gr.Request):
    eu = quem(request)
    get_equipe().enviar(SALA_GERAL, eu, texto)
    get_equipe().marcar_lida(eu, SALA_GERAL)
    return "", _conversa_html(SALA_GERAL, eu, "Ninguém escreveu ainda. Comece a conversa."), \
        _atividade_html(eh_admin(request))


def escrever_doc(texto, doc_id, request: gr.Request):
    eu = quem(request)
    if not doc_id:
        return texto, _aviso_erro("escolha um documento na lista.")
    get_equipe().enviar(doc_id, eu, texto)
    get_equipe().marcar_lida(eu, doc_id)
    return "", _conversa_html(doc_id, eu, "Ninguém escreveu sobre este documento ainda.")

# ─── Abas e tempo real ────────────────────────────────────────────────────────

def _vendo(aba: str, doc_id: Optional[str], sec: str) -> str:
    if aba == "documentos":
        nome = _nome_doc(doc_id)
        return f"Documentos: {nome} ({sec.lower()})" if nome else "Documentos"
    return {"perguntar": "Perguntando à IA", "equipe": "Na conversa da equipe"}.get(aba, "")


def ao_abrir(request: gr.Request):
    eu = quem(request)
    get_equipe().sinal(eu, "Perguntando à IA")
    docs = _lista_docs()
    return {
        topo: _topo_html(eu),
        modelo: gr.update(choices=modelos_disponiveis(), value=DEFAULT_MODEL),
        escopo: gr.update(choices=_escopos(), value=ALL_DOCS),
        doc_lista: gr.update(choices=docs, value=docs[0][1] if docs else None),
        envio_destino: gr.update(choices=_destinos_envio(), value=NOVO_DOC),
        visto_versao: get_store().last_change(),
        visto_msg: get_equipe().ultima_mensagem(),
        conversa_state: uuid.uuid4().hex,
        ver_ia: gr.update(visible=eh_admin(request), value=VER_MINHA),
    }


def abrir_perguntar(escopo_atual, request: gr.Request):
    get_equipe().sinal(quem(request), "Perguntando à IA")
    ids = [v for _, v in _escopos()]
    return "perguntar", gr.update(choices=_escopos(),
                                  value=escopo_atual if escopo_atual in ids else ALL_DOCS)


def abrir_documentos(doc_id, sec, request: gr.Request):
    eu = quem(request)
    docs = _lista_docs()
    ids = [v for _, v in docs]
    sel = doc_id if doc_id in ids else (ids[0] if ids else None)
    get_equipe().sinal(eu, _vendo("documentos", sel, sec))
    out = {aba: "documentos", doc_lista: gr.update(choices=docs, value=sel),
           envio_destino: gr.update(choices=_destinos_envio())}
    out.update(_painel_doc(sel, sec, eu))
    return out


def abrir_equipe(request: gr.Request):
    eu = quem(request)
    get_equipe().sinal(eu, "Na conversa da equipe")
    get_equipe().marcar_lida(eu, SALA_GERAL)
    return ("equipe", _conversa_html(SALA_GERAL, eu, "Ninguém escreveu ainda. Comece a conversa."),
            _online_html(eu), _atividade_html(eh_admin(request)), _topo_html(eu))


def a_cada_instante(v_visto, m_visto, mine, aba_atual, doc_id, sec, escopo_atual, topo_antes,
                    online_antes, ver, autor_ia, ia_visto, request: gr.Request):
    """
    Roda a cada REFRESH_SECONDS em cada tela aberta: dá sinal de presença,
    avisa o que os outros fizeram e atualiza só a aba aberta.
    """
    eu = quem(request)
    eq, store = get_equipe(), get_store()
    eq.sinal(eu, _vendo(aba_atual, doc_id, sec))

    v_ultimo, m_ultimo = store.last_change(), eq.ultima_mensagem()
    mudou_v, mudou_m = v_ultimo != v_visto, m_ultimo != m_visto
    # Sempre ao menos uma chave: dicionário vazio o Gradio lê como "um valor só".
    out: Dict = {visto_versao: v_ultimo}

    if mudou_v:
        for _, nome, v in store.versions_since(v_visto or 0):
            if f"{v.document_id}:{v.version}" in mine:
                continue
            acao = f"enviou {nome}" if v.version == 1 else f"publicou a versão {v.version} de {nome}"
            gr.Info(f"{v.author or 'Alguém'} {acao}.", duration=20)
        out[visto_versao] = v_ultimo
    if mudou_m:
        for m_id in range((m_visto or 0) + 1, m_ultimo + 1):
            m = eq.mensagem(m_id)
            if m is None or m.autor == eu:
                continue
            vendo_sala = (aba_atual == "equipe" and m.sala == SALA_GERAL) or \
                         (aba_atual == "documentos" and sec == "Conversa" and m.sala == doc_id)
            if not vendo_sala:
                onde = "" if m.sala == SALA_GERAL else f" sobre {_nome_doc(m.sala)}"
                texto = m.texto if len(m.texto) <= 90 else m.texto[:87] + "..."
                gr.Info(f"{m.autor}{onde}: {texto}", duration=10)
        out[visto_msg] = m_ultimo

    novo_topo = _topo_html(eu)
    if novo_topo != topo_antes:
        out[topo] = novo_topo
        out[topo_cache] = novo_topo

    admin = eh_admin(request)
    ia_ultima = eq.ultima_pergunta_ia()
    if aba_atual == "perguntar":
        if mudou_v:
            ids = [v for _, v in _escopos()]
            out[escopo] = gr.update(choices=_escopos(),
                                    value=escopo_atual if escopo_atual in ids else ALL_DOCS)
        if admin and ver == VER_EQUIPE and ia_ultima != ia_visto:
            out[ia_lista] = _conversas_ia_html(autor_ia or "", eu)
            out[ia_filtro] = gr.update(choices=_filtro_pessoas())
            out[ia_visto_state] = ia_ultima
    elif aba_atual == "documentos":
        if mudou_v:
            out[doc_lista] = gr.update(choices=_lista_docs())
            out[envio_destino] = gr.update(choices=_destinos_envio())
            out[doc_cab] = _cabecalho_doc(doc_id)
            if sec == "Documento":
                out[doc_pdf] = _visualizacao(doc_id)
                out[doc_texto] = _texto_atual_md(doc_id, so_sem_pdf=True)
            elif sec == "Histórico":
                out.update(_painel_doc(doc_id, sec, eu))
        if mudou_m and sec == "Conversa" and doc_id:
            eq.marcar_lida(eu, doc_id)
            out[doc_conversa] = _conversa_html(doc_id, eu, "Ninguém escreveu sobre este documento ainda.")
    elif aba_atual == "equipe":
        if mudou_m:
            eq.marcar_lida(eu, SALA_GERAL)
            out[geral_conversa] = _conversa_html(SALA_GERAL, eu, "Ninguém escreveu ainda. Comece a conversa.")
        if mudou_v or mudou_m or (admin and ia_ultima != ia_visto):
            out[atividade] = _atividade_html(admin)
            out[ia_visto_state] = ia_ultima
        novo_online = _online_html(eu)
        if novo_online != online_antes:
            out[quem_aqui] = novo_online
            out[online_cache] = novo_online
    return out

# ─── CSS ──────────────────────────────────────────────────────────────────────

CUSTOM_CSS = f"""
@import url('https://api.fontshare.com/v2/css?f[]=general-sans@400,500,600&f[]=clash-display@500,600&display=swap');

:root {{ color-scheme: dark; }}
* {{ scrollbar-color: rgba(238,243,240,.22) transparent; scrollbar-width: thin; }}
::-webkit-scrollbar {{ width: 8px; height: 8px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: rgba(238,243,240,.22); border-radius: 999px; }}
::-webkit-scrollbar-corner {{ background: transparent; }}
.visor-pdf {{ width: 100%; height: calc(100vh - 300px); min-height: 420px; border: 1px solid var(--linha);
              border-radius: 10px; background: #fff; display: block; margin-bottom: 6px; }}
.aviso-pdf {{ border: 1px dashed var(--linha); border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; }}
#doc-texto {{ overflow-x: hidden; overflow-wrap: anywhere; }}
table.historico {{ width: 100%; table-layout: fixed; border-collapse: separate; border-spacing: 0;
                   font-size: 14px; border: 1px solid var(--linha); border-radius: 10px; }}
table.historico th {{ text-align: left; font-weight: 500; color: var(--suave); font-size: 13px;
                      padding: 10px 12px; border-bottom: 1px solid var(--linha); background: transparent; }}
table.historico td {{ padding: 10px 12px; border-bottom: 1px solid var(--linha); vertical-align: top;
                      overflow-wrap: anywhere; word-break: break-word; background: transparent; }}
table.historico tr:last-child td {{ border-bottom: 0; }}
table.historico th, table.historico td {{ border-left: 0 !important; border-right: 0 !important;
    border-top: 0 !important; border-color: var(--linha) !important; }}
table.historico .codigo {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--suave);
                           margin-top: 2px; }}
.nota-hist {{ margin-top: 8px; }}
.selo {{ display: inline-block; margin-left: 6px; padding: 0 8px; border-radius: 999px; font-size: 11px;
         border: 1px solid rgba(61,219,195,.5); color: var(--destaque); }}
:root {{ --fundo:{FUNDO}; --painel:{PAINEL}; --linha:{LINHA}; --texto:{TEXTO};
         --suave:{SUAVE}; --destaque:{DESTAQUE}; --atencao:{ATENCAO}; }}
.gradio-container {{ background: var(--fundo) !important; max-width: 1280px !important;
                     width: 100% !important;
                     margin: 0 auto !important; padding: 12px 20px !important; }}
footer {{ display: none !important; }}
.suave {{ color: var(--suave); font-size: 13px; }}

/* Topo */
.topo {{ display:flex; justify-content:space-between; align-items:center; gap:16px;
        padding: 6px 2px 10px; border-bottom: 1px solid var(--linha); }}
.marca {{ font-family: 'Clash Display', 'General Sans', sans-serif; font-weight: 600;
          font-size: 26px; letter-spacing: .01em; line-height: 1.1; }}
.marca::after {{ content: ''; display:inline-block; width:8px; height:8px; border-radius:50%;
                 background: var(--destaque); margin-left: 6px; }}
.topo-dir {{ display:flex; align-items:center; gap:8px; flex-wrap:wrap; justify-content:flex-end; }}
.eu {{ font-size: 13px; color: var(--suave); }}
.eu a {{ color: var(--suave) !important; }}
.chip {{ display:inline-flex; align-items:center; gap:6px; padding: 3px 10px; border-radius: 999px;
         border: 1px solid var(--linha); font-size: 12px; }}
.chip.atencao {{ border-color: var(--atencao); color: var(--atencao); }}
.ponto {{ width:7px; height:7px; border-radius:50%; background: var(--destaque); display:inline-block;
          box-shadow: 0 0 0 3px rgba(61,219,195,.15); flex-shrink:0; }}
#linha-topo {{ align-items: center; gap: 16px; border-bottom: 1px solid var(--linha);
                padding-bottom: 8px; margin-bottom: 4px; }}
#linha-topo .topo {{ border-bottom: 0; padding-bottom: 0; }}
#caixa-modelo {{ gap: 2px !important; }}
.rotulo {{ font-size: 12px; color: var(--suave); }}
button.secondary {{ border: 1px solid {LINHA_FORTE} !important; }}
.progress-text, .meta-text, .meta-text-center {{ display: none !important; }}

/* Abas */
button[role="tab"] {{ font-size: 15px !important; color: var(--suave) !important; }}
button[role="tab"][aria-selected="true"] {{ color: var(--texto) !important;
    border-bottom: 2px solid var(--destaque) !important; }}

/* Caixas de aviso */
.caixa {{ border: 1px solid; border-radius: 10px; padding: 12px 16px; line-height: 1.5;
          display:flex; flex-direction:column; gap:4px; font-size: 14px; }}
.alerta-titulo {{ font-size: 15px; }}

/* Perguntar */
#chat-ia {{ height: calc(100vh - 300px) !important; min-height: 320px; }}
#fontes {{ height: calc(100vh - 250px); min-height: 360px; overflow-y: auto; padding-right: 4px; }}
.fonte {{ border-top: 1px solid var(--linha); padding: 10px 0; }}
.fonte:first-child {{ border-top: 0; padding-top: 0; }}
.fonte-topo {{ display:flex; gap:6px; flex-wrap:wrap; align-items:baseline; margin-bottom:4px; }}
.fonte-texto {{ font-size: 13px; color: #CFDCDF; line-height: 1.5; }}

/* Documentos */
#doc-lista {{ max-height: calc(100vh - 470px); min-height: 140px; overflow-y: auto; }}
#doc-lista .wrap {{ flex-direction: column !important; gap: 4px !important; }}
#doc-lista label {{ width: 100% !important; justify-content: flex-start; min-width: 0; }}
#doc-lista label span {{ overflow-wrap: anywhere; word-break: break-word; white-space: normal; }}
#doc-lista, #doc-lista .wrap {{ overflow-x: hidden !important; }}
#ver-ia {{ margin-bottom: 6px; }}
#ia-lista {{ height: calc(100vh - 290px); min-height: 360px; overflow-y: auto; padding-right: 6px; }}
.conv-ia {{ border: 1px solid var(--linha); border-radius: 10px; margin-bottom: 8px; background: {PAINEL}; }}
.conv-ia summary {{ display: flex; gap: 10px; align-items: flex-start; padding: 10px 14px; cursor: pointer;
                    list-style: none; }}
.conv-ia summary::-webkit-details-marker {{ display: none; }}
.conv-ia[open] summary {{ border-bottom: 1px solid var(--linha); }}
.conv-topo {{ min-width: 0; }}
.conv-titulo {{ color: #CFDCDF; font-size: 14px; overflow-wrap: anywhere; }}
.ia-item {{ padding: 10px 14px; border-top: 1px solid var(--linha); }}
.ia-item:first-of-type {{ border-top: 0; }}
.ia-pergunta {{ font-weight: 500; margin: 4px 0; padding: 6px 10px; border-radius: 8px;
                border: 1px solid rgba(61,219,195,.35); background: rgba(61,219,195,.06);
                display: inline-block; max-width: 100%; overflow-wrap: anywhere; }}
.ia-resposta {{ font-size: 14px; line-height: 1.55; overflow-wrap: anywhere; }}
.ia-resposta p {{ margin: 6px 0; }}
.ia-fontes {{ margin-top: 4px; }}
.busca-extra {{ margin-bottom: 8px; }}
.doc-cab {{ padding: 2px 0 6px; }}
.doc-cab.vazio {{ color: var(--suave); padding: 24px 0; }}
.doc-nome {{ font-family: 'Clash Display', 'General Sans', sans-serif; font-size: 22px; font-weight: 500; }}
#doc-texto, #mud-rolagem {{ max-height: calc(100vh - 330px); overflow-y: auto; padding-right: 6px; }}
#doc-texto p {{ margin: 0 0 10px; line-height: 1.6; }}
#secao .wrap {{ gap: 4px !important; }}

/* Conversas */
.conversa {{ display:flex; flex-direction: column-reverse; gap: 10px; overflow-y:auto;
             height: calc(100vh - 360px); min-height: 220px; padding: 4px 4px 4px 0; }}
.conversa-vazia {{ color: var(--suave); margin: auto; }}
.msg {{ max-width: 78%; align-self: flex-start; background: {PAINEL_2}; border: 1px solid var(--linha);
        border-radius: 10px; padding: 8px 12px; }}
.msg.minha {{ align-self: flex-end; border-color: rgba(61,219,195,.35); background: rgba(61,219,195,.08); }}
.msg-topo {{ display:flex; align-items:center; gap:8px; font-size: 13px; margin-bottom: 2px; }}
.msg-texto {{ white-space: pre-wrap; line-height: 1.5; }}
.avatar {{ width: 24px; height: 24px; border-radius: 50%; border: 1px solid var(--linha-forte, {LINHA_FORTE});
           display:inline-flex; align-items:center; justify-content:center; font-size: 10px;
           font-family: 'JetBrains Mono', monospace; flex-shrink:0; }}
.pessoa {{ display:flex; align-items:center; gap: 10px; padding: 6px 0; }}
#atividade {{ max-height: calc(100vh - 470px); overflow-y: auto; }}
.evento {{ padding: 7px 0; border-top: 1px solid var(--linha); font-size: 14px; }}
.evento:first-child {{ border-top: 0; }}
.titulo-bloco {{ font-family: 'Clash Display', 'General Sans', sans-serif; font-size: 17px;
                 font-weight: 500; margin: 2px 0 6px; }}

@media (max-width: 900px) {{
  #chat-ia, #fontes, .conversa, #doc-texto, #mud-rolagem {{ height: auto !important; max-height: 60vh; }}
}}
"""

# ─── Interface ────────────────────────────────────────────────────────────────

with gr.Blocks(title="EixoAI") as demo:

    llm_state = gr.State(None)          # conversa desta pessoa com a IA
    mine_state = gr.State([])           # versões publicadas por esta sessão
    visto_versao = gr.State(0)          # última versão que esta tela já mostrou
    visto_msg = gr.State(0)             # última mensagem que esta tela já mostrou
    aba = gr.State("perguntar")         # aba aberta, para o tempo real não mexer na oculta
    topo_cache = gr.State("")
    online_cache = gr.State("")
    pendente_state = gr.State(None)     # mudanças já calculadas, esperando a seção aparecer
    conversa_state = gr.State(None)     # id da conversa atual com a IA (muda em "Nova conversa")
    ia_visto_state = gr.State(0)        # última pergunta à IA que a lista da equipe já mostrou

    with gr.Row(elem_id="linha-topo", equal_height=True):
        topo = gr.HTML(_topo_html("Visitante"), elem_id="topo")
        with gr.Column(scale=0, min_width=250, elem_id="caixa-modelo"):
            gr.HTML("<div class='rotulo'>Modelo de IA</div>")
            modelo = gr.Dropdown(show_label=False, container=False, elem_id="modelo",
                                 choices=[(DEFAULT_MODEL, DEFAULT_MODEL)], value=DEFAULT_MODEL)

    with gr.Tabs():
        # ── Perguntar ────────────────────────────────────────────────────────
        with gr.Tab("Perguntar") as tab_perguntar:
            ver_ia = gr.Radio([VER_MINHA, VER_EQUIPE], value=VER_MINHA, show_label=False,
                              container=False, visible=False, elem_id="ver-ia")
            with gr.Row(visible=False) as linha_equipe_ia:
                with gr.Column(scale=1, min_width=220):
                    ia_filtro = gr.Dropdown(label="De quem", choices=[("Todas as pessoas", "")],
                                            value="")
                    gr.HTML("<div class='suave'>Só quem tem cargo de admin vê esta lista. "
                            "Clique numa conversa para abrir.</div>")
                with gr.Column(scale=4):
                    ia_lista = gr.HTML("", elem_id="ia-lista")
            with gr.Row(visible=True) as linha_minha:
                with gr.Column(scale=3):
                    chat = gr.Chatbot(show_label=False, elem_id="chat-ia",
                                      placeholder="Pergunte qualquer coisa sobre os documentos da "
                                                  "equipe, do jeito que você falaria.")
                    with gr.Row():
                        pergunta = gr.Textbox(show_label=False, scale=5, container=False,
                                              placeholder="Ex.: em quantos dias recebo o reembolso?",
                                              autofocus=True)
                        btn_perguntar = gr.Button("Perguntar", variant="primary", scale=1,
                                                  min_width=120)
                with gr.Column(scale=2):
                    with gr.Row():
                        escopo = gr.Dropdown(label="Procurar em", choices=[("Todos os documentos", ALL_DOCS)],
                                             value=ALL_DOCS, scale=3)
                        btn_nova = gr.Button("Nova conversa", scale=1, min_width=120, size="sm")
                    gr.HTML("<div class='titulo-bloco'>De onde veio a resposta</div>")
                    fontes = gr.HTML("<div class='suave'>As fontes de cada resposta aparecem aqui.</div>",
                                     elem_id="fontes")

        # ── Documentos ───────────────────────────────────────────────────────
        with gr.Tab("Documentos") as tab_documentos:
            with gr.Row():
                with gr.Column(scale=1, min_width=260):
                    doc_lista = gr.Radio(label="Documentos da equipe", choices=[], value=None,
                                         elem_id="doc-lista")
                    with gr.Accordion("Enviar arquivo", open=False):
                        envio_arquivo = gr.File(label="Arquivo PDF", file_types=[".pdf"], height=140)
                        envio_destino = gr.Dropdown(label="Este arquivo é", value=NOVO_DOC,
                                                    choices=[("Um documento novo", NOVO_DOC)])
                        with gr.Column(visible=True) as envio_nome_box:
                            envio_nome = gr.Textbox(label="Nome do documento", max_lines=1,
                                                    placeholder="Ex.: Regulamento Interno",
                                                    info="Vazio: usa o nome do arquivo")
                        envio_motivo = gr.Textbox(label="O que mudou? (opcional)", lines=2,
                                                  max_lines=4, placeholder="Ex.: reembolso em 15 dias")
                        with gr.Accordion("Mais opções", open=False):
                            envio_apelidos = gr.Textbox(
                                label="Outros nomes usados para citar este documento", max_lines=1,
                                placeholder="Ex.: Regulamento, RI", info="Separados por vírgula")
                        btn_enviar = gr.Button("Enviar", variant="primary", elem_id="btn-enviar")
                        envio_status = gr.HTML("")

                with gr.Column(scale=3):
                    doc_cab = gr.HTML(_cabecalho_doc(None))
                    secao = gr.Radio(choices=SECOES, value="Documento", show_label=False,
                                     elem_id="secao", container=False)
                    with gr.Column(visible=True) as sec_texto:
                        doc_pdf = gr.HTML("")
                        doc_texto = gr.Markdown("", elem_id="doc-texto")
                    with gr.Column(visible=False) as sec_mud:
                        with gr.Column(elem_id="mud-rolagem"):
                            mud_alerta = gr.HTML("")
                            gr.HTML("<div class='titulo-bloco'>Resumo</div>")
                            mud_resumo = gr.Markdown("")
                            gr.HTML("<div class='titulo-bloco'>Outros documentos afetados</div>")
                            mud_impacto = gr.Markdown("")
                            with gr.Accordion("Ver o texto de antes e de depois", open=False):
                                mud_detalhe = gr.Markdown("")
                    with gr.Column(visible=False) as sec_hist:
                        doc_hist = gr.HTML("")
                        with gr.Row():
                            cmp_a = gr.Dropdown(label="Comparar a versão", choices=[], scale=2)
                            cmp_b = gr.Dropdown(label="com a versão", choices=[], scale=2)
                            btn_comparar = gr.Button("Comparar", scale=1, min_width=110,
                                                     elem_id="btn-comparar")
                    with gr.Column(visible=False) as sec_conv:
                        doc_conversa = gr.HTML("")
                        with gr.Row():
                            doc_msg = gr.Textbox(show_label=False, container=False, scale=5,
                                                 placeholder="Escreva para a equipe sobre este documento")
                            btn_doc_msg = gr.Button("Enviar", variant="primary", scale=1, min_width=110,
                                                    elem_id="btn-doc-msg")

        # ── Equipe ───────────────────────────────────────────────────────────
        with gr.Tab("Equipe") as tab_equipe:
            with gr.Row():
                with gr.Column(scale=3):
                    gr.HTML("<div class='titulo-bloco'>Conversa geral</div>")
                    geral_conversa = gr.HTML("")
                    with gr.Row():
                        geral_msg = gr.Textbox(show_label=False, container=False, scale=5,
                                               placeholder="Escreva para todo mundo")
                        btn_geral = gr.Button("Enviar", variant="primary", scale=1, min_width=110,
                                              elem_id="btn-geral")
                with gr.Column(scale=2):
                    gr.HTML("<div class='titulo-bloco'>Quem está aqui</div>")
                    quem_aqui = gr.HTML("")
                    gr.HTML("<div class='titulo-bloco'>Atividade recente</div>")
                    atividade = gr.HTML("", elem_id="atividade")

    # ── Eventos ──────────────────────────────────────────────────────────────


    # Perguntar
    for gatilho in (btn_perguntar.click, pergunta.submit):
        gatilho(perguntar, inputs=[pergunta, chat, llm_state, modelo, escopo, conversa_state],
                outputs=[pergunta, chat, fontes, llm_state, conversa_state], show_progress="minimal")
    btn_nova.click(nova_conversa, inputs=[llm_state],
                   outputs=[chat, fontes, llm_state, conversa_state], show_progress="hidden")
    ia_args = dict(fn=preencher_ia, inputs=[ver_ia, ia_filtro],
                   outputs=[ia_lista, ia_filtro, ia_visto_state], show_progress="hidden")
    ver_ia.input(mostrar_ver_ia, inputs=[ver_ia], outputs=[linha_minha, linha_equipe_ia],
                 show_progress="hidden", queue=False).then(**ia_args)
    ia_filtro.input(**ia_args)

    # Documentos
    painel = [doc_cab, doc_pdf, doc_texto, doc_hist, cmp_a, cmp_b, doc_conversa,
              mud_alerta, mud_resumo, mud_impacto, mud_detalhe]
    secoes = [sec_texto, sec_mud, sec_hist, sec_conv]
    preencher_args = dict(fn=preencher,
                          inputs=[secao, doc_lista, pendente_state, modelo, llm_state],
                          outputs=painel + [pendente_state, llm_state], show_progress="minimal")
    demo.load(ao_abrir, inputs=None,
              outputs=[topo, modelo, escopo, doc_lista, envio_destino, visto_versao, visto_msg,
                       conversa_state, ver_ia],
              show_progress="hidden").then(**preencher_args)
    doc_lista.input(**preencher_args)
    secao.input(mostrar_secao, inputs=[secao], outputs=secoes, show_progress="hidden",
                queue=False).then(**preencher_args)
    envio_destino.change(mudar_destino, inputs=[envio_destino], outputs=[envio_nome_box],
                         show_progress="hidden")
    btn_enviar.click(
        enviar_arquivo,
        inputs=[envio_arquivo, envio_destino, envio_nome, envio_motivo, envio_apelidos, modelo,
                llm_state, mine_state],
        outputs=[envio_status, envio_arquivo, envio_motivo, envio_nome, envio_destino, envio_nome_box,
                 doc_lista, llm_state, mine_state, secao, pendente_state] + secoes,
        show_progress="minimal", show_progress_on=[envio_status],
    ).then(**preencher_args)
    btn_comparar.click(comparar, inputs=[doc_lista, cmp_a, cmp_b, modelo, llm_state],
                       outputs=[pendente_state, llm_state, secao] + secoes,
                       show_progress="minimal", show_progress_on=[btn_comparar]
                       ).then(**preencher_args)
    for gatilho in (btn_doc_msg.click, doc_msg.submit):
        gatilho(escrever_doc, inputs=[doc_msg, doc_lista], outputs=[doc_msg, doc_conversa],
                show_progress="hidden")

    # Equipe
    for gatilho in (btn_geral.click, geral_msg.submit):
        gatilho(escrever_geral, inputs=[geral_msg], outputs=[geral_msg, geral_conversa, atividade],
                show_progress="hidden")

    # Abas: cada uma se atualiza ao ser aberta
    tab_perguntar.select(abrir_perguntar, inputs=[escopo], outputs=[aba, escopo],
                         show_progress="hidden")
    tab_documentos.select(abrir_documentos, inputs=[doc_lista, secao],
                          outputs=[aba, doc_lista, envio_destino] + painel, show_progress="hidden")
    tab_equipe.select(abrir_equipe, outputs=[aba, geral_conversa, quem_aqui, atividade, topo],
                      show_progress="hidden")

    # Tempo real
    relogio = gr.Timer(REFRESH_SECONDS)
    relogio.tick(
        a_cada_instante,
        inputs=[visto_versao, visto_msg, mine_state, aba, doc_lista, secao, escopo, topo_cache,
                online_cache, ver_ia, ia_filtro, ia_visto_state],
        outputs=[visto_versao, visto_msg, topo, topo_cache, escopo, doc_lista, envio_destino,
                 doc_cab, doc_pdf, doc_texto, doc_hist, cmp_a, cmp_b, doc_conversa, geral_conversa,
                 atividade, quem_aqui, online_cache, ia_lista, ia_filtro, ia_visto_state],
        show_progress="hidden",
        concurrency_limit=None,      # consulta rápida; não espera na fila dos envios
    )


def _enderecos(porta: int) -> List[str]:
    """Endereços desta máquina em cada placa de rede (Wi-Fi, VPN), para a equipe entrar."""
    achados: Dict[str, str] = {}
    try:
        import psutil                   # vem com as dependências; lista também a VPN
        for placa, ends in psutil.net_if_addrs().items():
            for e in ends:
                if e.family == socket.AF_INET:
                    achados.setdefault(e.address, placa)
    except Exception:
        try:
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
                achados.setdefault(info[4][0], "")
        except OSError:
            pass
    # 127.x é a própria máquina; 169.254.x é endereço automático, sem rede de verdade.
    return [f"http://{ip}:{porta}" + (f"  ({placa})" if placa else "")
            for ip, placa in sorted(achados.items())
            if not ip.startswith(("127.", "169.254."))]


if __name__ == "__main__":
    porta = int(os.environ.get("EIXOAI_PORTA", "7860"))
    publico = os.environ.get("EIXOAI_LINK_PUBLICO", "1") != "0"
    warm_up()
    print("\nEixoAI no ar.")
    print(f"  Neste computador: http://localhost:{porta}")
    for e in _enderecos(porta):
        print(f"  Para a equipe (mesma rede ou VPN): {e}")
    if CONTAS:
        print(f"  Login ativo para: {', '.join(c.nome for c in CONTAS)}")
    else:
        print("  Sem login: defina EIXOAI_USUARIOS no .env para cada pessoa entrar com o seu nome.")
    print()
    demo.launch(
        share=publico, server_name="0.0.0.0", server_port=porta, theme=THEME, css=CUSTOM_CSS,
        allowed_paths=[str(pasta_pdfs())],
        auth=[(c.login, c.senha) for c in CONTAS] or None,
        auth_message="Entre com o seu usuário da equipe.",
        inbrowser=os.environ.get("EIXOAI_ABRIR_NAVEGADOR", "1") != "0",
    )
