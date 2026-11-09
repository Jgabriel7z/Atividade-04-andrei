import gradio as gr
import html
import inspect
import logging
import re
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from ingester import extract_text_from_pdf
from llm_chain import LLMChain
from retriever import DocumentRetriever
from versioning import (
    VersionStore, VersionDiff, compare_versions, publish_version, sha256_file, unified_diff,
)
from impact import ImpactReport, analyze_impact, format_report, md_escape

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("../data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALL_DOCS = "__todos__"

# Recursos compartilhados entre sessões: o modelo de embeddings, o ChromaDB e o
# registro de versões. O que é de cada usuário (histórico do chat, documento
# escolhido) fica em gr.State, não em variável global.
_retriever: Optional[DocumentRetriever] = None
_store: Optional[VersionStore] = None
_retriever_lock = threading.Lock()


def get_retriever() -> DocumentRetriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:          # o warm_up e um upload podem chegar juntos
            if _retriever is None:
                _retriever = DocumentRetriever()
    return _retriever


def get_store() -> VersionStore:
    global _store
    if _store is None:
        _store = VersionStore()
    return _store


def warm_up() -> None:
    """
    Carrega o modelo de embeddings em segundo plano assim que o app sobe.
    Antes, o primeiro upload esperava o carregamento (9–15 s nesta máquina).
    """
    threading.Thread(target=get_retriever, name="warm-up", daemon=True).start()

# ─── Paleta: vermelho, preto e branco ─────────────────────────────────────────

BLACK = "#0a0a0a"       # fundo
PANEL = "#141414"       # blocos
BORDER = "#2e2e2e"
WHITE = "#fafafa"
MUTED = "#a3a3a3"
RED = "#dc2626"
RED_LIGHT = "#f87171"
RED_BG = "#1f0707"
MONO = "font-family: 'JetBrains Mono', monospace; font-size: 13px;"


def _box(border: str, bg: str, body: str, dashed: bool = False, width: str = "1.5px") -> str:
    style = "dashed" if dashed else "solid"
    return (f"<div style='display:flex;flex-direction:column;gap:6px;padding:14px 18px;"
            f"border-radius:10px;background:{bg};border:{width} {style} {border};"
            f"color:{WHITE};{MONO}'>{body}</div>")


def _dot(color: str, glow: bool = False) -> str:
    shadow = f"box-shadow:0 0 6px {color}aa;" if glow else ""
    return (f"<span style='display:inline-block;width:10px;height:10px;border-radius:50%;"
            f"background:{color};{shadow}margin-right:10px;flex-shrink:0'></span>")

# ─── HTML dos estados de status ────────────────────────────────────────────────

def _status_idle() -> str:
    return _box(BORDER, PANEL,
                f"<div style='display:flex;align-items:center'>{_dot('#525252')}"
                f"<span style='color:{MUTED}'>Nenhum PDF carregado — faça upload para começar</span></div>",
                dashed=True)


def _status_loading() -> str:
    return f"""
    <style>
    @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
    @keyframes pulse-text {{ 0%,100% {{ opacity: 1; }} 50% {{ opacity: 0.55; }} }}
    </style>
    <div style="display:flex;align-items:center;gap:12px;padding:14px 18px;border-radius:10px;
                background:{PANEL};border:1.5px solid {RED};{MONO}">
        <div style="width:16px;height:16px;border-radius:50%;border:2.5px solid {RED}44;
                    border-top-color:{RED};animation:spin 0.8s linear infinite;flex-shrink:0"></div>
        <span style="color:{WHITE};animation:pulse-text 1.4s ease-in-out infinite;">
            Processando PDF e gerando embeddings…
        </span>
    </div>
    """


def _status_ready(filename: str, chunks: int, chars: int, extra: str = "") -> str:
    return _box(WHITE, PANEL,
                f"<div style='display:flex;align-items:center'>{_dot(RED, glow=True)}"
                f"<span style='font-weight:600'>PDF pronto — pode enviar mensagens</span></div>"
                f"<div style='display:flex;gap:16px;padding-left:20px;color:{MUTED};font-size:11px'>"
                f"<span>📄 {html.escape(filename)}</span><span>·</span>"
                f"<span>🧩 {chunks} chunks</span><span>·</span>"
                f"<span>📝 {chars:,} caracteres</span></div>{extra}")


def _status_error(msg: str) -> str:
    return _box(RED, RED_BG,
                f"<div style='display:flex;align-items:center'>{_dot(RED)}"
                f"<span>{html.escape(msg)}</span></div>")


def _status_base(n_docs: int, n_chunks: int) -> str:
    return _box(WHITE, PANEL,
                f"<div style='display:flex;align-items:center'>{_dot(RED, glow=True)}"
                f"<span>📚 <b>{n_docs}</b> documento(s) na base · 🧩 {n_chunks} chunks vigentes — "
                f"pode enviar mensagens</span></div>")


def _note(text: str) -> str:
    return f"<div style='padding-left:20px;color:{MUTED};font-size:11px'>{text}</div>"


def _alert(diff: Optional[VersionDiff], impact: Optional[ImpactReport], name: str,
           old_v: Optional[int], new_v: int) -> str:
    """Faixa de aviso que aparece quando um documento muda."""
    name = html.escape(name)
    if old_v is None or diff is None:
        return _box(WHITE, PANEL, f"<div>📄 <b>{name}</b> entrou na base (versão {new_v}).</div>")
    n_chg = len([c for c in diff.changed if c.key != "preambulo"])
    parts = []
    if not diff.structured:
        parts.append("texto alterado (documento sem artigos)")
    elif n_chg:
        parts.append(f"{n_chg} artigo(s) alterado(s)")
    if diff.removed:
        parts.append(f"{len(diff.removed)} removido(s)")
    if diff.added:
        parts.append(f"{len(diff.added)} incluído(s)")
    what = ", ".join(parts) or "só mudanças fora dos artigos"
    n_direct = len(impact.direct) if impact else 0
    n_sug = len(impact.suggestions) if impact else 0
    docs = sorted({c.document_name for c in impact.direct}) if impact else []
    head = f"<b>{name}</b> mudou (v{old_v} → v{new_v}): {what}."
    if n_direct or n_sug:
        body = (f"<div style='font-size:14px'>⚠️ {head}</div>"
                f"<div><b style='color:{RED_LIGHT}'>{n_direct}</b> trecho(s) em {len(docs)} outro(s) "
                f"documento(s) citam esses artigos ({html.escape(', '.join(docs)) or '—'}); "
                f"<b style='color:{RED_LIGHT}'>{n_sug}</b> sugestão(ões) por semelhança. "
                f"Veja a lista abaixo.</div>")
        return _box(RED, RED_BG, body, width="2px")
    return _box(WHITE, PANEL, f"<div>✅ {head}</div>"
                f"<div style='color:{MUTED}'>Nenhum outro documento da base depende dos artigos alterados.</div>")


# Tema: preto de fundo, texto branco, vermelho nos destaques. Os valores
# "_dark" repetem os claros para a página ficar igual em qualquer navegador.
def _theme() -> gr.themes.Base:
    t = gr.themes.Base(
        primary_hue=gr.themes.colors.red,
        secondary_hue=gr.themes.colors.red,
        neutral_hue=gr.themes.colors.neutral,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    )
    values = dict(
        body_background_fill=BLACK, body_text_color=WHITE, body_text_color_subdued=MUTED,
        background_fill_primary=BLACK, background_fill_secondary=PANEL,
        block_background_fill=PANEL, block_border_color=BORDER, block_border_width="1px",
        block_label_background_fill=RED, block_label_text_color=WHITE,
        block_title_text_color=WHITE, block_info_text_color=MUTED,
        border_color_primary=BORDER, border_color_accent=RED,
        color_accent=RED, color_accent_soft=RED_BG,
        input_background_fill="#1c1c1c", input_border_color=BORDER,
        input_border_color_focus=RED, input_placeholder_color="#737373",
        button_primary_background_fill=RED, button_primary_background_fill_hover="#b91c1c",
        button_primary_text_color=WHITE, button_primary_border_color=RED,
        button_secondary_background_fill="#1c1c1c", button_secondary_background_fill_hover="#262626",
        button_secondary_text_color=WHITE, button_secondary_border_color=BORDER,
        link_text_color=RED_LIGHT, link_text_color_hover=WHITE,
        table_even_background_fill=PANEL, table_odd_background_fill="#101010",
        table_border_color=BORDER, checkbox_label_background_fill=PANEL,
        panel_background_fill=PANEL, panel_border_color=BORDER,
    )
    accepted = inspect.signature(t.set).parameters
    t.set(**values, **{f"{k}_dark": v for k, v in values.items() if f"{k}_dark" in accepted})
    return t


THEME = _theme()

# ─── Versões: registro, comparação, resumo e impacto ──────────────────────────

def _doc_choices() -> List[Tuple[str, str]]:
    return [(d.name, d.id) for d in get_store().list_documents()]


def _scope_choices() -> List[Tuple[str, str]]:
    return [("Todos os documentos", ALL_DOCS)] + _doc_choices()


_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")


def _fence(text: str, lang: str = "") -> str:
    """Bloco de código que o próprio texto não consegue fechar (crases vindas do PDF)."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    f = "`" * max(3, longest + 1)
    return f"{f}{lang}\n{text}\n{f}"


def _summarize(diff: VersionDiff, name: str, llm: Optional[LLMChain]) -> Tuple[str, bool]:
    """
    Resumo pela LLM só com os artigos que mudaram; sem LLM, o resumo do diff.
    Devolve (texto, veio_da_llm): só o resumo da LLM é gravado no histórico,
    para não ficar preso ao resumo reserva depois que a chave for configurada.
    """
    if not diff.has_changes:
        return "Nenhuma diferença de conteúdo entre as versões.", False
    before = "\n\n".join([c.before for c in diff.changed] + [a.text for a in diff.removed])
    after = "\n\n".join([c.after for c in diff.changed] + [a.text for a in diff.added])
    try:
        chain = llm or LLMChain(stream=False)
        text = chain.summarize_change(before, after, name, diff.as_text())
        # Um PDF pode tentar induzir a LLM a pôr imagem externa no resumo.
        return _MD_IMAGE.sub("", text), True
    except Exception as e:
        logger.warning(f"Resumo pela LLM falhou, usando o diff: {e}")
        return ("_Resumo automático (sem LLM — confira a GROQ_API_KEY):_\n\n"
                + "\n".join(f"- {md_escape(line)}" for line in diff.as_text().splitlines()),
                False)


def _diff_markdown(diff: Optional[VersionDiff]) -> str:
    if diff is None:
        return "_Primeira versão: não há com o que comparar._"
    if not diff.has_changes:
        return "_As duas versões têm o mesmo conteúdo._"
    out = [f"**{len(diff.changed)} alterado(s) · {len(diff.removed)} removido(s) · "
           f"{len(diff.added)} incluído(s) · {diff.unchanged} sem mudança**", ""]
    for c in diff.changed:
        out.append(f"#### ✏️ {c.label} — alterado (semelhança {c.similarity:.0%})")
        out.append(md_escape(c.edits_text()))
        out.append(_fence(unified_diff(c.before, c.after, c.label), "diff"))
    for a in diff.removed:
        out.append(f"#### ➖ {a.label} — removido")
        out.append("> " + md_escape(" ".join(a.text.split())))
    for a in diff.added:
        out.append(f"#### ➕ {a.label} — incluído")
        out.append("> " + md_escape(" ".join(a.text.split())))
    return "\n\n".join(out)


def _history_rows(doc_id: Optional[str]) -> List[List]:
    if not doc_id:
        return []
    return [
        [v.version, v.created_at.replace("T", " "), v.author or "—", v.reason or "—",
         v.filename, v.sha256[:12], v.n_chunks, "✅" if v.is_current else ""]
        for v in get_store().list_versions(doc_id)
    ]


def _version_choices(doc_id: Optional[str]) -> List[Tuple[str, int]]:
    if not doc_id:
        return []
    return [(f"v{v.version} — {v.created_at[:10]} — {v.author or 'sem autor'}", v.version)
            for v in get_store().list_versions(doc_id)]


def _register(file, name: str, author: str, reason: str, aliases: str,
              llm: Optional[LLMChain]):
    """Registra o PDF como versão nova e calcula diff, resumo e impacto."""
    file_path = file.get("name") if isinstance(file, dict) else file
    text = extract_text_from_pdf(str(file_path))
    if not text:
        raise ValueError("PDF sem texto — pode ser um arquivo escaneado ou corrompido.")
    filename = Path(file_path).name
    store = get_store()
    # O nome digitado pode ser o id de um documento existente (valor do dropdown).
    existing = store.get_document(name) if name else None
    display = existing.name if existing else (name or Path(filename).stem)
    alias_list = [a.strip() for a in (aliases or "").split(",") if a.strip()]

    res = publish_version(
        store, get_retriever(), name=display, text=text, filename=filename,
        sha256=sha256_file(file_path), author=author or "", reason=reason or "",
        aliases=[display] + alias_list, document_id=existing.id if existing else None,
    )
    impact = summary = None
    if res.status == "created" and res.diff is not None:
        impact = analyze_impact(store, get_retriever(), res.document.id, res.diff)
        summary, from_llm = _summarize(res.diff, res.document.name, llm)
        if from_llm:
            store.set_summary(res.document.id, res.version.version, summary)
    return res, impact, summary, len(text)


def upload_version(file, name, author, reason, aliases, llm):
    """Aba Versões: enviar nova versão com autor e motivo."""
    if file is None:
        return (_status_error("Selecione um PDF."), "", "", "", gr.update(), gr.update(), llm)
    try:
        res, impact, summary, _ = _register(file, name, author, reason, aliases, llm)
    except Exception as e:
        logger.error(f"Erro ao registrar versão: {e}")
        return (_status_error(f"Erro: {e}"), "", "", "", gr.update(), gr.update(), llm)

    doc = res.document
    if res.status == "unchanged":
        alert = _status_error(f"Arquivo idêntico à versão vigente de {doc.name} "
                              f"(v{res.version.version}, mesmo SHA-256). Nenhuma versão criada.")
        summary_md = diff_md = impact_md = ""
    else:
        old_v = res.previous.version if res.previous else None
        alert = _alert(res.diff, impact, doc.name, old_v, res.version.version)
        summary_md = summary or "_Primeira versão._"
        diff_md = _diff_markdown(res.diff)
        impact_md = format_report(impact) if impact else "_Primeira versão: nada a verificar._"
    choices = _doc_choices()
    return (alert, summary_md, diff_md, impact_md,
            gr.update(choices=choices, value=doc.id),
            gr.update(choices=choices, value=doc.id), llm)


def show_history(doc_id):
    vs = _version_choices(doc_id)
    a = vs[-2][1] if len(vs) > 1 else (vs[0][1] if vs else None)
    b = vs[-1][1] if vs else None
    return (_history_rows(doc_id), gr.update(choices=vs, value=a), gr.update(choices=vs, value=b))


def compare_selected(doc_id, va, vb, llm):
    if not doc_id or va is None or vb is None:
        return "_Escolha o documento e as duas versões._", "", "", llm
    store = get_store()
    a, b = store.get_version(doc_id, int(va)), store.get_version(doc_id, int(vb))
    if a is None or b is None:
        return "_Versão não encontrada._", "", "", llm
    diff = compare_versions(a.text, b.text)
    if b.change_summary and a.version == b.version - 1:
        summary = b.change_summary                        # já gerado no envio
    else:
        summary, from_llm = _summarize(diff, store.get_document(doc_id).name, llm)
        if from_llm and a.version == b.version - 1:
            store.set_summary(doc_id, b.version, summary)
    impact = analyze_impact(store, get_retriever(), doc_id, diff)
    return summary, _diff_markdown(diff), format_report(impact), llm

# ─── Chat ─────────────────────────────────────────────────────────────────────

def start_processing(file):
    """
    Chamado imediatamente ao clicar em 'Processar PDF'.
    Retorna o estado de carregamento e bloqueia o chat.
    """
    if file is None:
        return (
            _status_error("Selecione um arquivo PDF antes de processar."),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=True),   # clear_btn permanece ativo
        )
    return (
        _status_loading(),
        gr.update(interactive=False),  # bloqueia input de texto
        gr.update(interactive=False),  # bloqueia botão enviar
        gr.update(interactive=False),  # bloqueia limpar conversa
    )


def process_pdf(file, llm):
    """
    Processa o PDF e o registra na base como versão nova do documento com o
    mesmo nome de arquivo. Antes, o upload chamava retriever.clear_all() e
    apagava a base inteira; agora os documentos anteriores continuam lá.
    """
    has_docs = get_store().has_documents()
    if file is None:
        return (_status_error("Nenhum arquivo selecionado."), gr.update(interactive=has_docs),
                gr.update(interactive=has_docs), gr.update(interactive=True), gr.update(), llm)
    try:
        res, impact, _, n_chars = _register(file, "", "", "enviado pelo chat", "", llm)
        doc = res.document
        if res.status == "unchanged":
            extra = _note(f"Já era a versão vigente de {html.escape(doc.name)} "
                          f"(v{res.version.version}).")
        elif res.previous is not None:
            extra = _alert(res.diff, impact, doc.name, res.previous.version, res.version.version)
        else:
            n_docs, _ = get_store().current_totals()
            extra = _note(f"📚 {html.escape(doc.name)} — versão {res.version.version} · "
                          f"{n_docs} documento(s) na base")
        logger.info(f"PDF processado: {res.version.filename} ({res.version.n_chunks} chunks)")
        return (
            _status_ready(res.version.filename, res.version.n_chunks, n_chars, extra),
            gr.update(interactive=True),   # libera input
            gr.update(interactive=True),   # libera botão enviar
            gr.update(interactive=True),   # libera limpar conversa
            gr.update(choices=_scope_choices(), value=doc.id),
            llm,
        )
    except Exception as e:
        logger.error(f"Erro ao processar PDF: {str(e)}")
        return (_status_error(f"Erro: {str(e)}"), gr.update(interactive=has_docs),
                gr.update(interactive=has_docs), gr.update(interactive=True), gr.update(), llm)


def _expand_query(query: str, llm: "LLMChain") -> str:
    """
    Usa a LLM para reescrever a query adicionando termos jurídicos relacionados.
    Combina a query original com a versão expandida para maximizar o recall.
    Retorna a query original em caso de erro (nunca bloqueia o fluxo).
    """
    try:
        prompt = (
            "Reescreva a pergunta abaixo adicionando sinônimos e termos jurídicos "
            "da Constituição Federal brasileira que possam aparecer nos trechos relevantes. "
            "Responda APENAS com a pergunta reescrita, sem explicações.\n\n"
            f"Pergunta: {query}"
        )
        expanded = llm.generate_response(prompt)
        # Remove a pergunta do histórico — é uma chamada interna
        if llm.conversation_history:
            llm.conversation_history.pop()  # assistant
            llm.conversation_history.pop()  # user
        combined = f"{query} {expanded}"
        logger.info(f"🔎 Query expandida: {combined[:120]}…")
        return combined
    except Exception as e:
        logger.warning(f"Query expansion falhou, usando original: {e}")
        return query


def send_message(user_message: str, history: List[Dict], llm: Optional[LLMChain], scope: str):
    if not user_message.strip():
        return "", history, "", llm

    chunk_info = ""
    doc_filter = None if (not scope or scope == ALL_DOCS) else scope

    # Um LLMChain por sessão: o histórico não vaza entre usuários. Sem chave do
    # Groq, a busca roda mesmo assim e os trechos aparecem no rastreamento.
    llm_error = None
    if llm is None:
        try:
            llm = LLMChain(stream=False)
        except Exception as e:
            llm_error = e

    try:
        context = ""
        if get_store().has_documents():
            retriever = get_retriever()
            try:
                # Expande a query com termos jurídicos antes de buscar
                search_query = _expand_query(user_message, llm) if llm else user_message
                # Só a versão vigente de cada documento (padrão do search)
                results = retriever.search(search_query, top_k=5, rerank=True,
                                           document_id=doc_filter)
                if results:
                    chunk_parts = ["🔍 **CHUNKS RECUPERADOS — RASTREAMENTO COMPLETO**\n"]
                    chunk_parts.append("=" * 100)
                    for i, result in enumerate(results, 1):
                        chunk_parts.append(f"\n### **[{i}] CHUNK #{result['chunk']}**")
                        chunk_parts.append(f"📄 **Arquivo:** {md_escape(str(result['source']))}")
                        chunk_parts.append(f"🗂️ **Documento:** `{result.get('document_id')}` "
                                           f"(trecho vigente desde a v{result.get('version', '—')})")
                        chunk_parts.append(f"📍 **Página:** {result['page']}")
                        chunk_parts.append(f"🎯 **Relevância:** {result['relevance']:.1%}")
                        chunk_parts.append(f"🆔 **ID:** `{result['id']}`")
                        chunk_parts.append(f"📊 **Distância:** {result['distance']:.4f}")
                        chunk_parts.append(f"\n**CONTEÚDO COMPLETO:**")
                        chunk_parts.append(_fence(result["text"]))
                        chunk_parts.append("-" * 100)
                    chunk_info = "\n".join(chunk_parts)
                    # Antes: get_context() fazia a mesma busca de novo (2º embedding + 2ª query).
                    context = retriever.format_context(results)
                else:
                    chunk_info = "⚠️ Nenhum chunk relevante encontrado para esta pergunta."
            except Exception as e:
                logger.warning(f"Erro ao recuperar contexto: {str(e)}")
                chunk_info = f"❌ Erro ao recuperar chunks: {str(e)}"
        else:
            chunk_info = "📭 Nenhum PDF carregado."

        system_prompt = (
            "Você é um assistente útil que responde perguntas sobre documentos. "
            "Use o contexto fornecido para responder com precisão. "
            "Se a informação não estiver no contexto, diga claramente."
        )

        # Limite de segurança: nunca enviar mais do que ~12 000 caracteres de contexto
        # (~3 000 tokens), deixando espaço para o histórico e a resposta.
        MAX_CONTEXT_CHARS = 12_000
        safe_context = context[:MAX_CONTEXT_CHARS] if context else ""

        if llm is None:
            response = (f"⚠️ A LLM não está disponível ({llm_error}). Configure a GROQ_API_KEY "
                        f"no .env. Os trechos encontrados estão no rastreamento abaixo.")
        else:
            response = llm.generate_response(
                user_message=user_message,
                system_prompt=system_prompt,
                context=safe_context,
            )

    except Exception as e:
        response = f"❌ Erro ao processar mensagem:\n{str(e)}"
        # Se a busca deu certo e só a LLM falhou, os chunks continuam visíveis.
        if not chunk_info.startswith("🔍"):
            chunk_info = f"❌ Erro: {str(e)}"
        logger.error(f"Erro ao chamar LLM: {str(e)}")

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})
    return "", history, chunk_info, llm


def clear_chat(llm):
    """
    Limpa só a conversa desta sessão. Antes, 'Limpar tudo' apagava o documento
    do banco; com histórico de versões, apagar documento deixou de ser papel do chat.
    """
    if llm is not None:
        llm.clear_history()
    return [], "Nenhum chunk recuperado ainda.", llm


def on_load():
    """Ao abrir a página: habilita o chat se a base já tem documentos."""
    n_docs, n_chunks = get_store().current_totals()
    has = n_docs > 0
    status = _status_base(n_docs, n_chunks) if has else _status_idle()
    return (status, gr.update(interactive=has), gr.update(interactive=has),
            gr.update(choices=_scope_choices(), value=ALL_DOCS))


# No Gradio 6.3, componente de aba oculta que recebe atualização fica preso com
# o "carregando" por cima quando a aba aparece. Por isso cada aba se atualiza
# ao ser aberta, em vez de receber atualização da outra aba.

def on_chat_tab(scope_value):
    choices = _scope_choices()
    has = len(choices) > 1
    keep = scope_value if scope_value in [v for _, v in choices] else ALL_DOCS
    return (gr.update(choices=choices, value=keep),
            gr.update(interactive=has), gr.update(interactive=has))


def on_versions_tab(h_doc_value):
    docs = _doc_choices()
    ids = [d for _, d in docs]
    sel = h_doc_value if h_doc_value in ids else (ids[0] if ids else None)
    return (gr.update(choices=docs), gr.update(choices=docs, value=sel), *show_history(sel))

# ─── CSS customizado ───────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Chat bloqueado — feedback visual claro */
.chat-blocked textarea {
    opacity: 0.45 !important;
    cursor: not-allowed !important;
}
.chat-blocked button {
    opacity: 0.45 !important;
    cursor: not-allowed !important;
}

/* Remove borda padrão do Gradio no status */
#pdf-status, #version-alert {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

/* Área de chunk com fonte mono */
#chunk-tracker .prose {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
}

/* Botão primário */
.primary-btn {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.03em !important;
}

/* Paleta vermelho / preto / branco */
.gradio-container { background: #0a0a0a !important; }
.prose h1, .prose h2, .prose h3, .prose h4 { color: #fafafa !important; }
.prose h1 { border-bottom: 3px solid #dc2626; padding-bottom: 6px; display: inline-block; }
.prose strong { color: #fafafa; }
.prose code { background: #1c1c1c !important; color: #f87171 !important; }
.prose blockquote { border-left: 3px solid #dc2626 !important; color: #d4d4d4; }
button[role="tab"][aria-selected="true"] { color: #fafafa !important; border-bottom: 2px solid #dc2626 !important; }
button[role="tab"] { color: #a3a3a3 !important; }
"""

# ─── Interface Gradio ──────────────────────────────────────────────────────────

with gr.Blocks(title="EixoAI") as demo:

    llm_state = gr.State(None)     # LLMChain desta sessão (histórico do chat)

    gr.Markdown("# 📚 EixoAI — Chat com PDFs")
    gr.Markdown("Busca semântica com rastreamento de chunks e histórico de versões dos documentos")

    with gr.Tabs():
        # ══ Aba 1: Chat ══════════════════════════════════════════════════════
        with gr.Tab("💬 Chat") as chat_tab:
            with gr.Row():

                # ── Coluna esquerda: upload + status ──────────────────
                with gr.Column(scale=1):
                    gr.Markdown("### 📄 Documento")

                    pdf_file = gr.File(
                        label="Selecione um PDF",
                        file_types=[".pdf"],
                    )

                    upload_btn = gr.Button(
                        "⚙️  Processar PDF",
                        variant="primary",
                        elem_classes=["primary-btn"],
                    )

                    # Painel de status — muda conforme o estado
                    pdf_status = gr.HTML(
                        value=_status_idle(),
                        elem_id="pdf-status",
                    )

                    scope = gr.Dropdown(
                        label="Buscar em",
                        choices=[("Todos os documentos", ALL_DOCS)],
                        value=ALL_DOCS,
                        info="Sempre na versão vigente de cada documento",
                    )

                    gr.Markdown("---")
                    clear_btn = gr.Button("🧹  Limpar conversa", variant="secondary")

                # ── Coluna direita: chat ───────────────────────────────
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 Chat")

                    chatbot = gr.Chatbot(
                        label="Histórico",
                        height=420,
                    )

                    with gr.Row():
                        msg_input = gr.Textbox(
                            label="Mensagem",
                            placeholder="Pergunte sobre os documentos da base (carregue um PDF se ela estiver vazia)…",
                            scale=4,
                            interactive=False,   # bloqueado até haver documento na base
                            elem_id="msg-input",
                        )
                        send_btn = gr.Button(
                            "Enviar",
                            scale=1,
                            interactive=False,
                            variant="primary",
                        )

                    gr.Markdown("---")
                    gr.Markdown("### 🔍 Rastreamento de Chunks")
                    chunk_tracker = gr.Markdown(
                        "Nenhum chunk recuperado ainda.",
                        elem_id="chunk-tracker",
                    )

        # ══ Aba 2: Versões ═══════════════════════════════════════════════════
        with gr.Tab("🗂️ Versões") as versions_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Enviar nova versão")
                    v_file = gr.File(label="PDF da nova versão", file_types=[".pdf"])
                    v_doc = gr.Dropdown(
                        label="Documento",
                        info="Escolha um existente ou digite o nome de um novo",
                        choices=[], value=None, allow_custom_value=True,
                    )
                    v_aliases = gr.Textbox(
                        label="Como outros documentos o citam (opcional)",
                        placeholder="ex.: Regulamento, RI",
                        info="Separado por vírgula. O nome do documento já entra.",
                    )
                    v_author = gr.Textbox(label="Autor", placeholder="Quem está enviando")
                    v_reason = gr.Textbox(label="Motivo da mudança", lines=2,
                                          placeholder="ex.: redução do prazo de reembolso")
                    v_btn = gr.Button("📤  Registrar versão", variant="primary",
                                      elem_classes=["primary-btn"])

                with gr.Column(scale=2):
                    v_alert = gr.HTML("", elem_id="version-alert")
                    gr.Markdown("### 📝 Resumo da mudança")
                    v_summary = gr.Markdown("_Envie uma versão ou compare duas abaixo._")
                    gr.Markdown("### 🔗 Impacto em outros documentos")
                    v_impact = gr.Markdown("")
                    with gr.Accordion("🔀 Comparação por artigo", open=False):
                        v_diff = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("### 🕓 Histórico")
            with gr.Row():
                h_doc = gr.Dropdown(label="Documento", choices=[], value=None, scale=2)
                h_a = gr.Dropdown(label="Versão A (antes)", choices=[], scale=1)
                h_b = gr.Dropdown(label="Versão B (depois)", choices=[], scale=1)
                h_btn = gr.Button("🔀  Comparar", scale=1)
            h_table = gr.Dataframe(
                headers=["Versão", "Data", "Autor", "Motivo", "Arquivo", "SHA-256", "Chunks", "Vigente"],
                value=[], interactive=False, wrap=True,
            )

    # ── Event handlers ─────────────────────────────────────────────────────────

    # 1. Clicar em "Processar": imediatamente mostra spinner e bloqueia chat
    upload_btn.click(
        fn=start_processing,
        inputs=[pdf_file],
        outputs=[pdf_status, msg_input, send_btn, clear_btn],
        queue=False,          # resposta imediata, sem fila
    ).then(
        # 2. Em seguida, registra a versão e atualiza o status
        fn=process_pdf,
        inputs=[pdf_file, llm_state],
        outputs=[pdf_status, msg_input, send_btn, clear_btn, scope, llm_state],
    )

    # Enviar mensagem
    send_btn.click(
        fn=send_message,
        inputs=[msg_input, chatbot, llm_state, scope],
        outputs=[msg_input, chatbot, chunk_tracker, llm_state],
    )

    msg_input.submit(
        fn=send_message,
        inputs=[msg_input, chatbot, llm_state, scope],
        outputs=[msg_input, chatbot, chunk_tracker, llm_state],
    )

    # Limpar conversa (não apaga documentos)
    clear_btn.click(
        fn=clear_chat,
        inputs=[llm_state],
        outputs=[chatbot, chunk_tracker, llm_state],
    )

    # Versões
    v_btn.click(
        fn=upload_version,
        inputs=[v_file, v_doc, v_author, v_reason, v_aliases, llm_state],
        outputs=[v_alert, v_summary, v_diff, v_impact, v_doc, h_doc, llm_state],
    ).then(fn=show_history, inputs=[h_doc], outputs=[h_table, h_a, h_b])

    chat_tab.select(fn=on_chat_tab, inputs=[scope], outputs=[scope, msg_input, send_btn])
    versions_tab.select(fn=on_versions_tab, inputs=[h_doc],
                        outputs=[v_doc, h_doc, h_table, h_a, h_b])

    h_doc.change(fn=show_history, inputs=[h_doc], outputs=[h_table, h_a, h_b])
    h_btn.click(
        fn=compare_selected,
        inputs=[h_doc, h_a, h_b, llm_state],
        outputs=[v_summary, v_diff, v_impact, llm_state],
    )

    demo.load(fn=on_load, outputs=[pdf_status, msg_input, send_btn, scope])


if __name__ == "__main__":
    warm_up()
    demo.launch(share=True, server_name="0.0.0.0", server_port=7860, theme=THEME, css=CUSTOM_CSS)
