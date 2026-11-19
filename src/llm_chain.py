import logging
import os
from typing import List, Dict, Optional, Generator
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

# Limite conservador: modelos Groq têm context window de 8 192 tokens para alguns
# e 32 768 para outros. Usamos 24 000 caracteres (~6 000 tokens) como teto total
# para histórico + contexto + pergunta, deixando ~1 024 tokens para a resposta.
MAX_HISTORY_TURNS = 10          # pares (user + assistant) mantidos no histórico
MAX_CONTEXT_CHARS = 12_000      # caracteres máximos do bloco de contexto RAG
MAX_TOTAL_MSG_CHARS = 24_000    # teto total de todos os conteúdos somados

# O llama-3.3-70b-versatile, modelo original do projeto, saiu do Groq.
# GROQ_MODEL no .env troca o modelo sem mexer no código.
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


class LLMChain:
    """Gerenciador de cadeia de LLM com Groq."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        top_p: float = 1.0,
        stream: bool = True,
    ):
        self.client = Groq()
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stream = stream
        self.conversation_history: List[Dict[str, str]] = []
        logger.info(f"LLMChain inicializado com modelo: {model}")

    # ── Histórico ────────────────────────────────────────────

    def add_message(self, role: str, content: str) -> None:
        if not content.strip():
            raise ValueError("Conteúdo da mensagem não pode estar vazio")
        self.conversation_history.append({"role": role, "content": content})

    def clear_history(self) -> None:
        self.conversation_history.clear()
        logger.info("Histórico de conversa limpo")

    def _trimmed_history(self) -> List[Dict[str, str]]:
        """
        Retorna os últimos MAX_HISTORY_TURNS pares do histórico.
        Garante que o histórico retornado sempre começa com uma mensagem
        de 'user' (nunca de 'assistant' isolada).
        """
        # Cada par = 2 mensagens (user + assistant)
        max_msgs = MAX_HISTORY_TURNS * 2
        trimmed = self.conversation_history[-max_msgs:]

        # Se começar com assistant, descarta a primeira entrada
        if trimmed and trimmed[0]["role"] == "assistant":
            trimmed = trimmed[1:]

        return trimmed

    # ── Preparação de mensagens ──────────────────────────────

    def _prepare_messages(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Histórico aparado
        messages.extend(self._trimmed_history())

        # Contexto truncado + pergunta
        safe_context = (context or "")[:MAX_CONTEXT_CHARS]
        content = (
            f"Contexto:\n{safe_context}\n\nPergunta:\n{user_message}"
            if safe_context
            else user_message
        )

        # Verificação de segurança: se o total ainda for grande demais,
        # descarta histórico mais antigo até caber
        while len(messages) > 1:  # preserva ao menos o system prompt
            total_chars = sum(len(m["content"]) for m in messages) + len(content)
            if total_chars <= MAX_TOTAL_MSG_CHARS:
                break
            # Remove o par mais antigo do histórico (user+assistant após system)
            start = 1 if messages and messages[0]["role"] == "system" else 0
            if start < len(messages) - 1:
                messages.pop(start + 1)  # assistant
                messages.pop(start)      # user
            else:
                break

        messages.append({"role": "user", "content": content})
        self.add_message("user", user_message)
        return messages

    # ── Geração ──────────────────────────────────────────────

    def generate_response(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
    ) -> str:
        try:
            messages = self._prepare_messages(user_message, system_prompt, context)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                top_p=self.top_p,
                stream=False,
            )

            assistant_message = response.choices[0].message.content
            self.add_message("assistant", assistant_message)
            logger.info(f"Resposta gerada ({len(assistant_message)} chars)")
            return assistant_message

        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {str(e)}")
            raise

    def generate_response_stream(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Generator[str, None, None]:
        try:
            messages = self._prepare_messages(user_message, system_prompt, context)
            full_response = ""

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_tokens,
                top_p=self.top_p,
                stream=True,
            )

            for chunk in completion:
                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    full_response += delta_content
                    yield delta_content

            self.add_message("assistant", full_response)
            logger.info(f"Streaming concluído ({len(full_response)} chars)")

        except Exception as e:
            logger.error(f"Erro ao fazer streaming: {str(e)}")
            raise

    # ── Resumo de mudança entre versões ──────────────────────

    SUMMARY_SYSTEM_PROMPT = (
        "Você explica mudanças em documentos normativos (regulamentos, contratos, "
        "manuais) para quem não é da área jurídica. Escreva em português do Brasil, "
        "em frases curtas, uma por mudança, citando o artigo e os valores antes e "
        "depois (ex.: 'O prazo do art. 12 caiu de 30 para 15 dias.'). "
        "Fale só do que mudou. Não invente nada que não esteja nos textos."
    )
    MAX_SUMMARY_INPUT_CHARS = 10_000

    def summarize_change(
        self,
        before: str,
        after: str,
        document_name: str = "documento",
        changes_hint: str = "",
    ) -> str:
        """
        Resume em linguagem simples o que mudou entre `before` e `after`.

        Chamada avulsa: monta as mensagens à parte e NÃO usa nem altera
        conversation_history, para o resumo não aparecer no chat.
        `changes_hint` é o diff por artigo já calculado (versioning.VersionDiff.as_text),
        que ajuda a LLM a não deixar mudança de fora.
        """
        half = self.MAX_SUMMARY_INPUT_CHARS // 2
        user = (
            f"Documento: {document_name}\n\n"
            + (f"Diferenças detectadas automaticamente:\n{changes_hint[:2000]}\n\n" if changes_hint else "")
            + f"TEXTO ANTES:\n{before[:half]}\n\nTEXTO DEPOIS:\n{after[:half]}\n\n"
            "Resuma o que mudou."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            max_completion_tokens=self.max_tokens,
            top_p=self.top_p,
            stream=False,
        )
        summary = response.choices[0].message.content.strip()
        logger.info(f"Resumo de mudança gerado ({len(summary)} chars)")
        return summary

    # ── Verificação de dependência (análise de impacto) ──────

    DEPENDENCY_SYSTEM_PROMPT = (
        "Você confere avisos de impacto entre documentos. Um artigo de um documento "
        "mudou (ou foi retirado) e um trecho de OUTRO documento foi apontado como "
        "possível dependente. O trecho depende do artigo quando repete, aplica, "
        "resume ou pressupõe a regra do artigo, de modo que a mudança pode deixar o "
        "trecho desatualizado, errado ou sem base. Falar de assunto só parecido "
        "(outro tipo de prazo, outro pagamento, outra instituição, outras pessoas) "
        "NÃO é depender. Na dúvida, responda que não depende. "
        'Responda só com JSON: {"depende": true ou false, "motivo": "uma frase curta, '
        'em português simples, para quem não é da área jurídica"}.'
    )
    DEPENDENCY_TIMEOUT = 20.0

    def verify_dependency(self, article_before: str, article_after: str,
                          passage: str) -> "tuple[bool, str]":
        """
        O trecho `passage` (de outro documento) depende do artigo que mudou?
        Devolve (depende, motivo). É o `verificar` de impact.analyze_impact.

        Chamada avulsa, como summarize_change: não usa nem altera
        conversation_history. Resposta fora do formato levanta ValueError;
        a análise de impacto trata o erro mantendo a sugestão sem confirmação.
        """
        import json
        import re

        user = (
            f"ARTIGO ANTES DA MUDANÇA:\n{article_before[:3000]}\n\n"
            f"ARTIGO DEPOIS DA MUDANÇA:\n{article_after[:3000]}\n\n"
            f"TRECHO DO OUTRO DOCUMENTO:\n{passage[:2000]}\n\n"
            "O trecho depende do artigo?"
        )
        extra = {"reasoning_effort": "low"} if "gpt-oss" in self.model else {}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.DEPENDENCY_SYSTEM_PROMPT},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
            max_completion_tokens=400,
            stream=False,
            timeout=self.DEPENDENCY_TIMEOUT,
            **extra,
        )
        text = (response.choices[0].message.content or "").strip()
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"resposta fora do formato: {text[:120]!r}")
        data = json.loads(m.group(0))
        depende = data.get("depende")
        if isinstance(depende, str):
            depende = depende.strip().lower() in ("true", "sim", "yes")
        if not isinstance(depende, bool):
            raise ValueError(f"resposta sem 'depende': {text[:120]!r}")
        return depende, " ".join(str(data.get("motivo", "")).split())

    # ── Reescrita da pergunta para a busca ───────────────────

    REWRITE_SYSTEM_PROMPT = (
        "Você ajuda um sistema de busca em documentos normativos (regulamentos, "
        "contratos, manuais, editais, leis). Reescreva a pergunta do usuário em "
        "3 variantes: a primeira como o próprio documento escreveria, com os termos "
        "formais ou jurídicos; a segunda em linguagem simples, com sinônimos comuns; "
        "a terceira só com as palavras-chave principais, com a ortografia corrigida.\n"
        "Responda APENAS com as 3 linhas, sem numeração, títulos ou comentários.\n"
        "Não responda à pergunta. Não acrescente valores, datas, números de artigo "
        "nem fatos que não estejam na pergunta.\n\n"
        "Exemplo para 'posso levar meu cachorro pro serviço?':\n"
        "É permitida a entrada de animais de estimação nas dependências do local de trabalho?\n"
        "Posso levar meu bicho de estimação para o trabalho?\n"
        "animais de estimação local de trabalho permissão"
    )

    def reescrever_pergunta(self, pergunta: str, timeout: float = 8.0) -> List[str]:
        """
        Variantes da pergunta para a busca (busca.buscar(..., reescrever=...)).

        Chamada avulsa, como summarize_change: NÃO usa nem altera
        conversation_history. Levanta exceção em caso de erro; quem chama
        (busca.buscar) ignora a falha e segue com a pergunta original.
        """
        extra = {}
        if self.model.startswith("openai/gpt-oss"):
            extra["reasoning_effort"] = "low"     # menos latência; a tarefa é simples
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": pergunta},
            ],
            temperature=0.2,
            max_completion_tokens=600,
            top_p=1.0,
            stream=False,
            timeout=timeout,
            **extra,
        )
        text = response.choices[0].message.content or ""
        # Modelos com raciocínio visível (qwen) podem devolver <think>…</think>.
        if "</think>" in text:
            text = text.split("</think>")[-1]
        variantes = []
        for line in text.splitlines():
            line = line.strip().lstrip("-*•").strip()
            if line[:2] in ("1)", "2)", "3)", "1.", "2.", "3."):
                line = line[2:].strip()
            # Linha que repete a instrução ("em linguagem simples:") não é variante.
            if line and not line.endswith(":") and line not in variantes:
                variantes.append(line)
        logger.info(f"Pergunta reescrita em {len(variantes)} variantes")
        return variantes[:3]


# ── Função auxiliar ──────────────────────────────────────────

def quick_response(
    message: str,
    system_prompt: str = "Você é um assistente útil.",
    stream: bool = False,
) -> str | Generator[str, None, None]:
    chain = LLMChain(stream=stream)
    if stream:
        return chain.generate_response_stream(message, system_prompt)
    return chain.generate_response(message, system_prompt)