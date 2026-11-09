# Histórico de versões com análise de impacto

Melhoria feita na Atividade 04 da disciplina de Agentes (UFPB, Prof. Andrei Formiga), executando o plano da Atividade 03.
Equipe: Luiz Alberto Rezende Costa e João Gabriel Nascimento Rodrigues.
Base: commit `36f10ca` de [kaio-w-b/EixoAi](https://github.com/kaio-w-b/EixoAi). O trabalho fica na branch `feature/versionamento-documentos` e não foi enviado ao repositório original.

## O que mudou para quem usa

Antes, o EixoAI era um chat com **um PDF por vez**: cada upload apagava a base inteira (`retriever.clear_all()`), e a versão anterior se perdia.

Agora:

1. **Vários documentos na mesma base.** Subir um PDF não apaga os outros. O chat pode buscar em todos ou em um só (campo "Buscar em").
2. **Versões.** Na aba **🗂️ Versões**, cada envio registra autor, data, motivo e o SHA-256 do arquivo. Mandar o mesmo arquivo de novo não cria versão.
3. **Só a versão vigente responde.** O texto de cada versão fica no SQLite, e os chunks que mudaram ficam no ChromaDB marcados como não vigentes. A busca do chat só vê a versão vigente. Chunks iguais entre versões não são duplicados.
4. **Comparação por artigo.** Mostra os artigos alterados (com o trecho que mudou, por exemplo `"30 (trinta)" → "15 (quinze)"`), removidos e incluídos.
5. **Resumo em linguagem simples**, gerado pela LLM. Sem `GROQ_API_KEY`, aparece o resumo automático do diff.
6. **Aviso de impacto.** Ao registrar uma versão nova, uma faixa vermelha avisa quantos trechos de **outros documentos** dependem dos artigos que mudaram:
   - ✅ **citação direta** (certo): "art. 12 do Regulamento", "arts. 3º e 7º do Regulamento Interno", "Regulamento Interno, art. 12, § 1º";
   - 💡 **sugestão** (semelhança): frases parecidas com o texto antigo do artigo, acima de um limiar mínimo, que não citam o artigo.

## Arquivos

| Arquivo | O que é |
|---|---|
| `src/versioning.py` (novo) | Registro de versões em SQLite (`data/versions.db`), divisão por artigo, comparação com `difflib`, `publish_version` |
| `src/impact.py` (novo) | Citações diretas (regex) e sugestões semânticas; relatório em Markdown |
| `src/retriever.py` | Metadados `is_current`, `version` e `version_to` nos chunks; `add_version` incremental e `rollback_version`; `search` filtra a versão vigente e aceita `document_id`/`exclude_document_id`; `search_many` em lote; `format_context`; `get_chunks`; `db_path` configurável |
| `src/llm_chain.py` | `summarize_change(antes, depois)`, chamada avulsa que não toca o histórico do chat |
| `src/app.py` | Sem `clear_all()` no upload; estado por sessão com `gr.State`; aba "Versões"; "Limpar tudo" virou "Limpar conversa"; paleta vermelha, preta e branca (`THEME`); modelo carregado em segundo plano |
| `src/eval/evaluator.py` | A busca da avaliação fica restrita ao documento avaliado |
| `src/eval/eval_versoes.py` (novo) | Mede os critérios da Atividade 03 no conjunto de exemplo |
| `src/eval/exemplos.py` (novo) | Gera os PDFs de exemplo |
| `src/test_suite/test_versioning.py`, `test_impact.py` (novos) | Testes automáticos das partes novas |
| `data/exemplos/` (novo) | Regulamento (v1 e v2), Contrato Modelo, Manual do Colaborador, gabarito e PDFs |

## Como rodar

Python 3.12. Na raiz do projeto:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac
pip install -r requirements.txt
```

Crie um `.env` na raiz com `GROQ_API_KEY=...` (o chat e o resumo pela LLM precisam dela; o versionamento e a análise de impacto funcionam sem).

```bash
cd src
python app.py
```

Abre em `http://localhost:7860`. Com `share=True` (já estava no projeto), o Gradio também imprime um link público `*.gradio.live` que dura 72 horas: é por ele que outra pessoa usa o sistema ao vivo, sem instalar nada.

> Rode de dentro de `src/`: o projeto original usa caminhos relativos (`../vector_db`, `../data/uploads`).

## Como testar

### Testes automáticos

```bash
cd src
python test.py --module versioning   # 21 testes, não precisa de chave
python test.py --module impact       # 16 testes, não precisa de chave
python test.py --all                 # todas as suítes (as de LLM precisam da GROQ_API_KEY)
```

### Critérios da Atividade 03

```bash
cd src
python eval/eval_versoes.py          # critérios 1 a 3
python eval/eval_versoes.py --sweep  # precisão das sugestões por limiar
python test.py --sample              # critério 4: Hit Rate na CF/88 (EVAL_PDF_PATH no .env)
```

Resultado em 2026-11-09 (conjunto de `data/exemplos/`: 3 documentos interligados, 10 referências cruzadas, 5 alterações):

| Critério | Meta | Resultado |
|---|---|---|
| 1. Artigos alterados detectados | 100% | **100%** (5/5: arts. 3, 7 e 12 alterados, art. 15 removido, art. 17 incluído). Também aponta o preâmbulo ("Versão 1" → "Versão 2"), que mudou de fato |
| 2. Citações diretas encontradas | 100% | **100%** (10/10), nenhuma citação falsa; as 5 iscas ("art. 12 do Código Civil", "art. 5º" da Constituição, artigos que não mudaram) ficaram de fora |
| 3. Precisão das sugestões | ≥ 70% | **100%** (4/4 frases sugeridas dependem mesmo de um artigo alterado). Em 2 das 4, o artigo apontado não é o exato do gabarito |
| 4. Hit Rate da CF/88 | não cair | **83,3% antes e 83,3% depois**, idêntico pergunta a pergunta (48 perguntas, 857 chunks). MRR 68,9%, NDCG 70,6%, Precision@5 32,1% nos dois |

Sobre os números:

- O critério 3 tem só 4 sugestões, num conjunto feito por nós, e a regra foi ajustada nele. É um resultado otimista. A **primeira versão** (só limiar de semelhança) deu **50%** (5/10) em qualquer limiar de 0,80 a 0,87. O e5 põe quase toda frase jurídica entre 0,87 e 0,90 de qualquer artigo. O que resolveu foi a regra do "artigo mais parecido": a frase é comparada com todos os artigos da versão antiga e só vira sugestão se o mais parecido for um dos que mudaram.
- Critério 4: o PDF usado foi a CF/88 do STF (`stf.jus.br/arquivo/cms/legislacaoConstituicao/anexo/CF.pdf`), medido no commit original e na branch nova, cada um com banco temporário próprio. O README original cita 79,2% com 1.499 chunks, que deve ser outro PDF.
- **Latência:** numa primeira medição, a busca pareceu ir de 96 ms para 202 ms, mas era ruído (as duas rodaram com a máquina carregada). Medida isolada, a busca custa ~23 ms, dos quais ~20 ms são o embedding da pergunta; o filtro `is_current` acrescenta ~4 ms. Depois da otimização, a avaliação da CF/88 mediu 32 ms por consulta.

Testes: `versioning` 21/21, `impact` 16/16, e as suítes originais `ingester` 7/7 e `retriever` 12/12. As suítes `llm_chain` (1/11) e `integration` (1/2) falham sem `GROQ_API_KEY`, igual no código original.

### Velocidade

Medido na CF/88 (857 chunks), CPU, com uma revisão de 3 artigos:

| Etapa | 1ª versão da melhoria | Agora | O que mudou |
|---|---|---|---|
| Publicar a revisão | 11,5 s | **1,1 s** | `add_version` só grava os chunks que mudaram e só aposenta os que saíram; antes, a coleção recebia todos os 857 de novo e remarcava os 1.700 |
| Análise de impacto | 5,7 s | **2,6 s** | artigos antigos cortados em 600 caracteres antes do embedding (o art. 5º tem ~15 mil), cache por versão, frases e buscas em lote |
| Primeiro upload depois de abrir o app | +9–15 s | 0 s | o modelo carrega em segundo plano quando o app sobe (`warm_up`) |
| Pergunta no chat | 2 buscas | 1 busca | o contexto é montado com os resultados já buscados (antes, `get_context` buscava de novo) |

### Revisão de código

Uma revisão independente do código novo levantou 14 pontos. Corrigidos:

- **Estilo CLT:** "Art. 5º - A todo trabalho…" virava artigo "5-A". Agora a letra só conta com hífen colado ("12-A").
- **Cabeçalho tomado por citação:** "Art. 3º Os prazos previstos no Regulamento…", num manual, contava como citação do art. 3 do Regulamento. Agora o cabeçalho do próprio documento é ignorado. Entre o número e o nome do documento só passam "§ 2º", "inciso II", "parágrafo único", "alínea b" e "caput".
- **Documento sem artigos:** mostrava "Art. documento". Agora diz que a comparação é do texto inteiro.
- **Resumo reserva:** o resumo sem LLM ficava gravado e impedia o resumo da LLM depois que a chave fosse configurada. Agora só o da LLM é gravado.
- **Publicações simultâneas** (aba Chat e aba Versões ao mesmo tempo): pegavam o mesmo número de versão e dessincronizavam o SQLite e o ChromaDB. Agora há uma trava, e a falha em qualquer etapa desfaz o índice.
- **Diff de texto grande sem artigos:** 12 mil palavras levavam 6 s. Agora usa a heurística `autojunk` do `difflib` acima de 3 mil palavras.
- **Localização das citações:** recalculada por citação (9 s em 3 mil citações). Agora usa `bisect`.
- **Cache dos artigos** sem trava entre threads.
- **Histórico** lia o texto inteiro de cada versão sem usar. Agora não lê.
- **Injeção de Markdown:** texto do PDF e nomes iam crus para `gr.Markdown`, então um PDF podia pôr imagem externa ou link na tela. Agora são escapados, os blocos de código usam cercas mais longas que as crases do texto, e imagens são removidas do resumo da LLM.
- **Sem chave do Groq**, o chat nem buscava. Agora busca e mostra os trechos.

Ficou documentado, sem correção: chunks indexados pela versão original não têm `is_current` e não aparecem (reprocesse os PDFs); e `python test.py --sample` usa o banco padrão do app durante a avaliação (rode com `CHROMADB_PATH` apontando para uma pasta temporária).

### Roteiro de demonstração (5 minutos)

1. Aba **🗂️ Versões**. Envie `data/exemplos/pdf/regulamento_v1.pdf` com o documento **Regulamento Interno**, apelido `Regulamento`, autor e motivo.
2. Envie `contrato_modelo.pdf` como **Contrato Modelo** e `manual.pdf` como **Manual do Colaborador**.
3. Envie `regulamento_v2.pdf` escolhendo **Regulamento Interno** na lista. Aparece a faixa vermelha: 3 artigos alterados, 1 removido e 1 incluído, com 10 citações diretas em 2 documentos. Abaixo vêm o resumo, a lista de impacto e a comparação por artigo.
4. Em **Histórico**, escolha o Regulamento: as duas versões aparecem com autor, data, motivo e SHA-256. Compare a v1 com a v2.
5. Aba **💬 Chat**: pergunte "qual o prazo de reembolso?". A resposta vem da v2 (15 dias), não da v1 (30 dias).

## Decisões e limites

- **Artigo é "Art. N" no começo da linha.** Pela LC 95/98, o caput começa com "Art." e a citação no meio do texto usa "art." minúsculo. A divisão do chunker original (`(?=\bArt\.\s*\d+)`) também corta em citações com "Art." maiúsculo no meio da frase. Por isso, para comparar versões, usamos a âncora de linha. A divisão dos chunks ficou igual para não mexer no Hit Rate.
- **Mudança no chunker:** a detecção de "texto com artigos" ignorava maiúsculas e minúsculas e casava com qualquer citação. Um manual que só *cita* "art. 12" ia para o chunker jurídico e virava um chunk gigante. Agora ele exige artigo no começo da linha. A CF/88 não muda, porque tem artigos no começo de linha.
- **Renumeração:** se um artigo é removido e os seguintes são renumerados, a comparação mostra todos como alterados. Não tratamos esse caso.
- **Apelidos:** a citação só é reconhecida pelo nome do documento ou pelos apelidos informados no envio ("Regulamento", "RI"). "art. 12 do Regulamento de Estágio" casaria com o apelido "Regulamento".
- **Sugestões:** o ChromaDB acha os chunks candidatos; dentro deles, cada frase é comparada com todos os artigos da versão antiga e só vira sugestão se o mais parecido foi alterado ou removido, com semelhança mínima `IMPACT_MIN_RELEVANCE` (padrão 0,85). A regra foi ajustada no conjunto de exemplo; com documentos reais, vale medir de novo.
- **Gradio 6.3 e abas:** um componente de aba oculta que recebe atualização fica preso com o "carregando" por cima quando a aba aparece, e não aceita clique. Por isso, cada aba se atualiza ao ser aberta (`Tab.select`) em vez de receber atualização da outra aba.
- **Upload pelo chat:** o nome do documento vem do nome do arquivo, então `regulamento_v1.pdf` e `regulamento_v2.pdf` viram documentos diferentes. Para registrar versões, use a aba Versões.
- **Bases antigas:** chunks indexados pela versão original não têm `is_current` e não aparecem na busca. Reprocesse os PDFs.
- Fica de fora, como combinado na Atividade 03: notificação por e-mail ou Teams, login, confirmação de ciência, edição dentro do sistema e escolha do modelo.
