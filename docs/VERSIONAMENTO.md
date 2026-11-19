# Histórico de versões com análise de impacto

Melhoria feita na Atividade 04 da disciplina de Agentes (UFPB, Prof. Andrei Formiga), executando o plano da Atividade 03.
Equipe: Luiz Alberto Rezende Costa e João Gabriel Nascimento Rodrigues.
Base: commit `36f10ca` de [kaio-w-b/EixoAi](https://github.com/kaio-w-b/EixoAi). O trabalho fica na branch `feature/versionamento-documentos` e não foi enviado ao repositório original.

## O que mudou para quem usa

Antes, o EixoAI era um chat com **um PDF por vez**: cada upload apagava a base inteira (`retriever.clear_all()`), e a versão anterior se perdia.

Agora:

1. **Vários documentos na mesma base.** Subir um PDF não apaga os outros. O chat pode buscar em todos ou em um só (campo "Procurar em").
2. **Versões.** Na aba **Documentos**, cada envio registra quem enviou (pelo login), data, motivo e o SHA-256 do arquivo. Mandar o mesmo arquivo de novo não cria versão.
3. **Só a versão vigente responde.** O texto de cada versão fica no SQLite, e os chunks que mudaram ficam no ChromaDB marcados como não vigentes. A busca do chat só vê a versão vigente. Chunks iguais entre versões não são duplicados.
4. **Comparação por artigo.** Mostra os artigos alterados (com o trecho que mudou, por exemplo `"30 (trinta)" → "15 (quinze)"`), removidos e incluídos.
5. **Resumo em linguagem simples**, gerado pela LLM. Sem `GROQ_API_KEY`, aparece o resumo automático do diff.
6. **Aviso de impacto.** Ao registrar uma versão nova, uma faixa amarela avisa quantos trechos de **outros documentos** dependem dos artigos que mudaram:
   - **citação direta** (certo): "art. 12 do Regulamento", "arts. 3º e 7º do Regulamento Interno", "Regulamento Interno, art. 12, § 1º";
   - **sugestão** (possível): frases que não citam o artigo, mas tratam do mesmo assunto, em documentos do mesmo tema do documento alterado. Se houver LLM, ela confirma cada uma e diz o motivo. Na tela, aparecem como "pode depender do art. 12", e as citações como "cita diretamente o art. 12".

## Arquivos

| Arquivo | O que é |
|---|---|
| `src/versioning.py` (novo) | Registro de versões em SQLite (`data/versions.db`), divisão por artigo, comparação com `difflib`, `publish_version` |
| `src/impact.py` (novo) | Citações diretas (regex) e sugestões semânticas (filtro de tema por documento, artigo mais parecido, verificação opcional por LLM); relatório em Markdown para leigos |
| `src/retriever.py` | Metadados `is_current`, `version` e `version_to` nos chunks; `add_version` incremental e `rollback_version`; `search` filtra a versão vigente e aceita `document_id`/`exclude_document_id`; `search_many` em lote; `format_context`; `get_chunks`; `db_path` configurável |
| `src/llm_chain.py` | `summarize_change(antes, depois)`, `verify_dependency` (confirma sugestões) e `reescrever_pergunta` (variantes para a busca), chamadas avulsas que não tocam o histórico do chat; modelo padrão em `GROQ_MODEL` |
| `src/app.py` | Abas Perguntar, Documentos e Equipe; login; escolha do modelo com reserva no limite do Groq; tempo real com `gr.Timer`; estado por sessão com `gr.State`; paleta e textos para leigos, sem emoji |
| `src/busca.py` (novo) | Busca híbrida: embeddings + BM25 com fusão RRF, correção de digitação e variantes opcionais pela LLM |
| `src/equipe.py` (novo) | Contas do `.env`, conversa geral e por documento, presença, leitura e atividade recente |
| `iniciar.cmd`, `.env.exemplo` (novos) | Ligar com dois cliques; modelo do `.env` |
| `src/eval/evaluator.py` | A busca da avaliação fica restrita ao documento avaliado |
| `src/eval/eval_versoes.py` (novo) | Mede os critérios da Atividade 03 no conjunto de exemplo |
| `src/eval/exemplos.py` (novo) | Gera os PDFs de exemplo |
| `src/test_suite/test_versioning.py`, `test_impact.py`, `test_busca.py`, `test_equipe.py` (novos) | Testes automáticos das partes novas |
| `src/eval/eval_busca.py`, `src/eval/parafrases.json` (novos) | Avaliação da busca com perguntas escritas de 3 formas |
| `data/exemplos/` (novo) | Regulamento (v1 e v2), Contrato Modelo, Manual do Colaborador, gabarito e PDFs |

## Como rodar

**Jeito simples (Windows):** dois cliques em `iniciar.cmd`, na raiz. Na primeira vez ele cria o ambiente, instala as dependências e cria o `.env` a partir do `.env.exemplo` (preencha a chave do Groq e os usuários). Depois disso, abre o navegador e pede só o login. Detalhes do uso em equipe em [TESTE_EM_DUPLA.md](TESTE_EM_DUPLA.md).

**Na mão:** Python 3.12. Na raiz do projeto:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux/Mac
pip install -r requirements.txt
```

Copie `.env.exemplo` para `.env` e preencha `GROQ_API_KEY` (o chat e o resumo pela LLM precisam dela; o versionamento e a análise de impacto funcionam sem) e `EIXOAI_USUARIOS` (sem ela, o app abre sem login).

```bash
cd src
python app.py
```

Abre em `http://localhost:7860`, e o terminal mostra também os endereços desta máquina na rede e na VPN, para a equipe entrar. Com `EIXOAI_LINK_PUBLICO=1` (padrão; o projeto já usava `share=True`), o Gradio tenta um link público `*.gradio.live` que dura 72 horas. Em alguns Windows o antivírus bloqueia o programa do túnel (`frpc`); aí, use a rede local ou uma VPN como o Radmin.

> Rode de dentro de `src/`: o projeto original usa caminhos relativos (`../vector_db`, `../data/uploads`).

## Como testar

### Testes automáticos

```bash
cd src
python test.py --module versioning   # 21 testes, não precisa de chave
python test.py --module impact       # 22 testes, não precisa de chave
python test.py --all                 # todas as suítes (as de LLM precisam da GROQ_API_KEY)
```

### Critérios da Atividade 03

```bash
cd src
python eval/eval_versoes.py          # critérios 1 a 3 e documentos de outro tema
python eval/eval_versoes.py --sweep  # precisão das sugestões por limiar (frase e tema)
python eval/eval_versoes.py --verificar-llm          # a LLM confirma cada sugestão (precisa da chave)
python eval/eval_versoes.py --base-real copia.db     # mede também uma base registrada pelo app
python test.py --sample              # critério 4: Hit Rate na CF/88 (EVAL_PDF_PATH no .env)
```

Resultado em 2026-11-09 (conjunto de `data/exemplos/`: 3 documentos interligados, 10 referências cruzadas, 5 alterações):

| Critério | Meta | Resultado |
|---|---|---|
| 1. Artigos alterados detectados | 100% | **100%** (5/5: arts. 3, 7 e 12 alterados, art. 15 removido, art. 17 incluído). Também aponta o preâmbulo ("Versão 1" → "Versão 2"), que mudou de fato |
| 2. Citações diretas encontradas | 100% | **100%** (10/10), nenhuma citação falsa; as 5 iscas ("art. 12 do Código Civil", "art. 5º" da Constituição, artigos que não mudaram) ficaram de fora |
| 3. Precisão das sugestões | ≥ 70% | **100%** (4/4 frases sugeridas dependem mesmo de um artigo alterado). Em 2 das 4, o artigo apontado não é o exato do gabarito. Com a verificação por LLM: **100%** (2/2), e as 2 recusadas são justamente as do artigo errado |
| Documentos de outro tema (novo, 2026-11-19) | 0 sugestões | **0**, com e sem LLM: edital de mestrado, proposta de projeto e regulamento de condomínio inventados (`OUTROS_TEMAS` em `eval_versoes.py`) e a base real do teste ao vivo. Antes da correção: 4 e 7 sugestões falsas |
| 4. Hit Rate da CF/88 | não cair | **83,3% antes e 83,3% depois**, idêntico pergunta a pergunta (48 perguntas, 857 chunks). MRR 68,9%, NDCG 70,6%, Precision@5 32,1% nos dois |

Sobre os números:

- O critério 3 tem só 4 sugestões, num conjunto feito por nós, e a regra foi ajustada nele. É um resultado otimista. A **primeira versão** (só limiar de semelhança) deu **50%** (5/10) em qualquer limiar de 0,80 a 0,87. O e5 põe quase toda frase jurídica entre 0,87 e 0,90 de qualquer artigo. O que resolveu foi a regra do "artigo mais parecido": a frase é comparada com todos os artigos da versão antiga e só vira sugestão se o mais parecido for um dos que mudaram.
- **Sugestões entre documentos de temas diferentes (corrigido em 2026-11-19).** No teste ao vivo, a base tinha o Regulamento (v1 e v2), um edital de seleção de mestrado (163 chunks) e uma proposta de projeto de mestrado. Comparar v1 com v2 gerou **7 sugestões, todas falsas e todas no edital**, por exemplo "O candidato deverá estar conectado com antecedência mínima de 5 (cinco) minutos" como dependente do art. 7 (férias com "antecedência mínima de 60 dias"), a 0,888. A regra do "artigo mais parecido" não protege desse caso: entre os artigos do Regulamento, sempre há um mais parecido, e se ele mudou a frase passa. Medido frase a frase, nenhum sinal separava as falsas das certas: semelhança 0,85–0,89 nas falsas contra 0,87–0,92 nas certas; a vantagem sobre o segundo artigo, a distância da mediana e o z-score se sobrepõem do mesmo jeito; os prefixos `query:`/`query:` (indicados pelo e5 para tarefas simétricas) e `query:`/`passage:` deixaram passar 22 e 12 frases falsas do edital e da proposta (e 1 falsa no exemplo, cada um), contra 8 com `passage:`/`passage:`, que ficou. O que separa é o **tema do documento**, medido no documento inteiro.
- **A regra nova** tem três passos: (1) **filtro de tema**: cada documento candidato recebe uma nota de tema, a média de duas vistas, o começo (primeiros 2.000 caracteres, onde o documento diz do que trata) e as partes (cada chunk do candidato contra o chunk mais parecido do documento alterado, média do quarto mais parecido, com os embeddings que já estão no ChromaDB). Abaixo de `IMPACT_MIN_DOC_SIMILARITY` (padrão 0,87), o documento não recebe sugestão; citação direta continua valendo. (2) a regra do artigo mais parecido, com `IMPACT_MIN_RELEVANCE` 0,85, igual antes. (3) **verificação por LLM, opcional**: `analyze_impact(..., verificar=llm.verify_dependency)` pergunta, para cada sugestão que sobrou (no máximo 8, duas por vez, 30 s no total), se o trecho depende do artigo, e recebe `(depende, motivo)`. A recusada sai do relatório; erro ou demora deixa a sugestão sem confirmação, como sem chave.
- Notas de tema contra o Regulamento, medidas em 2026-11-19 com 12 documentos (os 2 do exemplo, os 2 do teste ao vivo e 8 inventados só para a medição): **mesmo tema** Manual 0,940, Contrato 0,932, Política de Viagens 0,910, Código de Conduta 0,903, Comunicado de Férias 0,881; **outro tema** Privacidade de app de saúde 0,862, Condomínio 0,862, Regimento escolar 0,858, Proposta de projeto 0,840, Edital 0,839, Locação 0,840, Bula 0,817. O limiar 0,87 fica no meio da folga (0,862 a 0,881), que é estreita: o Comunicado de Férias passa por 0,011. Na varredura (`--sweep`), o exemplo mantém 4/4 de 0 a 0,92, e os documentos inventados de outro tema recebem 4 sugestões sem o filtro, 1 com 0,84 ou 0,85 e 0 a partir de 0,86.
- Com a LLM (`openai/gpt-oss-120b`), sem o filtro de tema, as 7 frases falsas do edital foram recusadas (7/7), com motivos como "O trecho fala sobre conexão antes de entrevista, não sobre solicitação de férias". No exemplo com os documentos inventados, das 6 sugestões que passam no filtro ficaram 3, todas certas; as 3 recusadas são uma falsa ("Diárias…", art. 7) e duas certas apontadas ao artigo errado. A verificação levou 2–3 s.
- Critério 4: o PDF usado foi a CF/88 do STF (`stf.jus.br/arquivo/cms/legislacaoConstituicao/anexo/CF.pdf`), medido no commit original e na branch nova, cada um com banco temporário próprio. O README original cita 79,2% com 1.499 chunks, que deve ser outro PDF.
- **Latência:** numa primeira medição, a busca pareceu ir de 96 ms para 202 ms, mas era ruído (as duas rodaram com a máquina carregada). Medida isolada, a busca custa ~23 ms, dos quais ~20 ms são o embedding da pergunta; o filtro `is_current` acrescenta ~4 ms. Depois da otimização, a avaliação da CF/88 mediu 32 ms por consulta.

Testes: `versioning` 21/21, `impact` 22/22, e as suítes originais `ingester` 7/7 e `retriever` 12/12. As suítes `llm_chain` (1/11) e `integration` (1/2) falham sem `GROQ_API_KEY`, igual no código original.

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

### Roteiro de demonstração (5 minutos, de preferência com duas pessoas logadas)

1. Aba **Documentos** › **Enviar arquivo**. Envie `data/exemplos/pdf/regulamento_v1.pdf` como "Um documento novo", com o nome **Regulamento Interno** e, em "Mais opções", o apelido `Regulamento`.
2. Envie `contrato_modelo.pdf` como **Contrato Modelo** e `manual.pdf` como **Manual do Colaborador**. A outra pessoa recebe um aviso a cada envio.
3. A outra pessoa, na aba **Perguntar**: "qual o prazo de reembolso?". A resposta é 30 dias (v1), com o documento e o artigo citados e as fontes ao lado.
4. Envie `regulamento_v2.pdf` escolhendo "Nova versão de Regulamento Interno". Aparece a seção **Mudanças**: faixa amarela com 3 artigos alterados, 1 removido e 1 incluído, 10 citações diretas em 2 documentos, resumo e a lista de quem é afetado.
5. A outra pessoa repete a pergunta, sem recarregar: agora é 15 dias (v2).
6. Em **Histórico**, as duas versões aparecem com quem enviou, quando, motivo, o link para o PDF de cada uma e o código de verificação (SHA-256). **Comparar** v1 com v2. Em **Documento**, o PDF da versão atual aparece na tela.
7. Em **Conversa** (do documento) ou na aba **Equipe**, uma pessoa escreve e a outra recebe na hora.

## Decisões e limites

- **Artigo é "Art. N" no começo da linha.** Pela LC 95/98, o caput começa com "Art." e a citação no meio do texto usa "art." minúsculo. A divisão do chunker original (`(?=\bArt\.\s*\d+)`) também corta em citações com "Art." maiúsculo no meio da frase. Por isso, para comparar versões, usamos a âncora de linha. A divisão dos chunks ficou igual para não mexer no Hit Rate.
- **Mudança no chunker:** a detecção de "texto com artigos" ignorava maiúsculas e minúsculas e casava com qualquer citação. Um manual que só *cita* "art. 12" ia para o chunker jurídico e virava um chunk gigante. Agora ele exige artigo no começo da linha. A CF/88 não muda, porque tem artigos no começo de linha.
- **Renumeração:** se um artigo é removido e os seguintes são renumerados, a comparação mostra todos como alterados. Não tratamos esse caso.
- **Apelidos:** a citação só é reconhecida pelo nome do documento ou pelos apelidos informados no envio ("Regulamento", "RI"). "art. 12 do Regulamento de Estágio" casaria com o apelido "Regulamento".
- **Sugestões:** o ChromaDB acha os chunks candidatos; o documento de cada um tem de ser do mesmo tema do documento alterado (`IMPACT_MIN_DOC_SIMILARITY`, padrão 0,87); dentro dos que sobram, cada frase é comparada com todos os artigos da versão antiga e só vira sugestão se o mais parecido foi alterado ou removido, com semelhança mínima `IMPACT_MIN_RELEVANCE` (padrão 0,85); se a interface passar um verificador, a LLM confirma ou recusa cada uma. Limites: os dois limiares foram ajustados em poucos documentos, e a folga do filtro de tema é estreita (0,862 a 0,881); um documento de outro tema muito próximo (outro regulamento trabalhista, de outra empresa) pode passar, e um documento do mesmo tema com começo fora do assunto (capa, sumário longo) pode cair. A sugestão aponta o artigo mais parecido, que às vezes não é o certo (2 de 4 no exemplo); a LLM, que só vê esse artigo, recusa esses casos. O plano gratuito do Groq dá 8 mil tokens por minuto e cada verificação gasta perto de 600; com o limite estourado, as sugestões aparecem sem confirmação.
- **Gradio 6.3 e abas:** um componente de aba oculta que recebe atualização fica preso com o "carregando" por cima quando a aba aparece, e não aceita clique. Por isso, cada aba se atualiza ao ser aberta (`Tab.select`) em vez de receber atualização da outra aba.
- **Envio único:** o envio pelo chat saiu (ele criava documentos diferentes para `regulamento_v1.pdf` e `regulamento_v2.pdf`). Agora todo envio é na aba Documentos, escolhendo "Um documento novo" ou "Nova versão de…".
- **Bases antigas:** chunks indexados pela versão original não têm `is_current` e não aparecem na busca. Reprocesse os PDFs.
- A Atividade 03 deixou de fora notificação por e-mail ou Teams, login, confirmação de ciência, edição dentro do sistema e escolha do modelo. Depois do teste em dupla, entraram como extras o **login**, a **escolha do modelo** e os **avisos dentro do app** (não por e-mail): ver [TESTE_EM_DUPLA.md](TESTE_EM_DUPLA.md). Continuam de fora e-mail/Teams, confirmação de ciência e edição dentro do sistema.
