# Segunda rodada: o que mudou depois do teste em dupla

Em 2026-11-19, João e Luiz usaram o EixoAI ao mesmo tempo, cada um no seu computador, ligados por VPN (Radmin) ao app rodando no computador do João. O teste mostrou problemas que o teste individual não mostrava. Este documento registra o que foi relatado, o que mudou e como conferir.

## O que foi relatado

| # | Relato | Quem |
|---|---|---|
| 1 | "Quando coloco um PDF eu fico preso e não consigo colocar mais" | João |
| 2 | "As mudanças não aparecem em tempo real" | João |
| 3 | Não dá para navegar por vários documentos | Luiz |
| 4 | Falta conversa entre as pessoas: um chat geral e um específico, ver o que os outros fazem | João |
| 5 | Dependendo das palavras da pergunta, a busca não acha nada; mudando algumas palavras, acha | Luiz |
| 6 | Dois arquivos de temas diferentes: um foi sugerido como afetado pelo outro, sem relação nenhuma | Luiz |
| 7 | Interface: limpa, minimalista, sem emoji, cabendo na tela, fácil para quem não conhece os termos | Luiz |
| 8 | Poder escolher o modelo do Groq, como no Claude | Luiz |
| 9 | Ao ligar, só fazer o login e já usar | Luiz |

## O que mudou

### 1. Envio que travava
Depois de processar, o PDF continuava na caixa de arquivo, e a caixa perdia a área de soltar: para escolher outro, era preciso achar o "x" pequeno no canto. Quem clicava em "Processar" de novo reenviava o mesmo arquivo (foi o que apareceu duas vezes no log). Agora a caixa se esvazia sozinha depois de cada envio. Reproduzido e conferido com navegador automático (Playwright).

### 2. Tempo real
O Gradio não empurra novidades do servidor para a tela. Cada tela passou a perguntar, a cada 3 segundos (`gr.Timer`), se surgiu versão ou mensagem nova (`VersionStore.last_change`, `Equipe.ultima_mensagem`, consultas baratas no SQLite). Se surgiu:
- quem **não** fez a mudança recebe um aviso ("João Gabriel publicou a versão 2 de Regulamento Interno");
- a aba aberta se atualiza (lista de documentos, histórico, conversas).

A aba fechada só se atualiza quando é aberta: no Gradio 6.3, componente de aba oculta que recebe atualização fica preso com o "carregando". Pelo mesmo motivo, as seções do documento trocam em dois passos (primeiro aparecem, depois recebem o conteúdo); antes disso, o histórico vinha vazio.

### 3. Navegar por vários documentos
A aba **Documentos** tem a lista de todos os documentos da equipe, com a versão atual. Ao escolher um, aparecem:
- **Texto atual**, com cada artigo em destaque, para ler na tela;
- **Mudanças**, com a última mudança (ou a comparação escolhida): o que mudou, resumo e quem é afetado;
- **Histórico**, com todas as versões e a comparação entre quaisquer duas;
- **Conversa**, só sobre aquele documento.

O envio virou um lugar só: "Um documento novo" ou "Nova versão de…". O nome sugerido vem do arquivo, legível (`regulamento_v1.pdf` vira "Regulamento").

### 4. Conversa e trabalho em equipe
Inspirado na "bolinha" que o time da Gravaê usa (feed, recados, presença, novidades desde a última visita), em `src/equipe.py`, no mesmo SQLite das versões:
- **Conversa geral** (aba Equipe) e **conversa por documento** (aba Documentos › Conversa);
- **Quem está aqui**: quem está com o app aberto e o que está vendo ("Documentos: Regulamento Interno (histórico)");
- **Atividade recente**: versões publicadas e mensagens numa linha do tempo só;
- **Mensagens novas**: contador no topo ("2 mensagens novas") e aviso na hora, a menos que a pessoa já esteja vendo aquela conversa.

### 5. Busca que dependia das palavras
Duas causas, medidas antes de mudar:
- **A reescrita da pergunta atrapalhava.** O app pedia à IA para reescrever a pergunta com "termos jurídicos da Constituição Federal", e isso afogava a pergunta: o acerto caía de 71,7% para 51,7%, e nas perguntas curtas de 62,5% para 32,5%. Era a causa principal do "não acha nada".
- **O documento grande toma o topo.** O edital tinha 163 dos 170 trechos da base; nas perguntas sobre os documentos pequenos, 2 a 3 dos 5 resultados vinham do edital. E o modelo de embeddings dá notas muito parecidas a tudo (0,79 a 0,90), então palavras exatas da pergunta ("ponto", "atestado", "estagiário") não pesavam nada.

A busca nova (`src/busca.py`) junta duas listas e faz a fusão (RRF): a do significado (embeddings, como antes) e a das palavras (BM25, sem acento, sem maiúsculas, sem palavras vazias, com radical), com correção de erro de digitação contra o vocabulário da base ("lates" vira "lattes"). Se a IA estiver disponível, ela acrescenta três variantes da pergunta (formal, leiga, palavras-chave), com peso menor e limite de 8 s; se falhar, a busca segue sem ela. Números na seção "Busca".

### 6. Sugestões entre documentos de temas diferentes
No caso real (Regulamento de uma cooperativa contra um edital de mestrado e uma proposta de pesquisa), a comparação v1 → v2 do Regulamento gerava **7 sugestões, todas falsas, todas no edital**. A regra do "artigo mais parecido" sempre acha algum artigo mais perto; se ele mudou, a frase virava sugestão. Nenhum sinal frase a frase separava as falsas das certas (semelhança 0,85–0,89 nas falsas contra 0,87–0,92 nas certas).

O que separa é o tema do documento inteiro. Agora:
1. **Filtro de tema por documento**: documento com nota abaixo de 0,87 contra o documento alterado não recebe sugestão (mesmo tema: 0,881–0,940; outro tema: 0,817–0,862; edital 0,839, proposta 0,840). Citação direta continua valendo sempre.
2. A regra do artigo mais parecido, como antes.
3. **Verificação pela IA** (opcional): "este trecho depende mesmo deste artigo?", com o motivo na tela. No máximo 8 chamadas por análise.

| Medida | Antes | Agora, sem IA | Agora, com IA |
|---|---|---|---|
| Artigos alterados detectados | 100% | 100% | 100% |
| Citações diretas | 10/10, 0 falsas | 10/10, 0 falsas | 10/10, 0 falsas |
| Precisão das sugestões (exemplo) | 4/4 | 4/4 | 2/2 |
| Caso real (edital e proposta) | 7 falsas | **0** | **0** |
| Outros temas inventados | 4 falsas | **0** | **0** |

### 7. Interface
- Paleta de [vianaworks.com.br](https://vianaworks.com.br): azul-petróleo escuro (`#081C25`), texto quase branco (`#EEF3F0`), verde-água nos destaques (`#3DDBC3`), amarelo para pedir atenção (`#F4C542`); fontes General Sans, Clash Display e JetBrains Mono; linhas finas, cantos de 10 px.
- **Sem emoji** em nenhum texto da tela.
- Palavras simples: "trecho" em vez de "chunk", "versão atual" em vez de "vigente", "De onde veio a resposta" em vez de "Rastreamento de chunks", "código de verificação" em vez de "SHA-256". As fontes mostram documento, página e relação (alta, média, baixa), sem distância nem id.
- Tudo cabe na tela: as áreas longas (chat, fontes, texto do documento, conversas) rolam por dentro.
- A IA responde em frases curtas, sem emoji, dizendo de qual documento e artigo veio a informação, e diz claramente quando não achou.

### 8. Escolha do modelo
No topo, "Modelo de IA" lista os modelos de conversa da conta do Groq (consultada a cada 10 minutos), com nomes simples ("GPT-OSS 120B · mais completo", "GPT-OSS 20B · mais rápido", "Qwen 3.8 27B · equilibrado"). A escolha é de cada pessoa e vale para as perguntas, os resumos e a verificação das sugestões; trocar de modelo mantém a conversa.

O plano gratuito do Groq aceita 8 mil tokens por minuto por modelo, e duas pessoas perguntando ao mesmo tempo chegam nesse limite. Quando o modelo escolhido está no limite, o app tenta outros dois e avisa na resposta ("O modelo GPT-OSS 120B chegou ao limite de uso por minuto; esta resposta veio do GPT-OSS 20B"). As novas tentativas automáticas do cliente do Groq, que esperavam 10 s ou mais, foram desligadas.

Também: o modelo original do projeto, `llama-3.3-70b-versatile`, saiu do Groq. O padrão virou `openai/gpt-oss-120b` (`GROQ_MODEL` no `.env` troca).

### 9. Ligar e usar
- `iniciar.cmd`, na raiz: dois cliques. Na primeira vez prepara o ambiente e cria o `.env`; depois abre o navegador e pede só o login.
- **Login** com as contas do `.env` (`EIXOAI_USUARIOS=login:senha:Nome`). O nome de quem enviou cada versão e escreveu cada mensagem vem do login: ninguém publica no nome do outro. Sem a variável, o app abre sem login.
- O terminal mostra os endereços para a equipe entrar (rede local e VPN).

## Conferência com o plano da Atividade 03

| Item do plano | Situação |
|---|---|
| Base com vários documentos e versões, sem apagar | Feito na primeira rodada; agora com navegação por documento |
| Registro de autor, data, motivo e SHA-256 | Feito; o autor agora vem do login |
| Busca restrita à versão vigente | Feito; conferido em dupla (30 dias na v1, 15 dias na v2, sem recarregar) |
| Comparação por artigo e resumo pela LLM | Feito; a comparação mostra o texto de antes e de depois em vez de diff técnico |
| Impacto: citação direta (certo) e sugestão | Feito; a sugestão ganhou filtro de tema e verificação pela IA |
| Aba "Versões" | Virou a aba "Documentos" (lista, texto, mudanças, histórico, conversa) |
| Testes automáticos das partes novas | Feito; mais `test_equipe.py` e os testes novos de impacto e busca |
| Critério 1: artigos alterados 100% | 100% |
| Critério 2: citações diretas 100% | 10/10 |
| Critério 3: precisão das sugestões ≥ 70% | 100% (4/4 sem IA; 2/2 com IA) |
| Critério 4: Hit Rate da CF/88 não cai | 83,3% na avaliação original (caminho do avaliador, igual); no caminho da interface, 83,3% → 95,8% |

**Fora do escopo da Atividade 03 que entrou agora**, por pedido do teste em dupla: login, escolha do modelo e avisos dentro do app. Continuam de fora: notificação por e-mail ou Teams, confirmação de ciência e edição de documentos dentro do sistema.

## Busca

Conjunto novo em `src/eval/parafrases.json`: 40 perguntas, cada uma escrita de 3 formas (técnica, leiga, curta ou com erro de digitação), 26 sobre os documentos de exemplo e 14 sobre o edital, com o trecho esperado. Acerto nos 5 primeiros (Hit@5) e MRR, buscando em todos os documentos:

| Método | Técnica | Leiga | Curta | Total |
|---|---|---|---|---|
| Antes, só a busca | 87,5% / 0,823 | 65,0% / 0,480 | 62,5% / 0,452 | 71,7% / 0,585 |
| Antes, como o app fazia (com a reescrita) | 67,5% / 0,434 | 55,0% / 0,282 | 32,5% / 0,135 | 51,7% / 0,284 |
| Nova, sem IA | 97,5% / 0,863 | 70,0% / 0,503 | 80,0% / 0,626 | 82,5% / 0,664 |
| **Nova, com IA** | **100% / 0,844** | **82,5% / 0,621** | **82,5% / 0,693** | **88,3% / 0,719** |

Com os textos extraídos dos PDFs (e não dos .txt), o total fica 73,3% antes e 87,5% depois.

CF/88 (857 trechos, 48 perguntas): o `test.py --sample` continua em 83,3% (ele usa o caminho do avaliador, que não mudou). No caminho que a interface usa, medido por `eval/eval_busca.py --cf88`: 83,3% antes, **95,8%** com a busca nova sem IA e 93,8% com IA. Passou a acertar 7 perguntas e errou 1 que acertava ("república ou monarquia").

Tempo: 25 a 27 ms por busca sem IA (22 a 25 ms antes); com IA, mais 0,5 a 1 s da chamada ao Groq.

Limites: o conjunto é pequeno e foi escrito por nós (uma pergunta vale 2,5 pontos por forma), e os pesos foram escolhidos nele, então os números são otimistas. Nas perguntas leigas do edital, sem IA, a busca nova fica abaixo da antiga (71,4% contra 92,9%, por ruído da palavra "projeto"); com IA, 85,7%. Sinônimo sem nenhuma palavra em comum ("farda" para uniforme) depende da IA. Sem o edital na base (só os 3 documentos de exemplo, 26 perguntas), as duas buscas acertam 96,2% sem IA; a nova só melhora a ordem (MRR 0,741 → 0,818). O ganho grande aparece quando um documento grande divide a base com os pequenos, que foi o caso do teste em dupla. O edital não vai no repositório: para repetir a medição completa, passe `--edital` (ou `EVAL_EDITAL_PATH`) com o PDF dele.

## Como conferir

```bash
cd src
python test.py --all                       # todas as suítes
python test.py --module equipe             # conversa, presença, contas
python eval/eval_versoes.py                # critérios 1 a 3 e o caso de outro tema
python eval/eval_busca.py --llm            # paráfrases (precisa do texto do edital: --edital)
```

Em dupla, siga o roteiro de demonstração de [VERSIONAMENTO.md](VERSIONAMENTO.md).

## Limites

- O tempo real é por consulta a cada 3 s, não por envio do servidor: o aviso pode levar até 3 s.
- "Quem está aqui" considera fora quem não dá sinal há 20 s.
- As senhas ficam em texto no `.env` de quem hospeda. Serve para uma equipe pequena numa rede de confiança, não para expor na internet.
- Os limiares do filtro de tema foram ajustados em poucos documentos; a folga é estreita (0,862 a 0,881).
- O plano gratuito do Groq limita o uso por minuto; com várias pessoas, a reserva de modelos ajuda, mas pode faltar cota.
