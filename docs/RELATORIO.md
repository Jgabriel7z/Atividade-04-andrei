# Atividade 04 — Manutenção em projeto existente: EixoAI

Equipe: Luiz Alberto Rezende Costa, João Gabriel Nascimento Rodrigues
Disciplina de Agentes, Prof. Andrei Formiga

## 1. O que foi feito

Executamos o plano da Atividade 03 no [EixoAI](https://github.com/kaio-w-b/EixoAi), um sistema de perguntas e respostas sobre documentos jurídicos com RAG (Gradio, ChromaDB, multilingual-e5-small, Groq). A melhoria é o **histórico de versões de documentos com análise de impacto**: a base guarda vários documentos e várias versões de cada um; cada versão registra autor, data, motivo e SHA-256; a busca do chat só vê a versão vigente; a comparação mostra os artigos alterados, removidos e incluídos; a LLM resume a mudança; e o sistema avisa quais trechos de outros documentos dependem dos artigos que mudaram.

O detalhe técnico, o build e os testes estão em [`VERSIONAMENTO.md`](VERSIONAMENTO.md). A comparação com o código original está em [`ANTES_E_DEPOIS.md`](ANTES_E_DEPOIS.md).

## 2. Resultado contra os critérios da Atividade 03

| Critério | Meta | Resultado |
|---|---|---|
| Artigos alterados detectados pela comparação | 100% | 100% (5/5) |
| Citações diretas encontradas pela análise de impacto | 100% | 100% (10/10), nenhuma falsa |
| Precisão das sugestões semânticas | ≥ 70% | 100% (4/4); 50% (5/10) na primeira tentativa |
| Hit Rate das 48 perguntas da CF/88 | não cair | 83,3% antes e depois, igual pergunta a pergunta |

As ressalvas estão em [`VERSIONAMENTO.md`](VERSIONAMENTO.md). As principais: o conjunto de teste é pequeno e feito por nós, e a regra das sugestões foi ajustada nele.

## 3. Processo com o agente

Usamos o Claude Code (modelo Claude Opus) no repositório clonado, numa branch própria. O push para o repositório original foi desligado no `git remote` antes de qualquer mudança, para não haver risco de enviar código ao autor.

**Como o trabalho andou:**

1. **Leitura antes de escrever.** O agente leu os PDFs das Atividades 03 e 04 e os módulos que a melhoria tocava (`app.py`, `retriever.py`, `llm_chain.py`, a suíte de testes e o avaliador) antes de mudar qualquer linha.
2. **Achados no código existente que mudaram o desenho:**
   - os IDs dos chunks (`{doc}_{n}`) colidiriam entre versões do mesmo documento;
   - a divisão por artigo do chunker também corta em citações com "Art." maiúsculo no meio da frase, então a comparação entre versões não podia reaproveitá-la como o plano dizia. Usamos "Art." no começo da linha (convenção da LC 95/98);
   - a detecção de "texto com artigos" ignorava maiúsculas e minúsculas: um manual que só cita "art. 12" virava um chunk gigante;
   - o avaliador buscava na coleção inteira, o que só funcionava porque a base tinha um documento;
   - `python test.py --sample` sem `EVAL_PDF_PATH` quebra, porque passa `evaluate_answer=` para uma função cujo parâmetro se chama `evaluate_answer_quality`. Não corrigimos, por estar fora do escopo.
3. **Ajustes de plano durante a execução:**
   - com documentos curtos, cada um vira um ou dois chunks de ~1000 caracteres, e a sugestão no nível do chunk não dizia onde olhar. A sugestão passou a ser a **frase** mais parecida dentro dos chunks candidatos que o ChromaDB devolve;
   - o plano previa "busca semântica com limiar mínimo". Medido, deu 50% de precisão em qualquer limiar, abaixo da meta de 70%, porque o e5 põe quase tudo entre 0,87 e 0,90. Um protótipo mostrou que comparar cada frase com **todos** os artigos e ficar com o mais parecido separava bem; com essa regra, a precisão foi a 100% (4/4).
4. **Teste na interface real:** o agente subiu o original e a versão nova em portas locais e usou o navegador (Playwright) para tirar as capturas do "antes e depois". Isso revelou um problema que os testes automáticos não pegavam: no Gradio 6.3, um componente de aba oculta que recebe atualização fica travado com o "carregando" por cima. Foi resolvido atualizando cada aba ao abri-la.
5. **Otimização guiada por medição:** a primeira medição de latência (96 → 202 ms) era ruído da máquina carregada. Um benchmark isolado mostrou que o filtro custa ~4 ms, e o profiling mostrou onde estava o tempo de verdade: a publicação de versão reinseria todos os chunks no ChromaDB. Com a publicação incremental, uma revisão da CF/88 caiu de 11,5 s para 1,1 s, e a análise de impacto de 5,7 s para 2,6 s.
6. **Revisão independente:** um segundo agente, sem acesso às nossas conclusões, revisou o código e achou 14 pontos. Entre eles, artigos no estilo da CLT ("Art. 5º - A…") viravam "5-A", publicações simultâneas dessincronizavam o SQLite e o ChromaDB, e texto do PDF ia cru para o Markdown da tela. Os pontos foram corrigidos e ganharam testes.
7. **Medição antes e depois:** o Hit Rate da CF/88 foi medido no commit original e na branch nova, com o mesmo PDF e bancos temporários separados.

**O que funcionou:**
- [preencher pela equipe]

**O que não funcionou ou deu trabalho:**
- A instalação das dependências (torch, ChromaDB, Gradio) foi a parte mais lenta: o pip deu timeout de rede na primeira tentativa.
- [preencher pela equipe]

**O que foi inesperado:**
- O limiar de semelhança precisou ser alto (0,86): o multilingual-e5 dá similaridade de 0,75 a 0,80 até entre trechos jurídicos sem relação.
- [preencher pela equipe]

## 4. Familiaridade prévia da equipe

[preencher pela equipe: Python, RAG, ChromaDB, embeddings, Gradio, LLM via API, testes]

## 5. O que aprendemos

[preencher pela equipe]
