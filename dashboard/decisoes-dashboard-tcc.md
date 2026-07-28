# Documento de Decisões Técnicas — Dashboard

**Projeto:** Sistema Inteligente de Apoio à Detecção de Afogamentos em Piscinas Baseado em Aprendizado de Máquina e Câmeras IP
**Autor:** Pedro Soares
**Orientador:** Prof. Willian Bogler
**Data:** 22/07/2026
**Status do documento:** rascunho para validação com o orientador

## 1. Objetivo do módulo

Construir uma interface web que cumpra três funções:

1. **Monitoramento ao vivo** — exibir o vídeo da câmera IP com as detecções sobrepostas em tempo quase real.
2. **Análise agregada** — gráficos que sintetizem o comportamento do sistema ao longo do tempo.
3. **Auditoria** — histórico navegável de eventos, cada um com snapshot e metadados da detecção.

### 1.1 Justificativa acadêmica

O dashboard não é um acessório estético. Ele é posicionado no trabalho como **interface de apoio à decisão do operador (salva-vidas)**, coerente com o termo "Apoio" já presente no título do TCC.

Além disso, ele passa a ser o **instrumento de coleta e apresentação das evidências das hipóteses**: os mesmos dados que alimentam os gráficos são os dados que sustentam o capítulo de resultados. Isso elimina o retrabalho de gerar gráficos separadamente para a monografia.

> **Pendência:** confirmar o enunciado exato de H1, H2 e H3 e mapear cada gráfico à hipótese correspondente (ver seção 8).

---

## 2. Arquitetura geral

Três componentes, com separação estrita de responsabilidades:

```
┌──────────────────────────────┐
│  Processo de Inferência      │   (o sistema Python que já existe)
│  - captura RTSP threaded     │
│  - inferência YOLO           │
│  - janela deslizante         │
│  - grava evento + snapshot   │
└──────────────┬───────────────┘
               │ escreve
               ▼
   ┌───────────────────────┐        ┌──────────────────┐
   │  SQLite (eventos.db)  │        │ /snapshots/*.jpg │
   └───────────┬───────────┘        └────────┬─────────┘
               │ lê                          │ serve
               ▼                             ▼
        ┌──────────────────────────────────────┐
        │  API (FastAPI)                       │
        │  - /api/stream  (MJPEG)              │
        │  - /api/eventos (REST)               │
        │  - /api/metricas                     │
        │  - /api/eventos/{id}/status          │
        │  - /api/eventos/stream (SSE)         │
        └──────────────────┬───────────────────┘
                           ▼
                  ┌────────────────┐
                  │  Frontend      │
                  └────────────────┘
```

### D-01 — O dashboard NÃO executa inferência própria

**Decisão:** o frontend e a API consomem exclusivamente o resultado produzido pelo processo de inferência já existente. Nenhuma cópia do modelo é carregada pela camada web.

**Motivo:** rodar um segundo pipeline de inferência duplicaria o consumo de CPU e **invalidaria as medições de FPS e latência** usadas na validação das hipóteses. A medição de desempenho precisa refletir o sistema em condição real de operação.

**Implementação:** o loop de inferência mantém em memória o último frame anotado (buffer de um único slot, protegido por lock) e o expõe ao processo web. Se API e inferência rodarem em processos separados, o frame compartilhado vai para memória compartilhada ou para um arquivo temporário sobrescrito a cada iteração.

---

## 3. Transmissão ao vivo

### D-02 — Streaming via MJPEG sobre HTTP

**Decisão:** o vídeo ao vivo é entregue como MJPEG, através de um endpoint FastAPI usando `StreamingResponse` com `multipart/x-mixed-replace; boundary=frame`. No frontend, consumido por uma única tag `<img src="/api/stream">`.

**Motivo:** navegadores **não reproduzem RTSP nativamente**. As alternativas foram avaliadas:

| Alternativa | Latência | Complexidade | Decisão |
|---|---|---|---|
| **MJPEG** | ~0,3–1 s | Muito baixa — nenhuma dependência extra | **Adotada** |
| HLS (via ffmpeg) | 6–20 s | Média — exige ffmpeg e segmentação | Rejeitada: latência incompatível com alerta de afogamento |
| WebRTC (MediaMTX / go2rtc) | < 0,5 s | Alta — servidor de mídia adicional, ICE/STUN | Rejeitada: complexidade não justificada no escopo |
| WebSocket + frames base64 | ~0,5 s | Média — overhead de codificação de ~33% | Rejeitada: MJPEG entrega o mesmo resultado com menos código |

**Parâmetros definidos:**

- Taxa de entrega ao navegador: **10–15 fps** (desacoplada da taxa de inferência).
- Qualidade JPEG: **70** — equilíbrio entre nitidez das bounding boxes e banda.
- Resolução de saída: **720p**, redimensionada a partir do frame original.
- O frame entregue é **sempre o já anotado** (bounding boxes, rótulo da classe, confiança). A anotação é feita uma única vez, no processo de inferência.

**Limitação conhecida a documentar no TCC:** MJPEG não usa compressão temporal, portanto consome mais banda que H.264. Aceitável porque o dashboard é projetado para operação em **rede local**, no mesmo ambiente da piscina.

---

## 4. Persistência

### D-03 — SQLite como banco de eventos, mantendo o CSV para métricas brutas

**Decisão:** adotar SQLite (`eventos.db`) como fonte de dados do dashboard. O log CSV existente é **mantido** para o registro bruto por frame, usado na validação offline.

**Motivo:** o CSV é adequado para escrita sequencial e análise em lote, mas não suporta bem filtragem, paginação e agregação — exatamente o que histórico e gráficos exigem. O SQLite resolve isso sendo ainda um arquivo único, sem servidor, portável e anexável ao trabalho. Exportação para CSV continua trivial via `SELECT`.

**Divisão de responsabilidade:**

- **CSV** — uma linha por frame processado. Dados brutos de desempenho.
- **SQLite** — uma linha por **evento** (alerta disparado pela janela deslizante). Dados de negócio.

### D-04 — Esquema da tabela `eventos`

```sql
CREATE TABLE eventos (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp_inicio    TEXT    NOT NULL,   -- ISO 8601, primeiro frame da janela
    timestamp_alerta    TEXT    NOT NULL,   -- ISO 8601, momento do disparo
    camera_id           TEXT    NOT NULL,
    classe              TEXT    NOT NULL,   -- afogamento | adulto_ok | crianca_ok | piscina
    confianca_media     REAL    NOT NULL,   -- média na janela deslizante
    confianca_maxima    REAL    NOT NULL,
    frames_positivos    INTEGER NOT NULL,   -- nº de frames positivos na janela
    frames_janela       INTEGER NOT NULL,   -- tamanho total da janela
    latencia_ms         INTEGER NOT NULL,   -- timestamp_alerta - timestamp_inicio
    modelo              TEXT    NOT NULL,   -- yolo11n | yolo26n
    caminho_snapshot    TEXT    NOT NULL,
    status              TEXT    DEFAULT 'pendente',  -- pendente | confirmado | falso_positivo
    revisado_em         TEXT,
    observacao          TEXT
);

CREATE INDEX idx_eventos_timestamp ON eventos(timestamp_alerta);
CREATE INDEX idx_eventos_status    ON eventos(status);
```

**Notas de projeto:**

- O campo `modelo` permite comparar YOLO11n e YOLO26n **dentro do mesmo banco**, sem separar execuções em arquivos distintos.
- `frames_positivos` e `frames_janela` são gravados separadamente (em vez de apenas a razão) para permitir recalcular o threshold em análise posterior sem reprocessar vídeo.
- `latencia_ms` é calculado e persistido no momento da escrita, não derivado em consulta — evita divergência entre o gráfico e o número citado no texto.

### D-05 — Snapshots em disco, não no banco

**Decisão:** o frame anotado do momento do alerta é salvo como JPEG em `snapshots/{ano}/{mes}/{id}.jpg`. O banco guarda apenas o caminho relativo.

**Motivo:** BLOBs inflariam o arquivo do banco e dificultariam a inspeção manual das imagens durante a análise dos resultados. Arquivos soltos são inspecionáveis com qualquer visualizador e servidos diretamente pelo FastAPI via `StaticFiles`.

**Parâmetros:** qualidade JPEG 85 (maior que a do stream — o snapshot é evidência), resolução original preservada, particionamento por ano/mês para evitar diretórios com milhares de arquivos.

---

## 5. Confirmação pelo operador

### D-06 — Cada evento do histórico recebe classificação manual

**Decisão:** todo evento no histórico exibe dois controles: **"Ocorrência real"** e **"Falso positivo"**, opcionalmente acompanhados de um campo de observação livre. A ação grava `status` e `revisado_em`.

**Motivo:** esta é a decisão de maior retorno acadêmico do módulo. Ela transforma o dashboard de visualizador passivo em **instrumento de coleta de ground truth em operação**, e permite calcular diretamente do banco:

- Precisão = confirmados / (confirmados + falsos positivos)
- Taxa de falsos positivos por hora de operação
- Distribuição de confiança dos falsos positivos versus verdadeiros positivos

Sem esse campo, os números de precisão viriam apenas do conjunto de teste rotulado — evidência mais fraca, porque não reflete condições reais de operação. Com ele, o TCC ganha uma seção de **validação em campo**.

**Observação metodológica:** a revisão manual introduz subjetividade do avaliador. Registrar no texto que a rotulagem foi feita pelo próprio autor e descrever o critério adotado.

> **Pendência:** recall exige conhecer os afogamentos que o sistema **não** detectou. Definir como isso será apurado — provavelmente por revisão do vídeo gravado durante as sessões com voluntários, após aprovação do CEP.

---

## 6. Comunicação em tempo real

### D-07 — Server-Sent Events (SSE) para novos alertas

**Decisão:** quando um novo evento é registrado, a API o notifica ao frontend via SSE (`text/event-stream`), no endpoint `/api/eventos/stream`.

**Motivo:** o fluxo é unidirecional (servidor → cliente). WebSocket seria bidirecional e traria gerenciamento de conexão desnecessário. SSE é HTTP puro, reconecta automaticamente no navegador (`EventSource`) e é suportado nativamente pelo FastAPI.

**Efeito na interface:** o alerta aparece na tela sem recarregar a página, dispara sinal sonoro e destaque visual, e insere o evento no topo do histórico.

---

## 7. Estrutura de telas

Três telas, navegação por abas, layout único e responsivo.

### 7.1 Ao Vivo

- Player MJPEG ocupando a área principal.
- Faixa de status: câmera conectada/desconectada, FPS atual, modelo em execução, tempo de atividade.
- Painel lateral com os últimos 5 eventos (atualizado por SSE).
- Banner de alerta em destaque quando `classe = afogamento`, com áudio.

### 7.2 Análise

Grade de gráficos (ver seção 8), com filtro global de período (hoje / 7 dias / 30 dias / intervalo personalizado) e filtro por modelo.

### 7.3 Histórico

- Tabela paginada, mais recentes primeiro.
- Colunas: miniatura do snapshot, data/hora, classe, confiança média, latência, modelo, status.
- Filtros: período, classe, status, modelo.
- Clique na linha abre painel lateral com o snapshot em tamanho real, todos os metadados e os controles de confirmação (D-06).
- Botão de exportação para CSV do resultado filtrado.

---

## 8. Gráficos definidos

| # | Gráfico | Tipo | Hipótese relacionada |
|---|---|---|---|
| G1 | Latência de detecção (do início do evento ao alerta) | Histograma + linha da média | H1 — viabilidade de detecção em tempo hábil |
| G2 | Distribuição de confiança das detecções | Histograma, separado por verdadeiro/falso positivo | H1 — justificativa do threshold adotado |
| G3 | FPS médio ao longo do tempo, YOLO11n vs YOLO26n | Linha, duas séries | H2 — ganho de desempenho do modelo NMS-free |
| G4 | Tempo de inferência por frame, comparativo entre modelos | Boxplot ou barras com desvio padrão | H2 |
| G5 | Eventos por hora do dia | Barras | H3 — caracterização do uso e da carga operacional |
| G6 | Precisão e taxa de falsos positivos ao longo do tempo | Linha | H3 — estabilidade do sistema em operação |

**Decisão de projeto:** todo gráfico exibido na tela deve ser **exportável em PNG e em CSV**, para ser levado direto à monografia. Isso evita gerar as figuras duas vezes.

> **Pendência:** revalidar o mapeamento gráfico ↔ hipótese após confirmar o enunciado de H1/H2/H3.

---

## 9. Stack do frontend

### D-08 — Decisão pendente entre duas opções

Ambas atendem ao escopo. A escolha depende do peso dado ao alinhamento com a ementa do curso.

**Opção A — Angular**

- Alinhada à formação acadêmica (Java/Spring Boot/Angular) e provavelmente ao que o orientador espera ver.
- Domínio prévio do autor.
- Custo: build step, servidor de desenvolvimento separado, configuração de CORS ou proxy, mais arquivos para manter.

**Opção B — SPA única em HTML + Chart.js, servida pelo próprio FastAPI**

- Sem build, sem CORS, sem deploy separado — `app.mount("/", StaticFiles(...))` e acabou.
- Um único artefato executável, mais simples de demonstrar na banca (um comando sobe tudo).
- Custo: menos estruturado; se o escopo crescer, o arquivo tende a ficar extenso.

**Recomendação:** Opção B. O escopo é fechado (três telas, seis gráficos) e o esforço economizado é melhor investido no modelo, que é o núcleo do trabalho. A demonstração na banca também fica mais robusta com um único processo.

**Bibliotecas previstas na Opção B:** Chart.js para os gráficos; nenhuma outra dependência de runtime.

---

## 10. Privacidade e conformidade ética

O projeto tramita em comitê de ética (CEP), com coleta de dados de voluntários. As decisões abaixo devem estar implementadas **antes** do início da coleta.

### D-09 — Retenção limitada de snapshots

Snapshots são excluídos automaticamente após período definido, por rotina agendada. O registro no banco é preservado (metadados numéricos), apenas a imagem é removida e `caminho_snapshot` marcado como expirado.

> **Pendência:** definir o prazo de retenção em conformidade com o que foi declarado no protocolo submetido ao CEP.

### D-10 — Anonimização facial nos snapshots

Aplicar borrão (blur gaussiano) nas regiões de rosto antes de gravar o snapshot em disco. O frame do stream ao vivo não é anonimizado, pois não é persistido e destina-se ao operador presente no local.

**Motivo:** o snapshot é o único artefato persistido com imagem identificável de voluntário. Implementar antes da coleta evita ter que descartar dados ou refazer sessões.

### D-11 — Acesso restrito

O dashboard opera em rede local, sem exposição à internet. Autenticação simples por usuário e senha na camada da API.

---

## 11. Endpoints previstos

| Método | Rota | Função |
|---|---|---|
| GET | `/api/stream` | Stream MJPEG do frame anotado |
| GET | `/api/status` | Estado da câmera, FPS, modelo ativo, uptime |
| GET | `/api/eventos` | Lista paginada, com filtros de período, classe, status e modelo |
| GET | `/api/eventos/{id}` | Detalhe de um evento |
| PATCH | `/api/eventos/{id}/status` | Confirmação do operador (D-06) |
| GET | `/api/eventos/stream` | SSE de novos alertas |
| GET | `/api/metricas` | Séries agregadas para os gráficos G1–G6 |
| GET | `/api/exportar` | Exportação CSV do conjunto filtrado |
| GET | `/snapshots/...` | Arquivos estáticos de snapshot |

---

## 12. Fora de escopo

Registrado explicitamente para delimitar o trabalho na defesa:

- Múltiplas câmeras simultâneas (o esquema já prevê `camera_id`, mas a interface trata uma câmera).
- Aplicativo móvel nativo.
- Notificação por SMS, e-mail ou push.
- Gestão de usuários e perfis de permissão.
- Reprodução de vídeo gravado — apenas snapshots estáticos são persistidos.
- Reprocessamento de vídeos históricos pela interface.

---

## 13. Riscos identificados

| Risco | Impacto | Mitigação |
|---|---|---|
| Concorrência de escrita no SQLite entre inferência e API | Bloqueio de escrita | Habilitar modo WAL; API acessa em leitura apenas, exceto pelo `PATCH` de status |
| Overhead do stream MJPEG afetando o FPS de inferência | Compromete medição de H2 | Codificação JPEG em thread separada; queda de frames em vez de bloqueio do loop principal |
| Crescimento do diretório de snapshots | Consumo de disco | Rotina de retenção (D-09) e particionamento por ano/mês |
| Perda de conexão RTSP | Tela ao vivo congelada sem aviso | Timestamp do último frame no `/api/status`; indicador visual de desconexão após N segundos |

---

## 14. Pendências consolidadas

1. Confirmar o enunciado exato de H1, H2 e H3 e revalidar o mapeamento da seção 8.
2. Definir prazo de retenção de snapshots conforme o protocolo do CEP (D-09).
3. Decidir entre Angular e SPA + Chart.js (D-08).
4. Definir o método de apuração de recall / falsos negativos (seção 5).
5. Validar este documento com o orientador antes de iniciar a implementação.
