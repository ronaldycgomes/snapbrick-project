# Projeto Técnico: SnapBrick

## 1. Visão Geral do Projeto
O **SnapBrick** é um sistema completo de Visão Computacional e Recomendação projetado para identificar peças de LEGO espalhadas sobre uma superfície a partir de uma única fotografia e sugerir modelos montáveis (*Sets* oficiais e *MOCs - My Own Creations*). O projeto adota padrões de arquitetura de cloud e repositório monolítico (Monorepo) alinhados às práticas de engenharia de grandes empresas de tecnologia (Big Tech).

---

## 2. Alocação de Hardware e Tecnologias por Etapa

| Etapa | Ferramentas & Tecnologias | Hardware Primário | Justificativa de Engenharia |
| :--- | :--- | :--- | :--- |
| **1. Dados Sintéticos** | Python 3.11+, Blender API (`bpy`), LDraw | **ROG Strix (RTX 5070)** | Renderização física (Cycles + OptiX) superior em GPUs NVIDIA. MVP restrito a 15-30 peças base. |
| **2. Treino & CV** | PyTorch, YOLOv11, Albumentations, OpenCV | **ROG Strix (RTX 5070)** | Treinamento com alta demanda de Tensor Cores, VRAM e cuDNN. |
| **3. MLOps & Otimização**| MLflow, ONNX, TensorRT | **ROG Strix + MacBook M5** | Otimização/quantização para CUDA no Strix; testes locais no Apple Silicon. |
| **4. Engine de Matching** | PostgreSQL, SQLAlchemy, Python | **ROG Strix / MacBook M5** | Modelagem relacional e queries analíticas em memória via índices invertidos (CSP). |
| **5. Backend & Cloud** | FastAPI, Docker, preparo AWS (EC2/RDS) | **MacBook Pro M5 / Strix** | Endpoints assíncronos desenhados para resiliência e implantações diretas em ambientes de cloud. |
| **6. Mobile Client** | React Native (Expo), TypeScript | **MacBook Pro M5** | Compilação nativa no ecossistema iOS (XCode) e testes fluidos no S26 Ultra via Expo Go. |

---

## 3. Estrutura do Repositório (Monorepo)

```text
snapbrick/
├── .github/                  # Workflows do GitHub Actions para CI/CD
├── data-pipeline/            # Scripts BlenderProc, catálogo LDraw (MVP: 15-30 peças)
├── ml-core/                  # Treinamento YOLOv11, algoritmos de cor CIE L*a*b*
├── backend/                  # API FastAPI, motor PostgreSQL, scripts ETL
├── mobile/                   # App React Native (TypeScript)
├── .gitignore                # Regras globais (node_modules, venvs, pesos .pt)
└── README.md                 # Documentação de arquitetura e instruções de setup
```

---

## 4. Matriz de Aprendizado por Etapa de Construção

### Etapa 1: Engenharia de Dados Sintéticos
* **Visão Computacional:** *Domain Randomization* (injeção de ruído, texturas, luz) para transpor o *Sim-to-Real gap*.
* **Engenharia:** Scripting headless no Blender (`bpy`) e automação I/O.

### Etapa 2: Visão Computacional
* **Visão Computacional:** Detectores Anchor-free (YOLOv11), métricas IoU/mAP e calibração de cor invariante (CIE L*a*b* e Delta E).
* **Engenharia:** Mixed Precision Training, otimização de alocação de VRAM e data augmentation.

### Etapa 3: MLOps
* **MLOps:** Rastreabilidade com MLflow e Quantização Pós-Treino (PTQ).

### Etapa 4: Engine de Matching (CSP)
* **Ciência da Computação:** *Constraint Satisfaction Problems (CSP)* e algoritmos de cobertura de subconjuntos.
* **Engenharia de Dados:** Aplicação de Índices Invertidos e modelagem relacional normalizada.

### Etapa 5: Backend & APIs
* **Engenharia de Software:** Arquitetura assíncrona (`asyncio`), isolamento de inferência (GIL) e preparação de microsserviços para cloud.

### Etapa 6: Mobile Client
* **Engenharia Frontend:** Manipulação de buffers de câmera em TypeScript, estado global e overlays vetoriais dinâmicos em tempo real.

---

## 5. Formulação Matemática do Matching Engine

Seja $I_{det}$ o inventário de peças detectadas representado por um multiconjunto:

$$I_{det} = \{ (p, c) \mapsto n_{det}(p, c) \}$$

Onde $p$ é o *Part ID*, $c$ o *Color ID* e $n_{det}(p, c)$ a contagem disponível.
Para cada conjunto $S$ no catálogo, o requisito é $I_{req}(S)$ com contagens $n_{req}(p, c, S)$.

A taxa de cobertura $\mathcal{C}(S)$ é:

$$\mathcal{C}(S) = rac{\sum_{(p, c) \in I_{req}(S)} \min(n_{det}(p, c), n_{req}(p, c, S))}{\sum_{(p, c) \in I_{req}(S)} n_{req}(p, c, S)}$$

---

## 6. Cronograma de Sprints (10 Semanas)

* **Sprint 1 (Semanas 1-2):** Fundação de Dados e Ambiente 3D (Pipeline Blender/LDraw).
* **Sprint 2 (Semanas 3-4):** Modelo de Visão Computacional (YOLOv11 + Espaço LAB).
* **Sprint 3 (Semanas 5-6):** Engine de Matching e Banco de Dados (PostgreSQL + CSP).
* **Sprint 4 (Semanas 7-8):** Backend API e Infraestrutura (FastAPI + Deploy em Cloud).
* **Sprint 5 (Semanas 9-10):** Mobile Client (React Native + TypeScript).

---

## 7. Guia de Setup & Workflow Multi-Máquinas

O ecossistema do SnapBrick foi arquitetado para desenvolvimento distribuído entre duas estações de trabalho:

```
┌──────────────────────────────────────┐       ┌──────────────────────────────────────┐
│        MacBook Pro (M-Series)        │       │       ROG Strix (RTX 5070 GPU)       │
│                                      │       │                                      │
│  📱 Mobile (Expo / TypeScript)       │  Git  │  🎨 Data Pipeline (Blender / LDraw)  │
│  ⚙️ Backend API (FastAPI / Docker)   │ ◄───► │  🧠 ML-Core (YOLOv11 / CUDA / ONNX)  │
│  🗄️ PostgreSQL (Docker Compose)      │       │  ⚡ MLOps & TensorRT Export          │
└──────────────────────────────────────┘       └──────────────────────────────────────┘
```

### 🚀 Onboarding no ROG Strix (Workstation de IA)

Ao abrir o repositório no PC **ROG Strix**, siga os passos abaixo para preparar o ambiente:

1. **Clonar o Repositório:**
   ```bash
   git clone https://github.com/ronaldycgomes/snapbrick-project.git
   cd snapbrick-project
   ```

2. **Configurar Variáveis de Ambiente:**
   ```bash
   cp .env.example .env
   ```

3. **Subir Infraestrutura de Banco (Docker):**
   ```bash
   docker compose up -d postgres
   ```

4. **Ambiente Python & CUDA (Data Pipeline & ML com `uv`):**
   * Certifique-se de ter o **Python 3.11+**, **CUDA Toolkit 12.x** e **NVIDIA Drivers** atualizados.
   * Instale o `uv` (se ainda não possuir) e crie o ambiente virtual:
     ```bash
     # Instalação do uv (Linux / WSL)
     curl -LsSf https://astral.sh/uv/install.sh | sh
     source $HOME/.local/bin/env

     # Configurar ambiente virtual no data-pipeline
     cd data-pipeline
     uv venv
     source .venv/bin/activate
     ```

5. **Progresso Realizado (Sprint 1 - Data Pipeline & 3D Rendering):**
   * ✅ **Download do Catálogo LDraw:** Script de ingestão automatizada ([`download_ldraw.sh`](data-pipeline/download_ldraw.sh)) e biblioteca de peças/primitivas LDraw estruturada em `data-pipeline/ldraw_lib/`.
   * ✅ **Script de Diagnóstico de Hardware:** Script ([`data-pipeline/src/check_gpu.py`](data-pipeline/src/check_gpu.py)) para inspeção e benchmark de aceleração CUDA/OptiX no WSL 2 com a NVIDIA RTX 5070.
   * ✅ **Pipeline de Renderização Headless:** Script ([`data-pipeline/src/render_single_part.py`](data-pipeline/src/render_single_part.py)) implementado com:
     - Parser recursivo de geometria LDraw (`.dat`) com resolução automática de sub-peças e primitivas.
     - Câmera inteligente com enquadramento automático (*Auto-Framing*) via Bounding Box em 640x640.
     - Iluminação de estúdio 3-pontos (*Key, Fill, Rim*) e Shaders PBR de plástico ABS LEGO.
     - Suporte a seleção de cores via CLI (`--color red`, `blue`, `yellow`, `green`, `#HEX`).
     - Aceleração por GPU com Ray Tracing no **Blender Cycles (CUDA)** executando a **< 1 segundo por imagem** na **RTX 5070**.
     - Validação de renderização de teste em [`data-pipeline/output/test_3001.png`](data-pipeline/output/test_3001.png).

    * ✅ **Pipeline de Domain Randomization & Gerador de Dataset Sintético:** Script ([`data-pipeline/src/generate_dataset.py`](data-pipeline/src/generate_dataset.py)) implementado com:
      - Sorteio e dispersão procedural de 32 classes (peças clássicas, acabamento e Technic) com algoritmo anti-sobreposição.
      - Shaders de fundo procedurais com variação de superfícies (*Wood Grain, Granite/Stone, Fabric/Carpet, Matte/Painted*).
      - Iluminação de cena dinâmica com randomização de posições, potência e temperaturas de cor (2800K a 6500K).
      - Câmera móvel simulando fotografia de smartphone (distância focal 35-70mm, elevação 35°-85°, rastreamento de enquadramento).
      - Cálculo e projeção matemática 3D para 2D com anotações automáticas de Bounding Boxes no formato YOLOv11 (`class_id x_center y_center width height`).
      - Estruturação automática em splits `images/train` (1.000 imgs), `images/val` (150 imgs), `images/test` (50 imgs) e manifesto `dataset.yaml`.
    * ✅ **Visualizador de Anotações:** Script ([`data-pipeline/src/visualize_labels.py`](data-pipeline/src/visualize_labels.py)) com suporte a OpenCV, Pillow e fallback vetorial puro em SVG/HTML.

6. **Progresso Realizado (Sprint 2 - Visão Computacional & YOLOv11 Medium):**
   * ✅ **Pipeline de Treinamento Estável e Acelerado por GPU (CUDA 13.0 / RTX 5070):** Script ([`ml-core/src/train.py`](ml-core/src/train.py)) com:
     - Auditoria automática de hardware (RTX 5070 Laptop GPU, 8GB VRAM, cuDNN 9.24).
     - Blindagem completa para WSL2: caching pré-processado em SSD NVMe (`cache='disk'`), estratégia IPC `file_system` no PyTorch e alocação de memória controlada (`batch=8`, ~5.2 GB VRAM).
     - Fine-tuning completo de 150 épocas do **YOLOv11 Medium (`yolo11m.pt`)** com Mixed Precision (`fp16`) e augmentations (*Mosaic*, *MixUp*, *HSV Jitter*).
     - Resultados consolidados no conjunto de validação com 50 classes: **$Precision = 80.78\%$**, **$Recall = 80.74\%$**, **$mAP_{50} = 75.43\%$** e **$mAP_{50-95} = 68.91\%$**.
     - Pesos otimizados salvos em [`ml-core/weights/best.pt`](ml-core/weights/best.pt).
   * ✅ **Detector de Alta Resolução & Classificador de Cores Invariante CIE L\*a\*b\*:** Script ([`ml-core/src/detect.py`](ml-core/src/detect.py)) com:
     - Fatiamento multi-quadrante de alta resolução (*Sliced SAHI*) para detecção precisa de peças pequenas.
     - Recorte automático de ROI das peças detectadas e amostragem no espaço CIE L\*a\*b\* com distância fotométrica $\Delta E$.
     - Exportação de imagem anotada e manifesto de inventário estruturado em JSON.
   * ✅ **Exportador ONNX:** Script ([`ml-core/src/export_onnx.py`](ml-core/src/export_onnx.py)) para portabilidade do modelo.

7. **Diagnóstico de Performance & Gap Sim-to-Real (Oportunidades de Melhoria):**
   Durante os testes com fotos reais capturadas por smartphone (`IMG_0032.jpg`, `IMG_0033.jpg`), foi identificado o desafio clássico de *Sim-to-Real Domain Gap* (discrepância entre os renders 3D do simulador e a fotografia física):
   * **Erros Mapeados em Fotos Reais:**
     1. *Confusão de Ângulo:* Peças inclinadas (*Slope 45 2x2 - 3039*) sendo identificadas como *Brick 2x2 (3003)* devido à predominância de ângulos de câmera elevados no gerador sintético.
     2. *Reflexos Especulares:* Peças lisas (*Tile 1x2 - 3069b*) com faixas brilhantes de luz física sendo classificadas como peças com ranhuras (*Tile 1x2 Grille - 30039*).
     3. *Escala em Peças Similares:* Pinos Technic de comprimentos próximos (*Technic Pin 2780* vs *Technic Pin 3L 6558*) apresentando confusão em tomadas abertas.
     4. *Interferência de Sombras na Cor:* Amostragem fotométrica influenciada por sombras de contato na mesa.

   * **Estratégias Planejadas para Refinamento do Modelo (Próximas Iterações):**
     * 🎯 **1. Injeção de Dados Reais (Few-Shot Real-World Fine-Tuning):** Coleta e anotação manual de um lote pequeno de fotos reais de smartphone (20 a 50 fotos) mescladas ao dataset sintético (proporção 90/10), ensinando a rede a reconhecer a assinatura visual do plástico real.
     * 📐 **2. Domain Randomization com Câmeras Rasantes no Blender:** Ajuste no [`generate_dataset.py`](data-pipeline/src/generate_dataset.py) para incluir ângulos de elevação baixos (25° a 50°), forçando a visibilidade da rampa das Slopes e perfis dos pinos.
     * 💡 **3. Data Augmentation Fotométrico Anti-Reflexo:** Adição de ruídos de reflexos especulares e variações de iluminação dura no pipeline de augmentations para desacoplar reflexos de texturas reais.
     * 🎨 **4. Segmentação por Máscara para Classificação de Cor:** Substituição da amostragem retangular por segmentação Otsu/K-Means para isolar exclusivamente os pixels da peça, eliminando o fundo e sombras no cálculo $\Delta E$.

8. **Como Executar os Módulos de ML:**
   ```bash
   # Ativar ambiente virtual
   source .venv/bin/activate

   # Treinar o modelo YOLOv11 Medium na GPU (otimizado para estabilidade no WSL2 e 8GB VRAM)
   python ml-core/src/train.py --model yolo11m.pt --epochs 150 --batch 8 --workers 4 --cache disk

   # Executar detecção e classificação de cor em uma foto
   python ml-core/src/detect.py --image ml-core/dataset/images/IMG_0033.jpg --conf 0.35

   # Exportar modelo treinado para ONNX
   python ml-core/src/export_onnx.py
   ```

9. **Próximo Passo (Sprint 3 - Backend API & Matching Engine):**
   - Configurar API FastAPI assíncrona (`backend/src/main.py`) e rotas REST.
   - Modelar schemas do PostgreSQL com SQLAlchemy/Alembic (tabelas `models` e `model_inventory`).
   - Implementar algoritmo CSP de Matching de peças e cálculo da taxa de cobertura $\mathcal{C}(S)$.

