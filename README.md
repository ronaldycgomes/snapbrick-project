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

6. **Progresso Realizado (Sprint 2 - Visão Computacional & YOLOv11):**
   * ✅ **Pipeline de Treinamento Acelerado por GPU (CUDA 13.0):** Script ([`ml-core/src/train.py`](ml-core/src/train.py)) com:
     - Auditoria automática de hardware (RTX 5070 Laptop GPU, 8GB VRAM, cuDNN 9.24).
     - Fine-tuning do **YOLOv11s** com Mixed Precision (`fp16`) e data augmentations (*Mosaic*, *MixUp*, *HSV Jitter*).
     - Treinamento completo de 100 épocas atingindo **$mAP_{50} = 87.03\%$** e **$mAP_{50-95} = 74.74\%$** no conjunto de teste cego.
     - Pesos otimizados salvos em [`ml-core/weights/best.pt`](ml-core/weights/best.pt).
   * ✅ **Detector & Classificador de Cores Invariante CIE L\*a\*b\*:** Script ([`ml-core/src/detect.py`](ml-core/src/detect.py)) com:
     - Recorte automático de ROI das peças detectadas e amostragem de cor central.
     - Cálculo de distância fotométrica $\Delta E$ no espaço CIE L\*a\*b\* para identificação de cores oficiais LEGO com rejeição de sombras/brilho.
     - Exportação de imagem anotada e manifesto de inventário em JSON estruturado.
   * ✅ **Exportador & Otimizador ONNX:** Script ([`ml-core/src/export_onnx.py`](ml-core/src/export_onnx.py)) para conversão de pesos PyTorch em runtime ONNX otimizado com FP16 para inferência ultrarrápida no backend e mobile.

7. **Como Executar os Módulos de ML:**
   ```bash
   # Ativar ambiente virtual
   source .venv/bin/activate

   # Treinar o modelo YOLOv11s na GPU
   python ml-core/src/train.py --model yolo11s.pt --epochs 100 --batch 32

   # Executar detecção e classificação de cor em uma foto
   python ml-core/src/detect.py --image data-pipeline/output/IMG_0032.jpg --conf 0.25

   # Exportar modelo treinado para ONNX
   python ml-core/src/export_onnx.py
   ```

8. **Próximo Passo (Sprint 3 - Backend API & Matching Engine):**
   - Configurar API FastAPI assíncrona (`backend/src/main.py`) e rotas REST.
   - Modelar schemas do PostgreSQL com SQLAlchemy/Alembic (tabelas `models` e `model_inventory`).
   - Implementar algoritmo CSP de Matching de peças e cálculo da taxa de cobertura $\mathcal{C}(S)$.

