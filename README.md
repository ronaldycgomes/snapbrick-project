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

4. **Ambiente Python & CUDA (Data Pipeline & ML):**
   * Certifique-se de ter o **Python 3.11+**, **CUDA Toolkit 12.x** e **NVIDIA Drivers** atualizados.
   * Crie o ambiente virtual para o pipeline de dados / visão computacional:
     ```bash
     python -m venv .venv
     # Windows:
     .venv\Scripts\activate
     # Linux:
     source .venv/bin/activate
     ```

5. **Próximo Objetivo Imediato no Strix (Sprint 1):**
   * Configurar a biblioteca de peças base do LDraw (`data-pipeline/`).
   * Desenvolver os scripts headless em Blender (`bpy` + Cycles/OptiX) para renderização sintética e geração do dataset inicial anotado no padrão YOLO (15 a 30 peças).

