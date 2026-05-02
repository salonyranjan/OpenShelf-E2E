<div align="center">

<img src="https://capsule-render.vercel.app/api?type=venom&color=0:000000,25:7c3aed,55:a855f7,80:06b6d4,100:000000&height=250&section=header&text=📚%20OPENSHELF&fontSize=88&fontColor=ffffff&fontAlignY=52&animation=fadeIn&stroke=7c3aed&strokeWidth=3&desc=Horizon%20%7C%20Cyber-Neon%20AI%20Book%20Recommender%20%7C%20End-to-End%20ML%20Pipeline&descSize=19&descAlignY=74&descColor=a855f7" />

<br/>

<img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&weight=900&size=22&pause=1000&color=a855f7&center=true&vCenter=true&width=900&height=55&lines=📚+Personalised+Book+Recommendations+via+Collaborative+Filtering;🤖+KNN+%2B+CSR+Matrix+%7C+270%2C000%2B+Ratings+Processed;🌌+Live+Cyber-Neon+Pipeline+Monitor+%7C+Real-Time+Bloom;🚀+Streamlit+%2B+Scikit-learn+%2B+Docker+%2B+AWS" alt="Typing SVG" />

<br/><br/>

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://openshelf-e2e.streamlit.app/)
&nbsp;
<img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
<img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
<img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />

<br/>

<img src="https://img.shields.io/badge/AWS_EC2-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white" />
<img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white" />
<img src="https://img.shields.io/badge/KNN_Model-7c3aed?style=for-the-badge" />
<img src="https://img.shields.io/badge/270K%2B_Ratings-a855f7?style=for-the-badge" />
<img src="https://img.shields.io/badge/License-MIT-06b6d4?style=for-the-badge" />

<br/>

<img src="https://img.shields.io/github/stars/salonyranjan/OpenShelf-E2E?style=for-the-badge&color=7c3aed" />
<img src="https://img.shields.io/github/forks/salonyranjan/OpenShelf-E2E?style=for-the-badge&color=a855f7" />
<img src="https://img.shields.io/github/last-commit/salonyranjan/OpenShelf-E2E?style=for-the-badge&color=06b6d4" />

<br/><br/>

> *"A high-performance, end-to-end ML application delivering personalised book recommendations — wrapped in a cinematic Cyber-Neon live pipeline dashboard."*

<br/>

<a href="https://openshelf-e2e.streamlit.app/"><img src="https://img.shields.io/badge/🚀_Launch_App-7c3aed?style=for-the-badge" /></a>
&nbsp;
<a href="#8--getting-started"><img src="https://img.shields.io/badge/📦_Quick_Setup-a855f7?style=for-the-badge" /></a>
&nbsp;
<a href="#5--ml-pipeline"><img src="https://img.shields.io/badge/🤖_ML_Pipeline-06b6d4?style=for-the-badge" /></a>
&nbsp;
<a href="#11--contributing"><img src="https://img.shields.io/badge/🤝_Contribute-10b981?style=for-the-badge" /></a>

</div>

---

## 📋 Table of Contents

1. [🌌 What is OpenShelf Horizon?](#1--what-is-openshelf-horizon)
2. [✨ Key Features](#2--key-features)
3. [🏗️ System Architecture](#3-%EF%B8%8F-system-architecture)
   - 3.1 [🗂️ Project Structure](#31-%EF%B8%8F-project-structure)
   - 3.2 [📐 Architecture Diagram](#32--architecture-diagram)
4. [🛠️ Tech Stack](#4-%EF%B8%8F-tech-stack)
5. [🤖 ML Pipeline](#5--ml-pipeline)
   - 5.1 [📊 Data Validation](#51--data-validation)
   - 5.2 [🔄 Transformation](#52--transformation)
   - 5.3 [🧠 Model Engine](#53--model-engine)
   - 5.4 [⚡ Pipeline Sequence](#54--pipeline-sequence)
6. [🌌 Horizon UI & Monitoring](#6--horizon-ui--monitoring)
7. [⚡ Performance](#7--performance)
8. [📦 Getting Started](#8--getting-started)
   - 8.1 [🔧 Prerequisites](#81--prerequisites)
   - 8.2 [⬇️ Install](#82-%EF%B8%8F-install)
   - 8.3 [🖥️ Run Locally](#83-%EF%B8%8F-run-locally)
9. [🚀 Deployment](#9--deployment)
   - 9.1 [☁️ Streamlit Cloud (Recommended)](#91-%EF%B8%8F-streamlit-cloud-recommended)
   - 9.2 [🐳 Docker + AWS EC2](#92--docker--aws-ec2)
10. [🗺️ Roadmap](#10-%EF%B8%8F-roadmap)
11. [🤝 Contributing](#11--contributing)
12. [📄 Changelog](#12--changelog)
13. [👤 Author](#13--author)
14. [⭐ Show Your Support](#14--show-your-support)

---

## 1. 🌌 What is OpenShelf Horizon?

**OpenShelf | Horizon** is a production-grade, end-to-end machine learning application that delivers personalised book recommendations using **Collaborative Filtering**. It processes over **270,000 user ratings** across **742+ books**, runs a live KNN recommendation engine, and wraps it all in a cinematic **Cyber-Neon** monitoring dashboard built in Streamlit.

> 🎯 **Quick Start:** The app is **Pre-Bloomed** — initial `artifacts/*.pkl` are committed to the repo so recommendations work instantly on arrival. The **🔄 Sync Library & Re-train** button remains available to refresh the model with the latest data whenever you need it.

| 🔖 | Version | 📦 Highlight |
|:---:|:---:|:---|
| 🆕 | `v2.0` | Live pipeline monitor, glassmorphism Horizon UI, Docker + AWS deploy |
| 🔄 | `v1.5` | CSR matrix optimisation, Minkowski metric KNN, sparsity handling |
| 🎉 | `v1.0` | Initial release — collaborative filtering on Book-Crossing dataset |

---

## 2. ✨ Key Features

<table>
  <tr><td>🤖</td><td><strong>Collaborative Filtering</strong></td><td>KNN model with Minkowski metric identifies the most similar readers to generate precise neighbour-based book picks</td></tr>
  <tr><td>📡</td><td><strong>Live Pipeline Monitor</strong></td><td>Real-time tracking of Data Validation → Transformation → Model Training stages — watch the ML engine bloom</td></tr>
  <tr><td>💾</td><td><strong>CSR Matrix Engine</strong></td><td>Compressed Sparse Row matrix handles 270,000+ ratings with minimal memory — production-grade sparsity management</td></tr>
  <tr><td>📊</td><td><strong>Health Metrics Dashboard</strong></td><td>Live display of Library Size (742+ books), Algorithm type, Model Accuracy, and System Growth indicators</td></tr>
  <tr><td>🌌</td><td><strong>Cyber-Neon UI</strong></td><td>Custom CSS deep-space gradients, glassmorphism navigation panels, and glowing neon interaction states</td></tr>
  <tr><td>🔄</td><td><strong>One-Click Re-train</strong></td><td>🔄 Sync Library & Re-train button triggers the full pipeline end-to-end — no CLI, no code</td></tr>
  <tr><td>🐳</td><td><strong>Containerised Deploys</strong></td><td>Official Dockerfile for Streamlit port 8501 — deploy to AWS EC2, GCP, or any container host</td></tr>
  <tr><td>⚙️</td><td><strong>GitHub Actions CI/CD</strong></td><td>Auto-deploy to Streamlit Cloud on every <code>git push</code> to main</td></tr>
</table>

---

## 3. 🏗️ System Architecture

### 3.1 🗂️ Project Structure

```
📚 OpenShelf-E2E/
│
├── 🚀 app.py                        # Streamlit entry point — Horizon UI
│
├── 🤖 src/                          # Core ML Pipeline Modules
│   ├── 📊 data_validation.py        # Pandas 2.0+ schema & quality checks
│   ├── 🔄 data_transformation.py    # CSR matrix builder & sparsity handler
│   ├── 🧠 model_trainer.py          # KNN model training & artifact export
│   └── 🔍 recommender.py            # Inference engine — query → neighbours
│
├── 📁 data/                         # Dataset Layer
│   ├── 📖 Books.csv                 # Book metadata (title, author, ISBN)
│   ├── 👤 Users.csv                 # Anonymised user profiles
│   └── ⭐ Ratings.csv               # 270,000+ user–book rating pairs
│
├── 🧪 artifacts/                    # Pre-trained ML Artifacts (committed to repo)
│   ├── 🧠 model.pkl                 # Trained KNN model — app works on arrival
│   ├── 💾 csr_matrix.pkl            # Sparse rating matrix
│   └── 📋 book_names.pkl            # Filtered book index
│
├── 🎨 styles/
│   └── 🌌 horizon.css               # Cyber-Neon custom Streamlit theme
│
├── 🐳 Dockerfile                    # Container definition (port 8501)
├── ⚙️ .github/workflows/deploy.yml  # GitHub Actions CI/CD pipeline
├── 📦 requirements.txt              # Python dependencies
└── 🔒 .env.example                  # Environment variable template
```

### 3.2 📐 Architecture Diagram

```mermaid
graph TD
    U[👤 USER] -->|Opens App| ST[🚀 Streamlit App — app.py]

    subgraph Pipeline ["🤖 ML PIPELINE — On Sync Trigger"]
        DV[📊 Data Validation<br/>Pandas 2.0+ · Schema Checks]
        DT[🔄 Data Transformation<br/>CSR Matrix · Sparsity Handling]
        MT[🧠 Model Training<br/>KNN · Minkowski · Brute-Force]
        ART[🧪 Artifacts<br/>model.pkl · csr_matrix.pkl]
    end

    subgraph Data ["📁 DATA LAYER"]
        BK[📖 Books.csv]
        US[👤 Users.csv]
        RT[⭐ Ratings.csv<br/>270K+ entries]
    end

    subgraph Inference ["🔍 RECOMMENDATION ENGINE"]
        REC[🔍 recommender.py]
        OUT[📚 Top-N Book Picks]
    end

    ST -->|🔄 Sync Trigger| DV
    BK & US & RT --> DV
    DV --> DT
    DT --> MT
    MT --> ART
    ART --> REC
    ST -->|User Query| REC
    REC --> OUT
    OUT --> ST

    subgraph Monitor ["🌌 HORIZON DASHBOARD"]
        HEALTH[📡 Live Health Metrics]
        STAGES[⚡ Pipeline Stage Tracker]
    end

    ST --> Monitor

    classDef app fill:#1e1b4b,stroke:#7c3aed,stroke-width:2px,color:#fff;
    classDef pipe fill:#0f172a,stroke:#a855f7,stroke-width:2px,color:#fff;
    classDef data fill:#0a1a0a,stroke:#10b981,stroke-width:2px,color:#10b981;
    classDef infer fill:#0a0a2e,stroke:#06b6d4,stroke-width:2px,color:#fff;
    classDef monitor fill:#1a0a2e,stroke:#ff007f,stroke-width:2px,color:#fff;
    classDef user fill:#000,stroke:#7c3aed,stroke-width:2px,color:#fff;

    class U user;
    class ST app;
    class DV,DT,MT,ART pipe;
    class BK,US,RT data;
    class REC,OUT infer;
    class HEALTH,STAGES monitor;
```

---

## 4. 🛠️ Tech Stack

### 🧠 ML & Data
<p>
  <img src="https://img.shields.io/badge/Python_3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/Pandas_2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/SciPy_CSR-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white" />
</p>

### 🌌 Frontend & UI
<p>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Custom_CSS-Horizon_Theme-7c3aed?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Glassmorphism-a855f7?style=for-the-badge" />
</p>

### ☁️ DevOps & Cloud
<p>
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/AWS_EC2-FF9900?style=for-the-badge&logo=amazon-aws&logoColor=white" />
  <img src="https://img.shields.io/badge/GitHub_Actions-2088FF?style=for-the-badge&logo=githubactions&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
</p>

| ⚙️ Capability | 🔬 Implementation | 🏆 Result |
|:---|:---|:---|
| 🧠 Recommendation | KNN · Minkowski · Brute-force | Precise neighbour detection |
| 💾 Memory Efficiency | CSR Sparse Matrix (SciPy) | 270K+ ratings, minimal RAM |
| 📊 Data Quality | Pandas 2.0+ pipeline | Schema-validated ingestion |
| 🔄 CI/CD | GitHub Actions → Streamlit Cloud | Auto-deploy on `git push` |
| 🐳 Portability | Docker on port 8501 | Deploy anywhere |

---

## 5. 🤖 ML Pipeline

The pipeline follows a strict **3-stage sequential flow**. Initial artifacts are pre-committed so the app works on arrival — the 🔄 Sync button is an optional refresh trigger for pulling the latest data.

### 5.1 📊 Data Validation

Ingests the raw Book-Crossing dataset using a **Pandas 2.0+** pipeline:

- Validates schema — column names, data types, null thresholds
- Filters books with a minimum rating count (removes cold-start noise)
- Outputs a clean, typed DataFrame ready for transformation

### 5.2 🔄 Transformation

Converts the validated ratings into a **CSR (Compressed Sparse Row) Matrix**:

- Pivots user-book ratings into a 2D matrix (users × books)
- Applies sparsity handling — only non-zero entries stored in memory
- Result: a memory-efficient structure over 270,000+ ratings

### 5.3 🧠 Model Engine

Trains a **K-Nearest Neighbors** model on the CSR matrix:

| ⚙️ Parameter | 🔬 Value | 📝 Reason |
|:---|:---:|:---|
| Algorithm | `brute` | Exact neighbour search — no approximation |
| Metric | `minkowski` | Generalised distance (p=2 → Euclidean) |
| n_neighbors | configurable | Tune via UI slider |
| Matrix input | CSR sparse | Memory-efficient for large rating sets |

**How CSR × KNN interact:**

```
Dense Matrix (users × books):     CSR Sparse Encoding:
┌─────────────────────────┐        Only non-zero ratings stored
│ 0  0  5  0  0  3  0  0 │   →    [row_ptr | col_idx | values]
│ 0  4  0  0  2  0  0  0 │        ~15 MB vs ~2 GB dense
│ 3  0  0  0  0  0  4  0 │
└─────────────────────────┘        KNN computes Minkowski distance
                                   only between non-zero vectors
                                   → O(k·n) not O(n²)
```

### 5.4 ⚡ Pipeline Sequence

```mermaid
sequenceDiagram
    autonumber
    participant U  as 👤 User
    participant ST as 🚀 Streamlit
    participant DV as 📊 Validator
    participant DT as 🔄 Transformer
    participant MT as 🧠 Trainer
    participant AR as 🧪 Artifacts

    U->>ST: Click 🔄 Sync Library & Re-train
    ST->>DV: Load Books, Users, Ratings CSVs
    DV-->>ST: ✅ Validation passed (stage 1/3)
    ST->>DT: Build CSR sparse matrix
    DT-->>ST: ✅ Transformation done (stage 2/3)
    ST->>MT: Fit KNN model on CSR matrix
    MT-->>AR: Export model.pkl + csr_matrix.pkl
    AR-->>ST: ✅ Model ready (stage 3/3)
    ST-->>U: 🌌 Horizon dashboard — BLOOM COMPLETE
```

---

## 6. 🌌 Horizon UI & Monitoring

**Horizon** is not just a recommender — it's a live ML observability dashboard:

| 🖥️ Panel | 📝 What It Shows |
|:---|:---|
| 📡 **Pipeline Stages** | Real-time progress through Validation → Transformation → Training |
| 📊 **Health Metrics** | Library size (742+ books), algorithm type, model status, system growth |
| 🔍 **Recommendation Output** | Top-N book picks with cover, author, and similarity score |
| 🌌 **Cyber-Neon Aesthetic** | Deep-space gradients, glassmorphism panels, neon glow states |

> 💡 The Horizon theme is powered by a custom `horizon.css` injected into Streamlit via `st.markdown()` — no external UI framework required.

---

## 7. ⚡ Performance

| 📊 Metric | 🎯 Value | 📝 Notes |
|:---|:---:|:---|
| 📖 Dataset Size | `270,000+` ratings | Book-Crossing dataset |
| 📚 Book Catalogue | `742+` books | After cold-start filter |
| 🏗️ Pipeline Runtime | `< 30s` | Full re-train on Streamlit Cloud |
| 💾 Memory (CSR) | `~15 MB` | vs ~2 GB dense matrix |
| 🧠 KNN Inference | `< 200ms` | Per recommendation query |
| 🔄 CI/CD Deploy | `< 2 min` | GitHub Actions → Streamlit Cloud |

---

## 8. 📦 Getting Started

### 8.1 🔧 Prerequisites

| 🛠️ Tool | 📌 Version | 🔗 Link |
|:---|:---:|:---|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) | `≥ 3.10` | [python.org](https://www.python.org/) |
| ![Conda](https://img.shields.io/badge/Conda-44A833?style=flat&logo=anaconda&logoColor=white) | any | [anaconda.com](https://www.anaconda.com/) |
| ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white) | any | [git-scm.com](https://git-scm.com/) |

### 8.2 ⬇️ Install

**📥 Step 1 — Clone**

```bash
git clone https://github.com/salonyranjan/OpenShelf-E2E.git
cd OpenShelf-E2E
```

**🐍 Step 2 — Create environment & install dependencies**

```bash
# With Conda (recommended)
conda create -n openshelf python=3.10 -y
conda activate openshelf
pip install -r requirements.txt

# Or with venv
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 8.3 🖥️ Run Locally

**🚀 Step 3 — Launch the Horizon dashboard**

```bash
streamlit run app.py
```

> 🌐 Opens at [http://localhost:8501](http://localhost:8501) — click **🔄 Sync Library & Re-train** to bloom the ML engine.

---

## 9. 🚀 Deployment

### 9.1 ☁️ Streamlit Cloud (Recommended)

```
1. Push your repo to GitHub
2. Go to share.streamlit.io → New app
3. Select repo, set main file to app.py, Python version 3.10
4. Click Deploy — auto-deploys on every git push ✅
```

### 9.2 🐳 Docker + AWS EC2

**Build & run the container:**

```bash
# Build image
docker build -t salonyranjan/openshelf:latest .

# Run on port 8501
docker run -d -p 8501:8501 salonyranjan/openshelf:latest
```

**Recommended `Dockerfile` (slim + health-checked):**

```dockerfile
# Use slim base — keeps image ~200 MB vs ~900 MB for full python:3.10
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

# Health check — Docker/AWS monitors if Streamlit is actually responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]
```

**On AWS EC2:**

```bash
# Pull & run on your instance
docker pull salonyranjan/openshelf:latest
docker run -d -p 8501:8501 salonyranjan/openshelf:latest
```

> ⚠️ **EC2 Gotcha:** Open **port 8501** in your AWS Security Group inbound rules, or the app will be unreachable from the browser.

---

## 10. 🗺️ Roadmap

| Status | 🚀 Feature | 🎯 Priority |
|:---:|:---|:---:|
| ✅ | KNN Collaborative Filtering engine | 🔴 Core |
| ✅ | CSR sparse matrix optimisation | 🔴 Core |
| ✅ | Horizon Cyber-Neon UI + live monitor | 🔴 Core |
| ✅ | Docker + AWS EC2 deployment | 🔴 Core |
| ✅ | GitHub Actions CI/CD | 🟡 High |
| 🔄 | **Content-Based Filtering** — genre + author similarity | 🟡 High |
| 🔄 | **Hybrid Recommender** — CF + Content-Based fusion | 🟡 High |
| 📅 | **User Auth** — personalised reading lists · use [`streamlit-authenticator`](https://github.com/mkhorasani/Streamlit-Authenticator) for easiest Horizon login integration | 🟢 Planned |
| 📅 | **ALS / Matrix Factorisation** — implicit feedback model | 🟢 Planned |
| 📅 | **LLM Integration** — natural language book search | 🟢 Planned |
| 💡 | **Community Reviews** — user annotations per title | 🔵 Idea |

> 💬 Have an idea? [Open a feature request →](https://github.com/salonyranjan/OpenShelf-E2E/issues/new)

---

## 11. 🤝 Contributing

All contributions are **warmly welcome**! 📚

```bash
# 1. Fork the repository on GitHub
# 2. Create your feature branch
git checkout -b feature/your-feature

# 3. Commit with conventional format
git commit -m "feat: add your feature"
# Prefixes: fix: | docs: | style: | refactor: | test: | chore:

# 4. Push & open a PR
git push origin feature/your-feature
```

**Priority areas:**

| 🔥 Area | 📝 What's Needed |
|:---|:---|
| 🧠 Hybrid Model | Combine CF + content-based filtering |
| 🧪 Tests | Pytest coverage for pipeline stages |
| 🌐 Dataset | Support for additional book datasets |
| 🎨 UI | More Horizon panel variants, dark/light toggle |

---

## 12. 📄 Changelog

| Version | Highlights |
|:---|:---|
| 🆕 `v2.0.0` | Horizon Cyber-Neon UI · Live pipeline monitor · Docker + AWS deploy |
| `v1.5.0` | CSR matrix optimisation · Minkowski KNN · sparsity handling |
| `v1.0.0` | 🎉 Initial release — collaborative filtering on Book-Crossing dataset |

---

## 13. 👤 Author

<table style="border:none;">
  <tr>
    <td align="center" style="border:none;" width="160">
      <img src="https://github.com/salonyranjan.png" width="145" style="border-radius:50%; border:3px solid #7c3aed; box-shadow:0 0 25px #7c3aed, 0 0 50px #a855f740;" alt="Salony Ranjan" />
    </td>
    <td style="border:none; padding-left:22px;">
      <h3>✦ Salony Ranjan</h3>
      <p>🤖 ML Engineer &nbsp;·&nbsp; 🧑‍💻 Full-Stack Dev &nbsp;·&nbsp; 🎨 UI/UX Specialist</p>
      <p><em>"Building intelligent systems that are as beautiful to look at as they are powerful to use."</em></p>
      <br/>
      <a href="https://www.linkedin.com/in/salony-ranjan-b63200280/"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" /></a>
      &nbsp;
      <a href="https://github.com/salonyranjan"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" /></a>
      &nbsp;
      <a href="mailto:salonyranjan@gmail.com"><img src="https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white" /></a>
      &nbsp;
      <a href="https://vertex-flow-phi.vercel.app/"><img src="https://img.shields.io/badge/Portfolio-7c3aed?style=for-the-badge&logo=react&logoColor=white" /></a>
    </td>
  </tr>
</table>

---

## 14. ⭐ Show Your Support

<div align="center">

If OpenShelf helped you find your next great read, impressed you with the ML pipeline, or inspired your own project — show it some love! 📚

> 💡 **Pro Tip:** Go to your GitHub repo **Settings → Social Preview** and upload a screenshot of the Cyber-Neon dashboard. When you share the repo link on LinkedIn, it'll show your stunning UI instead of a generic GitHub logo — instant impression upgrade.

<a href="https://github.com/salonyranjan/OpenShelf-E2E/stargazers"><img src="https://img.shields.io/badge/⭐_Star_This_Repo-7c3aed?style=for-the-badge&logo=github&logoColor=white" /></a>
&nbsp;
<a href="https://github.com/salonyranjan/OpenShelf-E2E/fork"><img src="https://img.shields.io/badge/🍴_Fork_&_Build-a855f7?style=for-the-badge&logo=github&logoColor=white" /></a>
&nbsp;
<a href="https://openshelf-e2e.streamlit.app/"><img src="https://img.shields.io/badge/🚀_Live_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" /></a>
&nbsp;
<a href="https://github.com/salonyranjan/OpenShelf-E2E/issues/new"><img src="https://img.shields.io/badge/💡_Feature_Request-06b6d4?style=for-the-badge" /></a>

<br/><br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:7c3aed,40:a855f7,75:06b6d4,100:000000&height=130&section=footer&animation=fadeIn" />

<br/>

*Developed with* 💜 *by* [**Salony Ranjan**](https://github.com/salonyranjan) &nbsp;·&nbsp; *© 2026 OpenShelf · MIT*

<img src="https://readme-typing-svg.herokuapp.com?font=Orbitron&weight=600&size=13&duration=4000&pause=1000&color=a855f7&center=true&vCenter=true&width=520&lines=SYSTEM+STATUS%3A+BLOOM+COMPLETE+🌌;270K%2B+RATINGS+PROCESSED+⚡;STAY+CURIOUS+·+READ+·+BUILD+EPIC" />

<img src="https://komarev.com/ghpvc/?username=salonyranjan&label=PROFILE+VIEWS&color=7c3aed&style=for-the-badge" />

</div>
