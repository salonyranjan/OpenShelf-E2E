# 📚 OpenShelf | Horizon: Cyber-Neon AI Recommender

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://openshelf-e2e.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**OpenShelf | Horizon** is an end-to-end machine learning application that provides personalized book recommendations using Collaborative Filtering. Built with a **Cyber-Neon** aesthetic, it features a live monitoring dashboard to track the ML pipeline in real-time.

---

## 🔗 Live Application
Access the deployed app here: **[OpenShelf Horizon Live Dashboard](https://openshelf-e2e.streamlit.app/)**

> **Note**: On your first visit, click the **🔄 Sync Library & Re-train** button to "bloom" the engine and generate the ML artifacts in the cloud environment.

---

## 🌌 Live Dashboard & Monitoring
Unlike static apps, **Horizon** includes a built-in **Live Pipeline Monitor**:
* **Real-time Status**: Track Data Validation (Stage 01), Transformation (Stage 02), and Model Training (Stage 03) as they happen.
* **Health Metrics**: Displays Library Size (742+ books), Algorithm type (KNN), and System Growth status.
* **Environment Parity**: Seamlessly transitions from local development in **Patna** to Linux-based Cloud deployment.

## 🏗️ Technical Architecture
The project follows a modular production-grade structure:
1.  **Data Validation**: Modern Pandas 2.0+ pipeline handling 270,000+ ratings.
2.  **Transformation**: Sparsity handling and CSR Matrix generation for memory efficiency.
3.  **Model**: KNN model with Brute-force algorithm and Minkowski metric.

## 🚀 Installation & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/salonyranjan/OpenShelf-E2E.git](https://github.com/salonyranjan/OpenShelf-E2E.git)
cd OpenShelf-E2E
```
### 2. Create Virtual Environment
```bash
conda create -n books python=3.10 -y
conda activate books
pip install -r requirements.txt
```
### 3. Run the Application
```bash
streamlit run app.py
```
Note: On first run, click the 🔄 Sync Library & Re-train button to "bloom" the engine and generate the ML artifacts.

---

## 🛠️ Tech Stack
Language: Python 3.10

ML Libraries: Scikit-learn, Pandas, NumPy

Frontend: Streamlit (Custom Cyber-Neon Horizon Theme)

Deployment: GitHub Actions & Streamlit Cloud
