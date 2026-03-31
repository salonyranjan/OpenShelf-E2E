## 📚 OpenShelf E2E
An End-to-End Book Recommender System using Collaborative Filtering 🔥

OpenShelf E2E is a full-stack machine learning application that provides personalized book recommendations. By analyzing user-item interaction patterns, the system identifies "latent" similarities between readers to suggest their next favorite book.

## 🚀 Overview
Traditional search engines rely on keywords. OpenShelf uses Collaborative Filtering, meaning it understands that if User A and User B both enjoyed The Hobbit, User A might also enjoy other books User B has rated highly

# End-to-End-Book-Recommender-System

## Workflow

- config.yaml
- entity
- config/configuration.py
- components
- pipeline
- main.py
- app.py


# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/entbappy/End-to-End-Book-Recommender-System.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n books python=3.7.10 -y
```

```bash
conda activate books
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


Now run,
```bash
streamlit run app.py
```


# Streamlit app Docker Image Deployment

## 1. Login with your AWS console and launch an EC2 instance
## 2. Run the following commands

Note: Do the port mapping to this port:- 8501

```bash
sudo apt-get update -y

sudo apt-get upgrade

#Install Docker

curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker
```

```bash
git clone "your-project"
```

```bash
docker build -t entbappy/stapp:latest . 
```

```bash
docker images -a  
```

```bash
docker run -d -p 8501:8501 entbappy/stapp 
```

```bash
docker ps  
```

```bash
docker stop container_id
```

```bash
docker rm $(docker ps -a -q)
```

```bash
docker login 
```

```bash
docker push entbappy/stapp:latest 
```

```bash
docker rmi entbappy/stapp:latest
```

```bash
docker pull entbappy/stapp
```







=======
# OpenShelf-E2E
🚀 Core Logic: Collaborative Filtering (Memory-based &amp; Model-based) 🛠️ Stack: Python, Pandas, Scikit-Learn, Flask/Streamlit 📊 Objective: To solve the "Discovery Problem" in digital libraries by identifying latent patterns in user behavior. Includes data preprocessing, similarity matrix computation, and a responsive UI.
>>>>>>> a5a8c41b525cfe3cf71457e928e12d60134eb274
