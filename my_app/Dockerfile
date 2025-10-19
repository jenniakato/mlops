# Base image officielle Python 3.12 slim
FROM python:3.12.12-slim

# Répertoire de travail dans le conteneur
WORKDIR /my_app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    libffi-dev \
    libssl-dev \
    libpq-dev \
    curl \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    python3-dev \
    libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

# Copier uniquement requirements.txt pour le cache Docker
COPY requirements.txt .

# Mettre à jour pip et installer d'abord les packages lourds
RUN pip install --upgrade pip wheel setuptools
RUN pip install numpy==1.26.4 pandas==2.3.3 scipy==1.16.2 scikit-learn==1.7.2

# Installer le reste des dépendances en préférant les wheels binaires
RUN pip install --no-cache-dir -r requirements.txt --prefer-binary

# Copier le reste de l'application
COPY . .

# Exposer le port Streamlit
EXPOSE 8501

# Lancer l'application Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
