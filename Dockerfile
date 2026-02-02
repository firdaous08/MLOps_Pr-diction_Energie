# 1. Image de base (Python léger)
FROM python:3.9-slim

# 2. Dossier de travail dans le conteneur
WORKDIR /code

# 3. Copie des dépendances et installation
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 4. Copie de tout le code
COPY . /code

# 5. Droits d'accès (Indispensable pour Hugging Face)
RUN chmod -R 777 /code

# 6. Lancement de l'API sur le port 7860 (Port obligatoire HF Spaces)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]