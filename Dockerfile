# Self-contained demo image: Python service + Yelp restaurant dataset.
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Service code
COPY data.py parse.py demo.py invite.py api.py blend.py vibe.py ./

# Frontend served at / by the API (single-URL deploy).
COPY frontend.html ./

# Dataset (61 MB CSV) fetched at build time so the repo stays light;
# social signals baked in so the container runs with no mounts.
ADD https://huggingface.co/datasets/jaimik69/Yelp-Restaurant-Dataset/resolve/main/restaurants.csv ./dataset/restaurants.csv
COPY dataset/social_signals.json ./dataset/social_signals.json
ENV DINING_DATA_PATH=/app/dataset/restaurants.csv

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
