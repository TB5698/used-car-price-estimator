# Used car price estimator: Streamlit chat app + trained model.
# Build:  docker build -t used-car-estimator .
# Run:    docker run --rm -p 8501:8501 --env-file .env used-car-estimator
# Then open http://localhost:8501
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# LightGBM needs the OpenMP runtime, which the slim image leaves out
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# dependencies first, so rebuilds after a code change reuse this layer
COPY requirements-app.txt .
RUN pip install -r requirements-app.txt

# src/ must be present: the saved model refers to src.preprocess.prepare_features
COPY src/ src/
COPY config.yaml streamlit_app.py ./
COPY models/ models/

# run as a normal user, not root
RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# the API key is not in the image: pass LLM_API_KEY and LLM_MODEL at run time
CMD ["streamlit", "run", "streamlit_app.py", \
     "--server.address=0.0.0.0", "--server.port=8501", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
