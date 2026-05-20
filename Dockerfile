FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY app/ ./app/

EXPOSE 7861 8000

# APP_MODE=api -> FastAPI on 8000; anything else -> Gradio on 7861
ENV APP_MODE=gradio \
    GRADIO_SERVER_PORT=7861
CMD ["sh", "-c", "if [ \"$APP_MODE\" = \"api\" ]; then uvicorn app.main:app --host 0.0.0.0 --port 8000; else python -m app.ui_gradio; fi"]
