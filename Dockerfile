# Live demo image (Hugging Face Spaces, Docker SDK, port 7860)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=7860

# Hugging Face Spaces runs containers as user 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user backend ./backend
COPY --chown=user frontend ./frontend

USER user
EXPOSE 7860

CMD ["gunicorn", "--chdir", "frontend", "dashboard:server", \
     "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "4", "--timeout", "180"]
