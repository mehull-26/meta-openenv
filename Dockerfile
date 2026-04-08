FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY --chown=user . /app/adaptive_learning_system

ENV PYTHONPATH="/app:${PYTHONPATH}"

EXPOSE 8000

CMD ["uvicorn", "adaptive_learning_system.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
