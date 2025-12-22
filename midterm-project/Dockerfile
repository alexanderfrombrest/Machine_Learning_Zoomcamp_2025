FROM python:3.12.4-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY ".python-version" "pyproject.toml" "uv.lock" "./"
RUN uv sync --locked

COPY "predict.py" "transform.py" "model_pipeline.bin" "./"

EXPOSE 9696

ENTRYPOINT ["/app/.venv/bin/uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]