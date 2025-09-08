FROM python:3.10-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /uvx /bin/

WORKDIR /app

COPY streamlit/ streamlit/
COPY utils/ utils/
COPY models/ models/
COPY scalers/ scalers/
COPY pyproject.toml .
COPY uv.lock .

RUN uv sync --locked

EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "streamlit/app.py"]
