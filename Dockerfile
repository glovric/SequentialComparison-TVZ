FROM python:3.10-slim

WORKDIR /app

COPY streamlit/ streamlit/
COPY utils/ utils/
COPY models/ models/
COPY scalers/ scalers/

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r streamlit/requirements.txt

EXPOSE 8501

CMD ["streamlit", "run", "streamlit/app.py"]
