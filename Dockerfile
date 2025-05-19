FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY streamlit/requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project code
COPY streamlit/ streamlit/
COPY utils/ utils/
COPY models/ models/
COPY scalers/ scalers/

# Expose port for Streamlit
EXPOSE 8501

# Default command
CMD ["streamlit", "run", "streamlit/app.py"]
