# Comparison of deep learning sequential models

Models LSTM, GRU and Transformer are trained and tested on stock price data.

## Local Installation

```python 
pip install -r requirements.txt
```

## Docker

```bash
cd SequentialComparison-TVZ

# Build image
docker build -t streamlit-app .

# Create container
docker run --name my_streamlit_app -p 8501:8501 streamlit-app

# Running existing container
docker start -a my_streamlit_app
```