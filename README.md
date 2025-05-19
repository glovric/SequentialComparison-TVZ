# Comparison of deep learning sequential models

Models LSTM, GRU and Transformer are trained and tested on stock price data.

## Local Installation

```python 
pip install -r requirements.txt
```

## Docker

```bash
cd SequentialComparison-TVZ
docker build -t streamlit-app . 
docker run --name my_streamlit_app -p 8501:8501 streamlit-app
```