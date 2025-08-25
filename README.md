# Comparison of deep learning sequential models

Models LSTM, GRU and Transformer are trained and tested on stock price data.

## 🚀 Live Demo

The app can be found at the following link: 

[![Streamlit Community Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://sequentialcomparison-tvz-luy7mu7dtew6g2i6crparp.streamlit.app/)

## Local Installation (uv)

```python 
uv sync
uv pip install torch==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

If you would like to use PyTorch with CUDA (adjust CUDA version if needed):

```python 
uv pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Local Installation (pip)

```python 
pip install -r requirements.txt
```

If you would like to use PyTorch with CUDA (adjust CUDA version if needed):

```python 
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Docker Setup

```bash
cd SequentialComparison-TVZ

# Build image
docker build -t seqcomp-tvz .

# Create container
docker run --name seq_app -p 8501:8501 seqcomp-tvz

# Running existing container
docker start -a seq_app
```