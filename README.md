# Comparison of deep learning sequential models

Models LSTM, GRU and Transformer are trained and tested on stock price data.

## Local Installation

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