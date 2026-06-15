File Structure:

Batched data generation process
Utilities/generate_model_responses.py Generates the initial reponses that we will use to train the linear probes on.

## Setup:

```
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install vLLM with CUDA 12.4
uv pip install vllm==0.20.2 --extra-index-url https://wheels.vllm.ai/0.20.2/cu124 --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match

pip install \
  transformers==4.57.6 \
  datasets \
  huggingface_hub \
  wandb \
  jupyterlab \
  ipykernel

#current uv env
uv add install torch torchvision torchaudio --default-url https://download.pytorch.org/whl/cu124


uv add transformers==4.57.6 \
  datasets \
  huggingface_hub \
  wandb \
  jupyterlab \
  ipykernel \
  google-genai \
  prefect
```

If you already don't have a dataset used for trainign your transformer probes you will have to setup a prefect server.  This will likely be a headache for some at times:

#todo: possible progres migration: prefect config set PREFECT_API_DATABASE_CONNECTION_URL="postgresql+asyncpg://postgres:yourTopSecretPassword@localhost:5432/prefect"

``` 
prefect server start
prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"
```

By default your dataset will be stored in ~/.prefect/prefect.db. No additional configuration is needed for basic use.


Note it is unclear within the retraction/correction pipeline if the corrected mistakes will be included within the training methodology IE will you have a response with multiple retractions within it?