# Steps

1. Start the llama.cpp server

```
llama-server \           
  -hf LiquidAI/LFM2.5-VL-1.6B-GGUF:Q8_0 \
  --port 8080 \
  --host 0.0.0.0 \
  --n-gpu-layers 999 \
  --ctx-size 2048
```