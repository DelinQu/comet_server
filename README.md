# Launch a Policy Server

- For single instance mode
```bash
docker run \
  --gpus all \
  -p 8000:8000 \
  -v $PATH_TO_MODELS:/root/models \
  -w /root/openpi \
  comet_submission:latest \
  bash run.sh $task_id
```

- For multi instance mode
```bash
docker run \
  --gpus all \
  -p 8000:8000 \
  -v $PATH_TO_MODELS:/root/models \
  -w /root/openpi \
  comet_submission:latest \
  bash -c """
    cp serve_b1k_multi.py .
    uv run scripts/serve_b1k_multi.py
  """
```