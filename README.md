# Launch a Policy Server

clone the repo to be mount in docker. 
```bash
git clone https://github.com/DelinQu/comet_server.git $PATH_TO_MODELS/comet_server
```

- For single instance mode
```bash
PATH_TO_MODELS=YOUR_HF_MODEL_PATH
task_id=0

docker run \
  -p 8000:8000 \
  -v $PATH_TO_MODELS:/root/models \
  -w /root/openpi \
  littlespray/comet-submission:latest \
  bash -c """
  . /root/.local/bin/env
  . /root/openpi/.venv/bin/activate
  bash run.sh $task_id
  """
```

- For multi instance mode

```bash
PATH_TO_MODELS=YOUR_HF_MODEL_PATH

docker run \
  -p 8000:8000 \
  -v $PATH_TO_MODELS:/root/models \
  -w /root/openpi \
  littlespray/comet-submission:latest \
  bash -c """
    . /root/.local/bin/env
    . /root/openpi/.venv/bin/activate
    
    python /root/models/comet_server/serve_b1k_multi.py policy:checkpoint
  """
```

- For Testing, we set the `example["task_id"] = torch.tensor(task_id)`

```bash
PATH_TO_MODELS=YOUR_HF_MODEL_PATH

docker run \
  -p 8000:8000 \
  -v $PATH_TO_MODELS:/root/models \
  -w /root/openpi \
  -it \
  littlespray/comet-submission:latest \
  bash

# 1. server terminal:
python /root/models/comet_server/serve_b1k_multi.py policy:checkpoint

# 2. client terminal
python /root/models/comet_server/test_b1k_openpi.py 

# 3. check the log file
```