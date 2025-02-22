This is the Python SDK of SingTown AI, develop with SingTown AI Desktop and SingTown AI Cloud

# install
```
pip install singtown_ai
```

# Runner
todo

# mock training
```
python -m singtown_ai.runner  http://127.0.0.1:8000 b45f8ac0-a3a7-43ac-9d15-d2f4a505425c 1234567890
```

# docker
```
docker build -t singtown-ai-runner-mock:0.1 .
docker run --network="host" --rm singtown-ai-runner-mock:0.1 python -m singtown_ai.mock_runner http://host.docker.internal:8000 b45f8ac0-a3a7-43ac-9d15-d2f4a505425c 1234567890
```
