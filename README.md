This is the Python SDK of SingTown AI, develop with SingTown AI Desktop and SingTown AI Cloud

# install
```
pip install singtown_ai
```

# use
```
from singtown_ai import runner

runner.login(host: str, token: str)
runner.retrieve_task(id: UUID)
runner.retrieve_project(id: UUID)
runner.update_status(id:UUID, status:TaskStatus)
runner.upload_result(id:str, result_file:str)
runner.upload_log(id:UUID, log: str)
runner.upload_metric(id:UUID, epoch:int, metric:str)
runner.downlaod_images(annotations: List[Annotation], save_path: str)
```

# mock training
```
python -m singtown_ai.mock_runner  http://127.0.0.1:8000 6ba7b810-9dad-11d1-80b4-00c04fd430c8 1234567890
```

# docker
```
docker build -t singtown-ai-runner-mock:0.1 .
docker run --network="host" --rm singtown-ai-runner-mock:0.1 python -m singtown_ai.mock_runner http://host.docker.internal:8000 6ba7b810-9dad-11d1-80b4-00c04fd430c8 1234567890
```
