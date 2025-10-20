# Setup env

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -e ".[dev]"
pytest
```
