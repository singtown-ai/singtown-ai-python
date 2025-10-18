import sys
from unittest.mock import patch
import runpy


def test_main():
    with patch.object(
        sys,
        "argv",
        [
            "singtown_ai.test",
            "--host=https://ai.singtown.com",
            "--token=012345",
            "--task_id=1",
            "--epochs=10",
            "--train_interval=0.1",
            "--mock=True",
        ],
    ):
        runpy.run_module("singtown_ai.dryrun", run_name="__main__")
