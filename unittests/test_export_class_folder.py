from singtown_ai import SingTownAIClient, export_class_folder
import os
import pytest


def test_export_class_folder_default(tmp_path, task_cf_file, dataset_cf_file):
    client = SingTownAIClient(
        mock_task_path=task_cf_file,
        mock_dataset_path=dataset_cf_file,
    )
    export_path = tmp_path / "dataset"
    export_class_folder(client, export_path)
    assert len(os.listdir(export_path / "TRAIN/cat")) == 7
    assert len(os.listdir(export_path / "TRAIN/dog")) == 7
    assert len(os.listdir(export_path / "VALID/cat")) == 2
    assert len(os.listdir(export_path / "VALID/dog")) == 2
    assert len(os.listdir(export_path / "TEST/cat")) == 1
    assert len(os.listdir(export_path / "TEST/dog")) == 1


def test_export_class_folder_strpath(tmp_path, task_cf_file, dataset_cf_file):
    client = SingTownAIClient(
        mock_task_path=task_cf_file,
        mock_dataset_path=dataset_cf_file,
    )
    export_path = tmp_path / "dataset"
    export_class_folder(client, str(export_path))
    assert len(os.listdir(export_path / "TRAIN/cat")) == 7
    assert len(os.listdir(export_path / "TRAIN/dog")) == 7
    assert len(os.listdir(export_path / "VALID/cat")) == 2
    assert len(os.listdir(export_path / "VALID/dog")) == 2
    assert len(os.listdir(export_path / "TEST/cat")) == 1
    assert len(os.listdir(export_path / "TEST/dog")) == 1


def test_export_class_folder_typeerror(tmp_path, task_od_file):
    client = SingTownAIClient(mock_task_path=task_od_file)
    export_path = tmp_path / "dataset"
    with pytest.raises(RuntimeError):
        export_class_folder(client, export_path)
