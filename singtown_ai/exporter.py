from pathlib import Path
from os import PathLike
import yaml


def export_class_folder(client, folder: str | PathLike):
    if client.task.project.type != "CLASSIFICATION":
        raise RuntimeError("export_class_folder only support CLASSIFICATION task")

    for annotation in client.dataset:
        image_folder = Path(folder) / annotation.subset / annotation.classification
        client.download_image(annotation.url, image_folder)


def export_yolo(client, folder: str | PathLike):
    if client.task.project.type != "OBJECT_DETECTION":
        raise RuntimeError("export_yolo only support OBJECT_DETECTION task")

    dataset_path = Path(folder)
    dataset_path.mkdir(parents=True, exist_ok=True)

    with open(dataset_path / "data.yaml", "w", encoding="utf-8") as f:
        datayaml = {
            "path": str(dataset_path.absolute()),
            "train": "images/TRAIN",
            "val": "images/VALID",
            "test": "images/TEST",
            "nc": len(client.task.project.labels),
            "names": client.task.project.labels,
        }
        yaml.dump(datayaml, f, allow_unicode=True, sort_keys=False)

    for annotation in client.dataset:
        images_subset_path = dataset_path / "images" / annotation.subset
        images_subset_path.mkdir(parents=True, exist_ok=True)
        image_path = client.download_image(annotation.url, images_subset_path)

        labels_subset_path = dataset_path / "labels" / annotation.subset
        labels_subset_path.mkdir(parents=True, exist_ok=True)

        label_filename = labels_subset_path / (image_path.stem + ".txt")
        with open(label_filename, "w") as f:
            for box in annotation.object_detection:
                cx = (box.xmin + box.xmax) / 2
                cy = (box.ymin + box.ymax) / 2
                w = box.xmax - box.xmin
                h = box.ymax - box.ymin
                if not (
                    (0 <= cx <= 1)
                    and (0 <= cy <= 1)
                    and (0 <= w <= 1)
                    and (0 <= h <= 1)
                ):
                    raise ValueError(
                        f"(cx, cy, w, h) must be between 0 and 1! cx: {cx}, cy: {cy}, w: {w}, h: {h}"
                    )
                class_id = client.task.project.labels.index(box.label)
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
