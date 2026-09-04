import os
import shutil


def labels_output_path(output_dir, dataset, seed):
    return os.path.join(output_dir, "labels", dataset, "seed_{}.csv".format(seed))


def materialize_labels(source_path, output_dir, dataset, seed):
    if not os.path.isfile(source_path):
        raise FileNotFoundError("Missing aligned labels CSV: {}".format(source_path))

    destination = labels_output_path(output_dir, dataset, seed)
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    if os.path.abspath(source_path) != os.path.abspath(destination):
        shutil.copyfile(source_path, destination)
    return destination


def materialize_row_labels(row, output_dir):
    candidates = []
    if row.get("labels_path"):
        candidates.append(row["labels_path"])
    if row.get("result_dir"):
        candidates.append(os.path.join(row["result_dir"], "labels.csv"))

    source_path = next((path for path in candidates if os.path.isfile(path)), None)
    if source_path is None:
        raise FileNotFoundError(
            "Missing aligned labels CSV for dataset={} seed={}; checked {}".format(
                row.get("dataset"), row.get("seed"), ", ".join(candidates) or "no paths"
            )
        )

    row["labels_path"] = materialize_labels(
        source_path,
        output_dir,
        row["dataset"],
        row["seed"],
    )
    return row
