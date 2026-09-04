import argparse
import csv
import glob
import os

from label_outputs import materialize_row_labels


DATASET_ORDER = [
    "ORL",
    "YaleB",
    "flower17",
    "COIL20",
    "Caltech101-7",
    "100leaves",
    "Mfeat",
    "UCI_Digits",
    "NTU2012_mvcnn_gvcnn",
    "MNIST",
    "animal",
    "ALOI",
    "VGGFace2-50",
    "CIFAR10_llc_with_img_fea",
    "Food-101",
]


def read_rows(paths, output_dir):
    rows = []
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                row["source_csv"] = path
                materialize_row_labels(row, output_dir)
                rows.append(row)
    return rows


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    summaries = []
    for dataset in DATASET_ORDER:
        dataset_rows = by_dataset.get(dataset, [])
        if not dataset_rows:
            continue
        dataset_rows = sorted(dataset_rows, key=lambda row: int(row["seed"]))
        first = dataset_rows[0]
        summary = {
            "dataset": dataset,
            "runs": len(dataset_rows),
            "best_lambda": first["best_lambda"],
            "best_beta": first["best_beta"],
            "param_group": first["param_group"],
        }
        for metric in ("nmi", "ari", "f1", "runtime_seconds"):
            values = [float(row[metric]) for row in dataset_rows]
            mean = sum(values) / len(values)
            if len(values) > 1:
                variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
                std = variance ** 0.5
            else:
                std = 0.0
            summary["{}_mean".format(metric)] = mean
            summary["{}_std".format(metric)] = std
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser(description="Merge per-GPU best-repeat shard CSV files.")
    parser.add_argument("--root", default="results_best_repeats_gpu")
    args = parser.parse_args()

    repeat_paths = sorted(
        glob.glob(os.path.join(args.root, "shards", "*", "repeat_metrics.csv"))
        + glob.glob(os.path.join(args.root, "shards", "*", "seed_*", "repeat_metrics.csv"))
    )
    rows = read_rows(repeat_paths, args.root)
    rows = sorted(rows, key=lambda row: (DATASET_ORDER.index(row["dataset"]), int(row["seed"])))

    repeat_fields = [
        "created_at", "dataset", "seed", "best_lambda", "best_beta", "param_group",
        "nmi", "ari", "f1", "acc", "pur", "runtime_seconds", "labels_path",
        "result_dir", "log_path", "source_csv",
    ]
    summary_fields = [
        "dataset", "runs", "best_lambda", "best_beta", "param_group",
        "nmi_mean", "nmi_std", "ari_mean", "ari_std", "f1_mean", "f1_std",
        "runtime_seconds_mean", "runtime_seconds_std",
    ]

    write_csv(os.path.join(args.root, "repeat_metrics.csv"), repeat_fields, rows)
    write_csv(os.path.join(args.root, "summary_mean_std.csv"), summary_fields, summarize(rows))
    print("Merged {} repeat rows from {} shard files".format(len(rows), len(repeat_paths)))


if __name__ == "__main__":
    main()
