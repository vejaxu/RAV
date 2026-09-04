import argparse
import csv
import glob
import os
import subprocess
import sys
import time
from datetime import datetime

from label_outputs import materialize_row_labels


DATASETS = [
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


def read_csv_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_repeat_rows(root, dataset_order):
    paths = sorted(
        glob.glob(os.path.join(root, "shards", "*", "repeat_metrics.csv"))
        + glob.glob(os.path.join(root, "shards", "*", "seed_*", "repeat_metrics.csv"))
    )
    rows = []
    for path in paths:
        for row in read_csv_rows(path):
            row["source_csv"] = path
            materialize_row_labels(row, root)
            rows.append(row)
    order = {dataset: index for index, dataset in enumerate(dataset_order)}
    return sorted(rows, key=lambda row: (order[row["dataset"]], int(row["seed"])))


def summarize(rows, dataset_order):
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    summaries = []
    for dataset in dataset_order:
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


def refresh_merged_outputs(root, dataset_order):
    rows = collect_repeat_rows(root, dataset_order)
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
    write_csv(os.path.join(root, "repeat_metrics.csv"), repeat_fields, rows)
    write_csv(os.path.join(root, "summary_mean_std.csv"), summary_fields, summarize(rows, dataset_order))


def launch_seed(args, dataset, seed, gpu):
    shard_dir = os.path.join(args.output_dir, "shards", dataset, "seed_{}".format(seed))
    log_path = os.path.join(args.output_dir, "orchestrator_logs", "{}_seed_{}_gpu{}.log".format(dataset, seed, gpu))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    command = [
        sys.executable,
        "run_best_repeats.py",
        "--resume",
        "--datasets",
        dataset,
        "--seeds",
        str(seed),
        "--output_dir",
        shard_dir,
        "--data_dir",
        args.data_dir,
        "--best_csv",
        args.best_csv,
        "--batch_size",
        str(args.batch_size),
        "--mse_epochs",
        str(args.mse_epochs),
        "--con_epochs",
        str(args.con_epochs),
        "--log_interval",
        str(args.log_interval),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = open(log_path, "w")
    log.write("[{}] GPU {} {}\n".format(datetime.now().isoformat(timespec="seconds"), gpu, " ".join(command)))
    log.flush()
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
    return process, log, log_path


def run_dataset(args, dataset, seeds, gpus):
    running = []
    for seed, gpu in zip(seeds, gpus):
        process, log, log_path = launch_seed(args, dataset, seed, gpu)
        running.append((seed, gpu, process, log, log_path))
        print("START {} seed {} on GPU {}".format(dataset, seed, gpu), flush=True)

    failures = []
    while running:
        next_running = []
        for seed, gpu, process, log, log_path in running:
            code = process.poll()
            if code is None:
                next_running.append((seed, gpu, process, log, log_path))
                continue
            log.close()
            if code != 0:
                failures.append((seed, gpu, code, log_path))
                print("FAILED {} seed {} GPU {} code {}".format(dataset, seed, gpu, code), flush=True)
            else:
                print("DONE {} seed {} GPU {}".format(dataset, seed, gpu), flush=True)
        running = next_running
        if running:
            time.sleep(args.poll_seconds)

    if failures:
        for seed, gpu, code, log_path in failures:
            print(
                "Failure detail: dataset={} seed={} gpu={} code={} log={}".format(
                    dataset, seed, gpu, code, log_path
                ),
                flush=True,
            )
        raise RuntimeError("{} failed for {} seed run(s)".format(dataset, len(failures)))


def main():
    parser = argparse.ArgumentParser(description="Run each dataset with five seeds in parallel on five GPUs.")
    parser.add_argument("--output_dir", default="results_best_repeats_gpu")
    parser.add_argument("--data_dir", default="/home/disk2/zhangh/research/clustering/mvcdatasets")
    parser.add_argument("--best_csv", default="results/results.csv")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--seeds", default="42,3407,4079,2024,0")
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--mse_epochs", default=200, type=int)
    parser.add_argument("--con_epochs", default=200, type=int)
    parser.add_argument("--log_interval", default=50, type=int)
    parser.add_argument("--poll_seconds", default=10, type=int)
    args = parser.parse_args()

    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    gpus = [int(item.strip()) for item in args.gpus.split(",") if item.strip()]
    if len(seeds) != len(gpus):
        raise ValueError("Need the same number of seeds and GPUs")

    os.makedirs(args.output_dir, exist_ok=True)
    for dataset in datasets:
        print("DATASET START {}".format(dataset), flush=True)
        run_dataset(args, dataset, seeds, gpus)
        refresh_merged_outputs(args.output_dir, datasets)
        print("DATASET DONE {}".format(dataset), flush=True)

    refresh_merged_outputs(args.output_dir, datasets)
    print("Wrote {}".format(os.path.join(args.output_dir, "summary_mean_std.csv")), flush=True)


if __name__ == "__main__":
    main()
