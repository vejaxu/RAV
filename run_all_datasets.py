import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime

from dataloader import load_data


def append_csv(path, fieldnames, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def discover_datasets(data_dir):
    datasets = []
    for name in sorted(os.listdir(data_dir)):
        if name.endswith(".mat"):
            datasets.append(os.path.splitext(name)[0])
    return datasets


def validate_dataset(name, data_dir):
    dataset, dims, view, data_size, class_num = load_data(name, data_dir)
    if view < 2:
        raise ValueError("Need at least two views")
    return {
        "dataset": name,
        "dims": "/".join(str(dim) for dim in dims),
        "view": view,
        "data_size": data_size,
        "class_num": class_num,
    }


def run_one_dataset(args, dataset_name):
    command = [
        sys.executable,
        "train.py",
        "--dataset", dataset_name,
        "--data_dir", args.data_dir,
        "--output_dir", args.output_dir,
        "--summary_csv", args.summary_csv,
        "--batch_size", str(args.batch_size),
        "--mse_epochs", str(args.mse_epochs),
        "--con_epochs", str(args.con_epochs),
        "--seed", str(args.seed),
        "--lambda_values", args.lambda_values,
        "--beta_values", args.beta_values,
        "--log_interval", str(args.log_interval),
    ]
    print("[{}] START {}".format(datetime.now().isoformat(timespec="seconds"), dataset_name), flush=True)
    subprocess.run(command, check=True)
    print("[{}] DONE {}".format(datetime.now().isoformat(timespec="seconds"), dataset_name), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run RAV hyperparameter search for all datasets in a directory.")
    parser.add_argument("--data_dir", default="/home/disk2/zhangh/research/clustering/mvcdatasets")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--summary_csv", default="results.csv")
    parser.add_argument("--datasets", default="all", help="Comma-separated dataset names without .mat, or all.")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--mse_epochs", default=200, type=int)
    parser.add_argument("--con_epochs", default=200, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lambda_values", default="1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000")
    parser.add_argument("--beta_values", default="1e-5,1e-4,1e-3,1e-2,1e-1,1")
    parser.add_argument("--log_interval", default=50, type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    skipped_path = os.path.join(args.output_dir, "skipped_datasets.csv")
    failed_path = os.path.join(args.output_dir, "failed_datasets.csv")

    if not args.resume:
        for path in (args.summary_csv, skipped_path, failed_path):
            if os.path.exists(path):
                os.remove(path)

    if args.datasets == "all":
        datasets = discover_datasets(args.data_dir)
    else:
        datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]

    skipped_fields = ["created_at", "dataset", "reason"]
    failed_fields = ["created_at", "dataset", "returncode", "reason"]

    for dataset_name in datasets:
        try:
            info = validate_dataset(dataset_name, args.data_dir)
            print(
                "Validated {dataset}: points={data_size}, views={view}, clusters={class_num}, dims={dims}".format(**info),
                flush=True,
            )
        except Exception as exc:
            append_csv(skipped_path, skipped_fields, {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "dataset": dataset_name,
                "reason": "{}: {}".format(type(exc).__name__, exc),
            })
            print("SKIP {}: {}".format(dataset_name, exc), flush=True)
            continue

        if args.resume:
            result_dir = os.path.join(args.output_dir, dataset_name)
            completed = 0
            if os.path.isdir(result_dir):
                for _, _, files in os.walk(result_dir):
                    if "final_metrics.csv" in files:
                        completed += 1
            expected = len(args.lambda_values.split(",")) * len(args.beta_values.split(","))
            if completed >= expected:
                print("SKIP {}: already has {} completed parameter groups".format(dataset_name, completed), flush=True)
                continue

        try:
            run_one_dataset(args, dataset_name)
        except subprocess.CalledProcessError as exc:
            append_csv(failed_path, failed_fields, {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "dataset": dataset_name,
                "returncode": exc.returncode,
                "reason": str(exc),
            })
            print("FAILED {}: {}".format(dataset_name, exc), flush=True)


if __name__ == "__main__":
    main()
