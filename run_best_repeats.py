import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime


TARGET_DATASETS = [
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


def read_best_params(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return {row["Datasets"]: row for row in rows}


def append_csv(path, fieldnames, row):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_single_final_metrics(path):
    with open(path, newline="") as f:
        return next(csv.DictReader(f))


def format_float(value):
    return "{:.12g}".format(float(value))


def run_repeat(args, dataset, best_row, seed):
    repeat_output_dir = os.path.join(args.output_dir, "runs", dataset, "seed_{}".format(seed))
    repeat_summary = os.path.join(repeat_output_dir, "best_summary.csv")
    log_path = os.path.join(args.output_dir, "logs", "{}_seed_{}.log".format(dataset, seed))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    command = [
        sys.executable,
        "train.py",
        "--dataset",
        dataset,
        "--data_dir",
        args.data_dir,
        "--output_dir",
        repeat_output_dir,
        "--summary_csv",
        repeat_summary,
        "--batch_size",
        str(args.batch_size),
        "--mse_epochs",
        str(args.mse_epochs),
        "--con_epochs",
        str(args.con_epochs),
        "--seed",
        str(seed),
        "--lambda_values",
        format_float(best_row["BestLambda"]),
        "--beta_values",
        format_float(best_row["BestBeta"]),
        "--log_interval",
        str(args.log_interval),
    ]

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    start = time.perf_counter()
    with open(log_path, "w") as log:
        log.write("[{}] {}\n".format(datetime.now().isoformat(timespec="seconds"), " ".join(command)))
        log.flush()
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
    elapsed = time.perf_counter() - start

    param_group = best_row["ParamGroup"]
    final_path = os.path.join(repeat_output_dir, dataset, param_group, "final_metrics.csv")
    metrics = read_single_final_metrics(final_path)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": dataset,
        "seed": seed,
        "best_lambda": best_row["BestLambda"],
        "best_beta": best_row["BestBeta"],
        "param_group": param_group,
        "nmi": float(metrics["nmi"]),
        "ari": float(metrics["ari"]),
        "f1": float(metrics["f1"]),
        "acc": float(metrics["acc"]),
        "pur": float(metrics["pur"]),
        "runtime_seconds": elapsed,
        "result_dir": os.path.dirname(final_path),
        "log_path": log_path,
    }


def summarize(rows):
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)

    summaries = []
    for dataset in TARGET_DATASETS:
        dataset_rows = by_dataset.get(dataset, [])
        if not dataset_rows:
            continue
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


def existing_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description="Run five repeats with the best RAV parameters from results.csv.")
    parser.add_argument("--data_dir", default="/home/disk2/zhangh/research/clustering/mvcdatasets")
    parser.add_argument("--best_csv", default="results/results.csv")
    parser.add_argument("--output_dir", default="results_best_repeats")
    parser.add_argument("--datasets", default=",".join(TARGET_DATASETS))
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--mse_epochs", default=200, type=int)
    parser.add_argument("--con_epochs", default=200, type=int)
    parser.add_argument("--log_interval", default=50, type=int)
    parser.add_argument("--cuda_visible_devices", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
    best_rows = read_best_params(args.best_csv)
    missing = [dataset for dataset in datasets if dataset not in best_rows]
    if missing:
        raise ValueError("Missing best parameters for: {}".format(", ".join(missing)))

    os.makedirs(args.output_dir, exist_ok=True)
    repeat_csv = os.path.join(args.output_dir, "repeat_metrics.csv")
    summary_csv = os.path.join(args.output_dir, "summary_mean_std.csv")

    repeat_fields = [
        "created_at", "dataset", "seed", "best_lambda", "best_beta", "param_group",
        "nmi", "ari", "f1", "acc", "pur", "runtime_seconds", "result_dir", "log_path",
    ]
    summary_fields = [
        "dataset", "runs", "best_lambda", "best_beta", "param_group",
        "nmi_mean", "nmi_std", "ari_mean", "ari_std", "f1_mean", "f1_std",
        "runtime_seconds_mean", "runtime_seconds_std",
    ]

    rows = existing_rows(repeat_csv) if args.resume else []
    completed = {(row["dataset"], int(row["seed"])) for row in rows}
    if not args.resume:
        for path in (repeat_csv, summary_csv):
            if os.path.exists(path):
                os.remove(path)

    for dataset in datasets:
        for seed in seeds:
            if args.resume and (dataset, seed) in completed:
                print("SKIP {} seed {}: already recorded".format(dataset, seed), flush=True)
                continue
            print("START {} seed {}".format(dataset, seed), flush=True)
            row = run_repeat(args, dataset, best_rows[dataset], seed)
            append_csv(repeat_csv, repeat_fields, row)
            rows.append(row)
            write_csv(summary_csv, summary_fields, summarize(rows))
            print(
                "DONE {} seed {} NMI {:.4f} ARI {:.4f} F1 {:.4f} time {:.2f}s".format(
                    dataset, seed, row["nmi"], row["ari"], row["f1"], row["runtime_seconds"]
                ),
                flush=True,
            )

    write_csv(summary_csv, summary_fields, summarize(rows))
    print("Wrote {}".format(summary_csv), flush=True)


if __name__ == "__main__":
    main()
