import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime


def discover_datasets(data_dir):
    return sorted(
        os.path.splitext(filename)[0]
        for filename in os.listdir(data_dir)
        if filename.endswith(".mat")
    )


def split_round_robin(items, count):
    groups = [[] for _ in range(count)]
    for index, item in enumerate(items):
        groups[index % count].append(item)
    return groups


def write_csv(path, fieldnames, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def merge_best_params(paths, datasets, output_path):
    by_dataset = {}
    fieldnames = None
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = fieldnames or reader.fieldnames
            for row in reader:
                by_dataset[row["Datasets"]] = row

    missing = [dataset for dataset in datasets if dataset not in by_dataset]
    if missing:
        raise RuntimeError("Parameter search produced no best row for: {}".format(", ".join(missing)))
    write_csv(output_path, fieldnames, [by_dataset[dataset] for dataset in datasets])


def launch_search_worker(args, script_dir, output_root, gpu, datasets):
    shard_dir = os.path.join(output_root, "search_shards", "gpu_{}".format(gpu))
    summary_path = os.path.join(shard_dir, "best_params.csv")
    log_path = os.path.join(output_root, "logs", "search_gpu_{}.log".format(gpu))
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    command = [
        sys.executable,
        os.path.join(script_dir, "run_all_datasets.py"),
        "--resume",
        "--data_dir", args.data_dir,
        "--output_dir", os.path.join(output_root, "search_runs"),
        "--summary_csv", summary_path,
        "--datasets", ",".join(datasets),
        "--batch_size", str(args.batch_size),
        "--mse_epochs", str(args.mse_epochs),
        "--con_epochs", str(args.con_epochs),
        "--seed", str(args.search_seed),
        "--lambda_values", args.lambda_values,
        "--beta_values", args.beta_values,
        "--log_interval", str(args.log_interval),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log = open(log_path, "a")
    log.write("[{}] {}\n".format(datetime.now().isoformat(timespec="seconds"), " ".join(command)))
    log.flush()
    process = subprocess.Popen(command, cwd=script_dir, stdout=log, stderr=subprocess.STDOUT, env=env)
    return process, log, log_path, summary_path


def run_search(args, script_dir, output_root, datasets, gpus):
    groups = split_round_robin(datasets, len(gpus))
    workers = []
    for gpu, group in zip(gpus, groups):
        if not group:
            continue
        process, log, log_path, summary_path = launch_search_worker(
            args, script_dir, output_root, gpu, group
        )
        workers.append((gpu, group, process, log, log_path, summary_path))
        print("SEARCH START GPU {}: {}".format(gpu, ",".join(group)), flush=True)

    failures = []
    while workers:
        running = []
        for gpu, group, process, log, log_path, summary_path in workers:
            code = process.poll()
            if code is None:
                running.append((gpu, group, process, log, log_path, summary_path))
                continue
            log.close()
            if code:
                failures.append((gpu, code, log_path))
                print("SEARCH FAILED GPU {} code {} log {}".format(gpu, code, log_path), flush=True)
            else:
                print("SEARCH DONE GPU {}".format(gpu), flush=True)
        workers = running
        if workers:
            time.sleep(args.poll_seconds)

    if failures:
        raise RuntimeError("{} parameter-search worker(s) failed".format(len(failures)))
    return [
        os.path.join(output_root, "search_shards", "gpu_{}".format(gpu), "best_params.csv")
        for gpu in gpus
    ]


def run_repeats(args, script_dir, output_root, datasets, best_params_path):
    command = [
        sys.executable,
        os.path.join(script_dir, "run_dataset_repeats_parallel.py"),
        "--output_dir", os.path.join(output_root, "repeats"),
        "--data_dir", args.data_dir,
        "--best_csv", best_params_path,
        "--datasets", ",".join(datasets),
        "--seeds", args.seeds,
        "--gpus", args.repeat_gpus,
        "--batch_size", str(args.batch_size),
        "--mse_epochs", str(args.mse_epochs),
        "--con_epochs", str(args.con_epochs),
        "--log_interval", str(args.log_interval),
        "--poll_seconds", str(args.poll_seconds),
    ]
    log_path = os.path.join(output_root, "logs", "repeats.log")
    with open(log_path, "a") as log:
        log.write("[{}] {}\n".format(datetime.now().isoformat(timespec="seconds"), " ".join(command)))
        log.flush()
        subprocess.run(command, cwd=script_dir, check=True, stdout=log, stderr=subprocess.STDOUT)


def main():
    parser = argparse.ArgumentParser(
        description="Search RAV parameters for every MAT file in a directory, then run repeated seeds."
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--search_gpus", default="0,1,2,3,4,5")
    parser.add_argument("--repeat_gpus", default="0,1,2,3,4")
    parser.add_argument("--seeds", default="42,3407,4079,2024,0")
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--mse_epochs", default=200, type=int)
    parser.add_argument("--con_epochs", default=200, type=int)
    parser.add_argument("--search_seed", default=42, type=int)
    parser.add_argument("--lambda_values", default="1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000")
    parser.add_argument("--beta_values", default="1e-5,1e-4,1e-3,1e-2,1e-1,1")
    parser.add_argument("--log_interval", default=50, type=int)
    parser.add_argument("--poll_seconds", default=10, type=int)
    args = parser.parse_args()

    args.data_dir = os.path.abspath(args.data_dir)
    output_root = os.path.abspath(args.output_dir)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    datasets = discover_datasets(args.data_dir) if args.datasets == "all" else [
        item.strip() for item in args.datasets.split(",") if item.strip()
    ]
    search_gpus = [int(item.strip()) for item in args.search_gpus.split(",") if item.strip()]
    seeds = [item.strip() for item in args.seeds.split(",") if item.strip()]
    repeat_gpus = [item.strip() for item in args.repeat_gpus.split(",") if item.strip()]
    if not datasets:
        raise ValueError("No datasets selected")
    if not search_gpus:
        raise ValueError("No search GPUs selected")
    if len(seeds) != len(repeat_gpus):
        raise ValueError("Need the same number of repeat seeds and repeat GPUs")

    os.makedirs(output_root, exist_ok=True)
    print("PIPELINE DATASETS {}".format(",".join(datasets)), flush=True)
    shard_summaries = run_search(args, script_dir, output_root, datasets, search_gpus)
    best_params_path = os.path.join(output_root, "best_params.csv")
    merge_best_params(shard_summaries, datasets, best_params_path)
    print("WROTE {}".format(best_params_path), flush=True)
    run_repeats(args, script_dir, output_root, datasets, best_params_path)
    print("PIPELINE COMPLETE {}".format(output_root), flush=True)


if __name__ == "__main__":
    main()
