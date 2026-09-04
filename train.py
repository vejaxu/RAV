import argparse
import csv
import os
import random
from datetime import datetime

import numpy as np
import torch

from dataloader import load_data
from loss import Loss
from metric import valid
from network import Network


def parse_values(values):
    return [np.float32(value.strip()) for value in values.split(",") if value.strip()]


def format_param(value):
    mantissa, exponent = "{:.0e}".format(float(value)).split("e")
    sign = exponent[0]
    digits = exponent[1:].rjust(2, "0")
    if sign == "-":
        return "{}e-{}".format(mantissa, digits)
    return "{}e{}".format(mantissa, digits)


def param_group(lmd, beta):
    return "lambda_{}_beta_{}".format(format_param(lmd), format_param(beta))


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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, device, view, criterion, loss_fn, phase, lmd, beta):
    total_loss = 0.0
    for xs, _, _ in loader:
        for v in range(view):
            xs[v] = xs[v].to(device, non_blocking=True)
        optimizer.zero_grad()

        if phase == "pretrain":
            _, xrs, _ = model(xs)
            losses = [criterion(xs[v], xrs[v]) for v in range(view)]
        else:
            qs, xrs, zs = model(xs)
            with torch.no_grad():
                view_matrix = model.compute_view_value(zs)
            s_weight, s = model.similarity_matrix(xs, 1.0)
            losses = []
            for v in range(view):
                for w in range(v + 1, view):
                    weight = (view_matrix[v][w] + view_matrix[w][v]) / 2
                    losses.append(lmd * weight * loss_fn.forward_label(qs[v], qs[w]))
                losses.append(beta * loss_fn.forward_feature(s, s_weight[v]))
                losses.append(criterion(xs[v], xrs[v]))

        loss = sum(losses)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


class CudaBatchLoader:
    def __init__(self, dataset, batch_size, device, drop_last=True):
        if not hasattr(dataset, "views"):
            raise ValueError("CUDA batch loader requires dataset.views")
        self.views = [
            torch.from_numpy(np.ascontiguousarray(view)).to(device, non_blocking=True)
            for view in dataset.views
        ]
        self.labels = torch.as_tensor(dataset.y, device=device)
        self.indices = torch.arange(len(dataset), device=device)
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.device = device

    def __len__(self):
        full_batches = len(self.indices) // self.batch_size
        if self.drop_last or len(self.indices) % self.batch_size == 0:
            return full_batches
        return full_batches + 1

    def __iter__(self):
        order = torch.randperm(len(self.indices), device=self.device)
        limit = (len(order) // self.batch_size) * self.batch_size if self.drop_last else len(order)
        for start in range(0, limit, self.batch_size):
            batch_idx = order[start:start + self.batch_size]
            if batch_idx.numel() < self.batch_size and self.drop_last:
                continue
            xs = [view.index_select(0, batch_idx) for view in self.views]
            yield xs, self.labels.index_select(0, batch_idx), batch_idx


_CUDA_LOADER_CACHE = {}


def make_train_loader(dataset, batch_size, device):
    if device.type == "cuda" and hasattr(dataset, "views"):
        key = (id(dataset), batch_size, str(device))
        if key not in _CUDA_LOADER_CACHE:
            _CUDA_LOADER_CACHE[key] = CudaBatchLoader(dataset, batch_size, device, drop_last=True)
        return _CUDA_LOADER_CACHE[key]
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=device.type == "cuda",
    )


def run_param(args, dataset, dims, view, data_size, class_num, lmd, beta):
    group = param_group(lmd, beta)
    result_dir = os.path.join(args.output_dir, args.dataset, group)
    final_path = os.path.join(result_dir, "final_metrics.csv")
    if os.path.exists(final_path):
        print("SKIP {} {}: final_metrics.csv exists".format(args.dataset, group), flush=True)
        return read_final_metrics(final_path, group)

    os.makedirs(result_dir, exist_ok=True)
    log_path = os.path.join(result_dir, "training_log.csv")
    if os.path.exists(log_path):
        os.remove(log_path)

    batch_size = min(args.batch_size, data_size)
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = make_train_loader(dataset, batch_size, device)
    if len(loader) == 0:
        raise ValueError("No training batches for {}".format(args.dataset))
    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    mse = torch.nn.MSELoss()
    loss_fn = Loss(batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

    log_fields = [
        "created_at", "dataset", "phase", "epoch", "loss", "seed", "batch_size",
        "learning_rate", "lambda", "beta", "temperature_f", "temperature_l",
    ]
    for epoch in range(1, args.mse_epochs + 1):
        loss = train_epoch(model, loader, optimizer, device, view, mse, loss_fn, "pretrain", lmd, beta)
        append_csv(log_path, log_fields, log_row(args, "pretrain", epoch, loss, batch_size, lmd, beta))
        print("{} {} pretrain epoch {} loss {:.6f}".format(args.dataset, group, epoch, loss), flush=True)

    for offset in range(1, args.con_epochs + 1):
        epoch = args.mse_epochs + offset
        loss = train_epoch(model, loader, optimizer, device, view, mse, loss_fn, "contrastive", lmd, beta)
        append_csv(log_path, log_fields, log_row(args, "contrastive", epoch, loss, batch_size, lmd, beta))
        print("{} {} contrastive epoch {} loss {:.6f}".format(args.dataset, group, epoch, loss), flush=True)

    nmi, ari, f1, acc, pur, labels, pred, aligned = valid(
        model, device, dataset, view, data_size, class_num, eval_h=False, return_labels=True
    )
    model_path = os.path.join(result_dir, "model.pth")
    torch.save(model.state_dict(), model_path)

    final_fields = [
        "created_at", "dataset", "data_size", "views", "clusters", "dims", "seed", "batch_size",
        "mse_epochs", "con_epochs", "feature_dim", "high_feature_dim", "learning_rate",
        "weight_decay", "lambda", "beta", "temperature_f", "temperature_l", "device",
        "nmi", "ari", "f1", "acc", "pur", "model_path",
    ]
    row = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "data_size": data_size,
        "views": view,
        "clusters": class_num,
        "dims": "/".join(str(dim) for dim in dims),
        "seed": args.seed,
        "batch_size": batch_size,
        "mse_epochs": args.mse_epochs,
        "con_epochs": args.con_epochs,
        "feature_dim": args.feature_dim,
        "high_feature_dim": args.high_feature_dim,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lambda": float(lmd),
        "beta": float(beta),
        "temperature_f": args.temperature_f,
        "temperature_l": args.temperature_l,
        "device": str(device),
        "nmi": nmi,
        "ari": ari,
        "f1": f1,
        "acc": acc,
        "pur": pur,
        "model_path": model_path,
    }
    write_csv(
        os.path.join(result_dir, "labels.csv"),
        ["index", "label", "pred", "aligned_pred"],
        [
            {"index": idx, "label": labels[idx], "pred": pred[idx], "aligned_pred": aligned[idx]}
            for idx in range(labels.shape[0])
        ],
    )
    # final_metrics.csv is the resume marker, so write it only after every
    # required per-run artifact has been persisted successfully.
    write_csv(final_path, final_fields, [row])
    return {
        "Datasets": args.dataset,
        "Points": data_size,
        "Views": view,
        "Dimensions": "/".join(str(dim) for dim in dims),
        "Clusters": class_num,
        "NMI": nmi,
        "ARI": ari,
        "F1": f1,
        "BestLambda": float(lmd),
        "BestBeta": float(beta),
        "ParamGroup": group,
        "ResultDir": result_dir,
    }


def log_row(args, phase, epoch, loss, batch_size, lmd, beta):
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "phase": phase,
        "epoch": epoch,
        "loss": loss,
        "seed": args.seed,
        "batch_size": batch_size,
        "learning_rate": args.learning_rate,
        "lambda": float(lmd),
        "beta": float(beta),
        "temperature_f": args.temperature_f,
        "temperature_l": args.temperature_l,
    }


def read_final_metrics(path, group):
    with open(path, newline="") as f:
        row = next(csv.DictReader(f))
    result_dir = os.path.dirname(path)
    return {
        "Datasets": row["dataset"],
        "Points": row["data_size"],
        "Views": row["views"],
        "Dimensions": row["dims"],
        "Clusters": row["clusters"],
        "NMI": float(row["nmi"]),
        "ARI": float(row["ari"]),
        "F1": float(row["f1"]),
        "BestLambda": float(row["lambda"]),
        "BestBeta": float(row["beta"]),
        "ParamGroup": group,
        "ResultDir": result_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="RAV hyperparameter search")
    parser.add_argument("--dataset", default="ALOI")
    parser.add_argument("--data_dir", default="./data/")
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--summary_csv", default="results.csv")
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--temperature_f", default=0.5, type=float)
    parser.add_argument("--temperature_l", default=0.5, type=float)
    parser.add_argument("--learning_rate", default=0.0003, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--mse_epochs", default=200, type=int)
    parser.add_argument("--con_epochs", default=200, type=int)
    parser.add_argument("--feature_dim", default=512, type=int)
    parser.add_argument("--high_feature_dim", default=128, type=int)
    parser.add_argument("--lambda_values", default="1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100,1000")
    parser.add_argument("--beta_values", default="1e-5,1e-4,1e-3,1e-2,1e-1,1")
    parser.add_argument("--log_interval", default=50, type=int)
    args = parser.parse_args()

    dataset, dims, view, data_size, class_num = load_data(args.dataset, args.data_dir)
    print(
        "Dataset {}: points={}, views={}, clusters={}, dims={}".format(
            args.dataset, data_size, view, class_num, "/".join(str(dim) for dim in dims)
        ),
        flush=True,
    )

    best = None
    for lmd in parse_values(args.lambda_values):
        for beta in parse_values(args.beta_values):
            row = run_param(args, dataset, dims, view, data_size, class_num, lmd, beta)
            if best is None or float(row["NMI"]) > float(best["NMI"]):
                best = row

    fields = ["Datasets", "Points", "Views", "Dimensions", "Clusters", "NMI", "ARI", "F1",
              "BestLambda", "BestBeta", "ParamGroup", "ResultDir"]
    append_csv(args.summary_csv, fields, best)


if __name__ == "__main__":
    main()
