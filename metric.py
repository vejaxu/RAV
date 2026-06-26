import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score, f1_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def hungarian_match_labels(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    assert y_pred.size == y_true.size

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)
    w = np.zeros((pred_labels.size, true_labels.size), dtype=np.int64)
    pred_index = {label: idx for idx, label in enumerate(pred_labels)}
    true_index = {label: idx for idx, label in enumerate(true_labels)}

    for pred, true in zip(y_pred, y_true):
        w[pred_index[pred], true_index[true]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    label_map = {
        pred_labels[row]: true_labels[col]
        for row, col in zip(row_ind, col_ind)
    }

    mapped_pred = np.empty_like(y_pred)
    for pred_label in pred_labels:
        mask = y_pred == pred_label
        if pred_label in label_map:
            mapped_pred[mask] = label_map[pred_label]
        else:
            counts = np.bincount(y_true[mask])
            mapped_pred[mask] = counts.argmax()
    return mapped_pred


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    aligned_pred = hungarian_match_labels(label, pred)
    f1 = f1_score(label, aligned_pred, average="macro")
    return nmi, ari, f1, acc, pur, aligned_pred


def inference(loader, model, device, view, data_size):
    model.eval()
    soft_vector = []
    pred_vectors = []
    labels_vector = []
    for v in range(view):
        pred_vectors.append([])

    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            qs, _, zs = model(xs)
            q = sum(qs)/view
        for v in range(view):
            pred_label = torch.argmax(qs[v], dim=1)
            pred_vectors[v].extend(pred_label.cpu().detach().numpy())
        q = q.detach()
        soft_vector.extend(q.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
    for v in range(view):
        pred_vectors[v] = np.array(pred_vectors[v])

    labels_vector = np.array(labels_vector).reshape(data_size)
    total_pred = np.argmax(np.array(soft_vector), axis=1)
    return total_pred, pred_vectors, labels_vector


def valid(model, device, dataset, view, data_size, class_num, eval_h=False, return_labels=False):
    test_loader = DataLoader(
            dataset,
            batch_size=data_size, #256
            shuffle=False,
        )
    total_pred, pred_vectors, labels_vector = inference(test_loader, model, device, view, data_size)
    print("Clustering results on semantic labels: " + str(labels_vector.shape[0]))
    nmi, ari, f1, acc, pur, aligned_pred = evaluate(labels_vector, total_pred)
    print('NMI = {:.4f} ARI = {:.4f} F1 = {:.4f} ACC = {:.4f} PUR={:.4f}'.format(nmi, ari, f1, acc, pur))
    if return_labels:
        return nmi, ari, f1, acc, pur, labels_vector, total_pred, aligned_pred
    return nmi, ari, f1, acc, pur


