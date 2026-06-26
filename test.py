import torch
from network import Network
from metric import valid
import argparse
from dataloader import load_data
import os

Dataname = '3Sources'
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5, type=float)
parser.add_argument("--temperature_l", default=1.0, type=float)
parser.add_argument("--learning_rate", default=0.0003, type=float)
parser.add_argument("--weight_decay", default=0., type=float)
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--mse_epochs", default=200, type=int)
parser.add_argument("--con_epochs", default=50, type=int)
parser.add_argument("--tune_epochs", default=50, type=int)
parser.add_argument("--feature_dim", default=512, type=int)
parser.add_argument("--high_feature_dim", default=128, type=int)
parser.add_argument("--output_dir", default="results")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--data_dir", default="./data/")
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset, dims, view, data_size, class_num = load_data(args.dataset, args.data_dir)
model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
model = model.to(device)
checkpoint_path = args.checkpoint or os.path.join(args.output_dir, args.dataset, "model.pth")
try:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
except TypeError:
    checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)
print("Dataset:{}".format(args.dataset))
print("Datasize:" + str(data_size))
print("Loading model:{}".format(checkpoint_path))
valid(model, device, dataset, view, data_size, class_num, eval_h=False)
