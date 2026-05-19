import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import Loss
from dataloader import load_data
import os


Dataname = 'ALOI'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--temperature_f", default=0.5)
parser.add_argument("--temperature_l", default=0.5)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--workers", default=8)
parser.add_argument('--seed', type=int, default=10)
parser.add_argument("--mse_epochs", default=200)
parser.add_argument("--con_epochs", default=200)
parser.add_argument("--feature_dim", default=512)
parser.add_argument("--high_feature_dim", default=128)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# The code has been optimized.
# The seed was fixed for the performance reproduction, which was higher than the values shown in the paper.

if args.dataset == "ALOI":
    args.con_epochs = 200
    seed = 5
if args.dataset == "NGs":
    args.con_epochs = 200
    seed = 5

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(seed)


dataset, dims, view, data_size, class_num = load_data(args.dataset)

data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )


def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        _, xrs, _ = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))


def contrastive_train(epoch,lmd,beta):
    tot_loss = 0.
    mes = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        qs, xrs, zs = model(xs)
        with torch.no_grad():
            view_matrix = model.compute_view_value(zs)
        S_weight, S = model.similarity_matrix(xs, 1.0)
        loss_list = []
        for v in range(view):
            for w in range(v+1, view):
                loss_list.append(lmd * (((view_matrix[v][w]+view_matrix[w][v])/2)* criterion.forward_label(qs[v], qs[w])) )
            loss_list.append(beta * (criterion.forward_feature(S, S_weight[v])))
            loss_list.append(mes(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))


accs = []
nmis = []
purs = []
if not os.path.exists('./models'):
    os.makedirs('./models')


T = 1
for i in range(T):
    lamda = np.array([0.01], dtype=np.float32)
    betas = np.array([0.01], dtype=np.float32)
    seeds = np.array([10], dtype=np.int32)
    for lmd_idx in range(lamda.shape[0]):
        lmd = lamda[lmd_idx]
        for beta_idx in range(betas.shape[0]):
            beta = betas[beta_idx]
            for seed_idx in range(seeds.shape[0]):
                args.seed = seeds[seed_idx]
                print("ROUND:{}".format(i+1))
                setup_seed(args.seed)
                model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
                print(model)
                model = model.to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
                criterion = Loss(args.batch_size, class_num, args.temperature_f, args.temperature_l, device).to(device)

                epoch = 1
                while epoch <= args.mse_epochs:
                    pretrain(epoch)
                    epoch += 1
                while epoch <= args.mse_epochs + args.con_epochs:
                    contrastive_train(epoch,lmd,beta)
                    if epoch == args.mse_epochs + args.con_epochs:
                        acc, nmi, pur,ari = valid(model, device, dataset, view, data_size, class_num, eval_h=False)
                        state = model.state_dict()
                        torch.save(state, './models/' + args.dataset+ '1' + '.pth')
                        print('Saving..')
                        accs.append(acc)
                        nmis.append(nmi)
                        purs.append(pur)
                        # with torch.no_grad():
                        #     common,y= model.common_z(data_loader,device)
                        #     tSNE_PLOT(common[0:1000,:].cpu().detach().numpy(),y[0:1000],args.dataset +"_K_"+str(class_num)+"_")
                        with open(args.dataset + '_result2.txt', 'a+') as f:
                            f.write(
                                '{} \t {} \t {} \t {} \t {} \t {:.5f} \t {:.5f} \t {:.3f} \t {:.3f} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \n'.format(
                                    args.high_feature_dim, args.feature_dim, args.seed, args.batch_size,
                                    args.learning_rate, lmd, beta, args.temperature_f, args.temperature_l, acc,
                                    nmi, pur,ari))
                            f.flush()
                    epoch += 1


