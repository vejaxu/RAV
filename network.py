import torch.nn as nn
from torch.nn.functional import normalize
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)


        self.label_contrastive_module = nn.Sequential( #软标签，软分配从Z->Q
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.view = view

    def forward(self, xs):
        qs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z =self.encoders[v](x)
            q = self.label_contrastive_module(z)
            xr = self.decoders[v](z)
            zs.append(z)
            qs.append(q)
            xrs.append(xr)
        return qs, xrs, zs

    def similarity_matrix(self, xs, sigma=1.0):
        S_weight = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            distance_sq = torch.cdist(z, z, p=2).pow(2)
            w = torch.exp(-distance_sq / sigma)
            S_weight.append(w)
            zs.append(z)
        common_z = torch.cat(zs,dim=1)
        distance = torch.cdist(common_z, common_z, p=2).pow(2)
        S = torch.exp(-distance / sigma)
        return S_weight,S

    def compute_view_value(self,zs):
        normalized_zs = [normalize(z, dim=1).flatten() for z in zs]
        w = torch.empty((self.view, self.view), device=zs[0].device, dtype=zs[0].dtype)
        for i in range(self.view):
            zi = torch.sort(normalized_zs[i]).values
            for j in range(i, self.view):
                zj = torch.sort(normalized_zs[j]).values
                distance = torch.mean(torch.abs(zi - zj))
                value = torch.exp(-distance)
                w[i][j] = value
                w[j][i] = value
        return w / w.sum(dim=1, keepdim=True)
