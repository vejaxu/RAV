
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch.utils.data import Dataset
import scipy.io
import torch


class BDGP(Dataset):
    def __init__(self, path):
        data1 = scipy.io.loadmat(path+'BDGP.mat')['X1'].astype(np.float32)
        data2 = scipy.io.loadmat(path+'BDGP.mat')['X2'].astype(np.float32)
        labels = scipy.io.loadmat(path+'BDGP.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()

class CCV(Dataset):
    def __init__(self, path):
        self.data1 = np.load(path+'STIP.npy').astype(np.float32)
        scaler = MinMaxScaler()
        self.data1 = scaler.fit_transform(self.data1)
        self.data2 = np.load(path+'SIFT.npy').astype(np.float32)
        self.data3 = np.load(path+'MFCC.npy').astype(np.float32)
        self.labels = np.load(path+'label.npy')

    def __len__(self):
        return 6773

    def __getitem__(self, idx):
        x1 = self.data1[idx]
        x2 = self.data2[idx]
        x3 = self.data3[idx]

        return [torch.from_numpy(x1), torch.from_numpy(
           x2), torch.from_numpy(x3)], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()


class MNIST_USPS(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'MNIST_USPS.mat')['Y'].astype(np.int32).reshape(5000,)
        self.V1 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'MNIST_USPS.mat')['X2'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Fashion(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion.mat')['Y'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion.mat')['X3'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class Fashion_new(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Fashion_new.mat')['Y_new'].astype(np.int32).reshape(10000,)
        self.V1 = scipy.io.loadmat(path + 'Fashion_new.mat')['X1_new'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Fashion_new.mat')['X2_new'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Fashion_new.mat')['X3_new'].astype(np.float32)

    def __len__(self):
        return 10000

    def __getitem__(self, idx):

        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        x3 = self.V3[idx].reshape(784)

        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(np.array(idx)).long()


class Caltech(Dataset):
    def __init__(self, path, view):
        data = scipy.io.loadmat(path)
        scaler = MinMaxScaler()
        self.view1 = scaler.fit_transform(data['X1'].astype(np.float32))
        self.view2 = scaler.fit_transform(data['X2'].astype(np.float32))
        self.view3 = scaler.fit_transform(data['X3'].astype(np.float32))
        self.view4 = scaler.fit_transform(data['X4'].astype(np.float32))
        self.view5 = scaler.fit_transform(data['X5'].astype(np.float32))
        self.labels = scipy.io.loadmat(path)['Y'].transpose()
        self.view = view

    def __len__(self):
        return 1400

    def __getitem__(self, idx):
        if self.view == 2:
            return [torch.from_numpy(
                self.view1[idx]), torch.from_numpy(self.view2[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 3:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 4:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(self.view2[idx]), torch.from_numpy(
                self.view5[idx]), torch.from_numpy(self.view4[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()
        if self.view == 5:
            return [torch.from_numpy(self.view1[idx]), torch.from_numpy(
                self.view2[idx]), torch.from_numpy(self.view5[idx]), torch.from_numpy(
                self.view4[idx]), torch.from_numpy(self.view3[idx])], torch.from_numpy(self.labels[idx]), torch.from_numpy(np.array(idx)).long()

class NUSWIDE(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'NUSWIDE.mat')['Y'].astype(np.int32).reshape(5000, )
        self.V1 = scipy.io.loadmat(path + 'NUSWIDE.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'NUSWIDE.mat')['X2'].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'NUSWIDE.mat')['X3'].astype(np.float32)
        self.V4 = scipy.io.loadmat(path + 'NUSWIDE.mat')['X4'].astype(np.float32)
        self.V5 = scipy.io.loadmat(path + 'NUSWIDE.mat')['X5'].astype(np.float32)

    def __len__(self):
        return 5000

    def __getitem__(self, idx):

        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]), torch.from_numpy(self.V3[idx]), torch.from_numpy(self.V4[idx]), torch.from_numpy(self.V5[idx])], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

class Synthetic3d(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'synthetic3d.mat')['Y'].astype(np.int32).reshape(600, )
        self.V1 = scipy.io.loadmat(path + 'synthetic3d.mat')['X'][0][0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'synthetic3d.mat')['X'][1][0].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'synthetic3d.mat')['X'][2][0].astype(np.float32)

    def __len__(self):
        return 600

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]), torch.from_numpy(self.V3[idx])
               ], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

class Hdigit(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Hdigit.mat')['truelabel'][0][0].astype(np.int32).reshape(10000, )
        self.V1 = scipy.io.loadmat(path + 'Hdigit.mat')['data'][0][0].T.astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Hdigit.mat')['data'][0][1].T.astype(np.float32)


    def __len__(self):
        return 10000

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx])
               ], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

class RGB(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'RGB-D.mat')['Y'].astype(np.int32).reshape(1449, )
        self.V1 = scipy.io.loadmat(path + 'RGB-D.mat')['X1'].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'RGB-D.mat')['X2'].astype(np.float32)


    def __len__(self):
        return 1449

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx])
               ], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()
class Scene(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Scene15.mat')['Y'].astype(np.int32).reshape(4485, )
        self.V1 = scipy.io.loadmat(path + 'Scene15.mat')['X'][0][0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Scene15.mat')['X'][0][1].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Scene15.mat')['X'][0][2].astype(np.float32)


    def __len__(self):
        return 4485

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),torch.from_numpy(self.V3[idx])
               ], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

class YTF10(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'YTF10.mat')['Y'].astype(np.int32).reshape(38654, )
        self.V1 = scipy.io.loadmat(path + 'YTF10.mat')['X'][0][0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'YTF10.mat')['X'][1][0].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'YTF10.mat')['X'][2][0].astype(np.float32)
        self.V4 = scipy.io.loadmat(path + 'YTF10.mat')['X'][3][0].astype(np.float32)

    def __len__(self):
        return 38654

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),torch.from_numpy(self.V3[idx]),torch.from_numpy(self.V4[idx])
               ], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

class Hand(Dataset):
    def __init__(self, path):
        scaler = MinMaxScaler()
        self.Y = (scipy.io.loadmat(path + 'handwritten.mat')['Y']+1).astype(np.int32).reshape(2000, )
        self.V1 = scaler.fit_transform( scipy.io.loadmat(path + 'handwritten.mat')['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform( scipy.io.loadmat(path + 'handwritten.mat')['X'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform( scipy.io.loadmat(path + 'handwritten.mat')['X'][0][2].astype(np.float32))
        self.V4 = scaler.fit_transform(scipy.io.loadmat(path + 'handwritten.mat')['X'][0][3].astype(np.float32))
        self.V5 = scaler.fit_transform(scipy.io.loadmat(path + 'handwritten.mat')['X'][0][4].astype(np.float32))
        self.V6 = scaler.fit_transform(scipy.io.loadmat(path + 'handwritten.mat')['X'][0][5].astype(np.float32))

    def __len__(self):
        return 2000

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),torch.from_numpy(self.V3[idx]),torch.from_numpy(self.V4[idx])
               ,torch.from_numpy(self.V5[idx]),torch.from_numpy(self.V6[idx])] , self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

class ALOI(Dataset):
    def __init__(self, path):
        scaler = MinMaxScaler()
        self.Y = (scipy.io.loadmat(path + 'ALOI-100.mat')['Y']).astype(np.int32).reshape(10800, )
        self.V1 = scaler.fit_transform( scipy.io.loadmat(path + 'ALOI-100.mat')['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform( scipy.io.loadmat(path + 'ALOI-100.mat')['X'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform( scipy.io.loadmat(path + 'ALOI-100.mat')['X'][0][2].astype(np.float32))
        self.V4 = scaler.fit_transform(scipy.io.loadmat(path + 'ALOI-100.mat')['X'][0][3].astype(np.float32))

    def __len__(self):
        return 10800

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),torch.from_numpy(self.V3[idx]),torch.from_numpy(self.V4[idx])
        ] , self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

class Digit_Product(Dataset):
    def __init__(self, path):
        mat = scipy.io.loadmat(path + 'Digit-Product.mat')
        scaler = MinMaxScaler()
        self.Y =  np.array(np.squeeze(mat['Y'])).astype(np.int32).reshape(30000, )
        self.V1 = scaler.fit_transform( mat['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform( mat['X'][0][1].astype(np.float32))

    def __len__(self):
        return 30000

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx])
        ] , self.Y[idx], torch.from_numpy(
            np.array(idx)).long()


class Cora(Dataset):
    def __init__(self, path):
        mat = scipy.io.loadmat(path + 'Cora.mat')
        scaler = MinMaxScaler()
        self.Y = np.array(np.squeeze(mat['Y'])).astype(np.int32).reshape(2708, )
        self.V1 = scaler.fit_transform(mat['X'][0][0].astype(np.float32))
        self.V2 = scaler.fit_transform(mat['X'][0][1].astype(np.float32))
        self.V3 = scaler.fit_transform(mat['X'][0][2].astype(np.float32))
        self.V4 = scaler.fit_transform(mat['X'][0][3].astype(np.float32))
    def __len__(self):
        return 2708

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]),torch.from_numpy(self.V3[idx]),torch.from_numpy(self.V4[idx])
        ] , self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

class NGs(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'NGs.mat')['Y'].astype(np.int32).reshape(500, )
        x = scipy.io.loadmat(path + 'NGs.mat')['X']
        self.V1 = x[0][0].astype(np.float32)
        self.V2 = x[1][0].astype(np.float32)
        self.V3 = x[2][0].astype(np.float32)

    def __len__(self):
        return 500

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(2000)
        x2 = self.V2[idx].reshape(2000)
        x3 = self.V3[idx].reshape(2000)
        return [torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(x3)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

class NoisyMNIST(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'NoisyMNIST.mat')['trainLabel'].astype(np.int32).reshape(50000, )
        x = scipy.io.loadmat(path + 'NoisyMNIST.mat')
        self.V1 = x['X1'].astype(np.float32)
        self.V2 = x['X2'].astype(np.float32)

    def __len__(self):
        return 50000

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(784)
        x2 = self.V2[idx].reshape(784)
        return [torch.from_numpy(x1), torch.from_numpy(x2)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()

class Caltech101_20(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + 'Caltech101_20.mat')['Y'].astype(np.int32).reshape(2386,)
        self.V1 = scipy.io.loadmat(path + 'Caltech101_20.mat')['X'][0][0].astype(np.float32)
        self.V2 = scipy.io.loadmat(path + 'Caltech101_20.mat')['X'][0][1].astype(np.float32)
        self.V3 = scipy.io.loadmat(path + 'Caltech101_20.mat')['X'][0][2].astype(np.float32)
        self.V4 = scipy.io.loadmat(path + 'Caltech101_20.mat')['X'][0][3].astype(np.float32)
        self.V5 = scipy.io.loadmat(path + 'Caltech101_20.mat')['X'][0][4].astype(np.float32)
        self.V6 = scipy.io.loadmat(path + 'Caltech101_20.mat')['X'][0][5].astype(np.float32)


    def __len__(self):
        return 2386

    def __getitem__(self, idx):
        x1 = self.V1[idx].reshape(48)
        x2 = self.V2[idx].reshape(40)
        x3 = self.V3[idx].reshape(254)
        x4 = self.V4[idx].reshape(1984)
        x5 = self.V5[idx].reshape(512)
        x6 = self.V6[idx].reshape(928)
        return [torch.from_numpy(x1), torch.from_numpy(x2),torch.from_numpy(x3), torch.from_numpy(x4),torch.from_numpy(x5), torch.from_numpy(x6)], self.Y[idx], torch.from_numpy(
            np.array(idx)).long()


class ThreeSources(Dataset):
    def __init__(self, path):
        self.Y = scipy.io.loadmat(path + '3sources.mat')['truelabel'][0][0].astype(np.int32).reshape(169,)
        self.V1 = scipy.io.loadmat(path + '3sources.mat')['data'][0][0].T.astype(np.float32)
        self.V2 = scipy.io.loadmat(path + '3sources.mat')['data'][0][1].T.astype(np.float32)
        self.V3 = scipy.io.loadmat(path + '3sources.mat')['data'][0][2].T.astype(np.float32)

    def __len__(self):
        return 169

    def __getitem__(self, idx):
        return [torch.from_numpy(self.V1[idx]), torch.from_numpy(self.V2[idx]), torch.from_numpy(self.V3[idx])
                ], self.Y[idx], torch.from_numpy(np.array(idx)).long()

class YouTubeVideo(Dataset):
    def __init__(self, path):
        scaler = MinMaxScaler()
        data1 = scaler.fit_transform(scipy.io.loadmat(path+'Video-3V.mat')['X1'].astype(np.float32))
        data2 = scaler.fit_transform(scipy.io.loadmat(path+'Video-3V.mat')['X2'].astype(np.float32))
        data3 = scaler.fit_transform(scipy.io.loadmat(path + 'Video-3V.mat')['X3'].astype(np.float32))
        labels = scipy.io.loadmat(path+'Video-3V.mat')['Y'].transpose()
        self.x1 = data1
        self.x2 = data2
        self.x3 = data3
        self.y = labels

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return [torch.from_numpy(self.x1[idx]), torch.from_numpy(
           self.x2[idx]), torch.from_numpy(
           self.x3[idx])], torch.from_numpy(self.y[idx]), torch.from_numpy(np.array(idx)).long()



def load_data(dataset):
    if dataset == "BDGP":
        dataset = BDGP('./data/')
        dims = [1750, 79]
        view = 2
        data_size = 2500
        class_num = 5
    elif dataset == "MNIST-USPS":
        dataset = MNIST_USPS('./data/')
        dims = [784, 784]
        view = 2
        class_num = 10
        data_size = 5000
    elif dataset == "CCV":
        dataset = CCV('./data/')
        dims = [5000, 5000, 4000]
        view = 3
        data_size = 6773
        class_num = 20
    elif dataset == "Fashion":
        dataset = Fashion('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Fashion_new":
        dataset = Fashion_new('./data/')
        dims = [784, 784, 784]
        view = 3
        data_size = 10000
        class_num = 10
    elif dataset == "Caltech-2V":
        dataset = Caltech('data/Caltech-5V.mat', view=2)
        dims = [40, 254]
        view = 2
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-3V":
        dataset = Caltech('data/Caltech-5V.mat', view=3)
        dims = [40, 254, 928]
        view = 3
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-4V":
        dataset = Caltech('data/Caltech-5V.mat', view=4)
        dims = [40, 254, 928, 512]
        view = 4
        data_size = 1400
        class_num = 7
    elif dataset == "Caltech-5V":
        dataset = Caltech('data/Caltech-5V.mat', view=5)
        dims = [40, 254, 928, 512, 1984]
        view = 5
        data_size = 1400
        class_num = 7
    elif dataset == "NUSWIDE":
        dataset = NUSWIDE('./data/')
        dims = [65,226,145,74,129]
        view = 5
        data_size = 5000
        class_num = 5
    elif dataset == "Synthetic3d":
        dataset = Synthetic3d('./data/')
        dims = [3,3,3]
        view = 3
        data_size = 600
        class_num = 3
    elif dataset == "Hdigit":
        dataset = Hdigit('./data/')
        dims = [784,256]
        view = 2
        data_size = 10000
        class_num = 10
    elif dataset == "RGB-D":
        dataset = RGB('./data/')
        dims = [2048,300]
        view = 2
        data_size = 1449
        class_num = 13
    elif dataset == "scene":
        dataset = Scene('./data/')
        dims = [20,59,40]
        view = 3
        data_size = 4485
        class_num = 15
    elif dataset == "YTF10":
        dataset = YTF10('./data/')
        dims = [944,576, 512, 640]
        view = 4
        data_size = 38654
        class_num = 10
    elif dataset == "Hand":
        dataset = Hand('./data/')
        dims = [240, 76, 216, 47 , 64, 6]
        view = 6
        data_size = 2000
        class_num = 10
    elif dataset == "ALOI":
        dataset = ALOI('./data/')
        dims = [77,13,64,125]
        view = 4
        data_size = 10800
        class_num = 100
    elif dataset == "Digit-Product":
        dataset = Digit_Product('./data/')
        dims = [1024,1024]
        view = 2
        data_size = 30000
        class_num = 10
    elif dataset == "Cora":
        dataset = Cora('./data/')
        dims = [2708,1433,2708,2708]
        view = 4
        data_size = 2708
        class_num = 7
    elif dataset == "NGs":
        dataset = NGs('./data/')
        dims = [2000, 2000, 2000]
        view = 3
        data_size = 500
        class_num = 5
    elif dataset == "NoisyMNIST":
        dataset = NoisyMNIST('./data/')
        dims = [784, 784]
        view = 2
        data_size = 50000
        class_num = 10
    elif dataset == "3Sources":
        dataset = ThreeSources('./data/')
        dims = [3560, 3631, 3068]
        view = 3
        data_size = 169
        class_num = 6
    elif dataset == "YouTubeVideo":
        dataset = YouTubeVideo('./data/')
        dims = [512, 647, 838]
        view = 3
        data_size = 101499
        class_num = 31

    else:
        raise NotImplementedError
    return dataset, dims, view, data_size, class_num
