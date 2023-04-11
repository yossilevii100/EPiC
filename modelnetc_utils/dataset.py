import os
import h5py
from torch.utils.data import Dataset
#from augmentation.PointWOLF.PointWOLF import PointWOLF

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'modelnet_c')


def load_h5(h5_name):
    #print(h5_name)
    f = h5py.File(h5_name, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    return data, label


class ModelNetC(Dataset):
    def __init__(self, args, split):
        h5_path = os.path.join(DATA_DIR, split + '.h5')
        self.data, self.label = load_h5(h5_path)
        #if args.use_wolfmix:
        #    self.PointWOLF = PointWOLF(args)

    def __getitem__(self, item):
        pointcloud = self.data[item]
        #if args.use_wolfmix and not args.eval:
        #    _, pointcloud = self.PointWOLF(pointcloud)
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
