import glob
import torch
import numpy as np


class EmbeddingsDataloader(torch.utils.data.Dataset):

    def __init__(self, mode='train', overlap=False, width=16):
        self.mode = mode
        self.n = width # num of frames in one datapoint
        self.overlap = overlap

        self.prefix = "data/frame_embeddings/"
        self.prefix_labels = "data/orig_data/"
        self.mode_prefix = mode + '/'
        self.embedings_fileregex = '*_embeddings.pt'
        self.labels_fileregex = '*.csv'

        self.labels = self._load_labels()
        self.embeddings = self._load_embeddings()

        # print("UNIQUE", np.unique(self.labels))
        # # UNIQUE [0. 1. 2. 3. 4. 5. 6. 7. 8.]

        # print("datas shape", self.labels.shape, self.embeddings.shape)
        ## torch.Size([72843]) torch.Size([72843, 384]) on train
    
    def _csv_load(self, path):
        data = []
        with open(path, 'r') as f:
            ll = f.readlines()
            [data.append(int(l)) for l in ll[0].split(',')]
        return np.array(data)

    def _load_labels(self):
        files = glob.glob(self.prefix_labels + self.mode_prefix + self.labels_fileregex)
        files.sort()
        labels = np.array([])
        for file in files:
            labels = np.concatenate([labels, self._csv_load(file)])
        return torch.from_numpy(labels).float()

    def _load_embeddings(self):
        files = glob.glob(self.prefix + self.mode_prefix + self.embedings_fileregex)
        files.sort()
        embeddings = torch.tensor([])
        for file in files:
            embed_tensor = torch.load(file)
            embeddings = torch.cat([embeddings, embed_tensor])
        return embeddings.float()

    def __len__(self):
        if self.overlap:
            return len(self.labels) - self.n
        return len(self.labels) // self.n

    def __getitem__(self, idx):
        n = self.n
        if self.overlap:
            return self.embeddings[idx:idx+n], self.labels[idx:idx+n]
        return self.embeddings[idx*n:idx*n+n], self.labels[idx*n:idx*n+n]

if __name__ == '__main__':
    dd = EmbeddingsDataloader()