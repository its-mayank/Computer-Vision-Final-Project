import numpy as np
import os
from PIL import Image
import pandas as pd 
from nltk import word_tokenize
import imageio
from torch.utils.data import Dataset, DataLoader

def get_image(arr, size):
    arr = word_tokenize(arr)
    arr = [float(t) for t in arr]
    return np.asarray(arr).reshape((size, size))

class ImageFolder(Dataset):
    def __init__(self, root, subset = 'train'):
        df = pd.read_csv(root + subset + '.csv')
        self.labels = df.emotion.tolist()
        self.pixels = df.pixels.tolist()
        assert len(self.labels) == len(self.pixels), 'Length of labels and pixels lists is not same!'
        self.length = len(self.labels)
        self.size = 48
        self.nb_classes = 7
        self.matrix = np.eye(self.nb_classes)

    def __getitem__(self, index):
        image_pixel = self.pixels[index]
        label = self.labels[index]
        # print(label)
        img = get_image(image_pixel, self.size)
        _min, _max = np.amin(img), np.amax(img)
        # print(_min, _max)
        #img = (img - _min)/ (_max - _min)
        img = np.stack((img, img, img), axis=2)
        img = np.reshape(img, (3, self.size, self.size))
        
        return img, label

    def __len__(self):
        return self.length

def get_loader(data_root, subset, batch_size, shuffle = True):
    dataset = ImageFolder(data_root, subset = subset)
    loader = DataLoader(dataset, num_workers = 12, batch_size = batch_size, shuffle = shuffle)
    return loader