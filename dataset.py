"""
@author: <nktoan163@gmail.com>
"""
from torch.utils.data import Dataset
import os
import cv2
import pandas as pd


class PesplanusDataset(Dataset):
    def __init__(self, root='./Pesplanus.v2i.multiclass', train=True, test=False, transform=None):
        self.classes = [' Pesplanus', ' Notpresplanus']
        if train:
            data_paths = os.path.join(root, 'train')
        else:
            if test:
                data_paths = os.path.join(root, 'test')
            else:
                data_paths = os.path.join(root, 'valid')

        self.images_paths = []
        self.labels = []

        # Load _classes.csv
        csv_path = os.path.join(data_paths, '_classes.csv')
        classes_df = pd.read_csv(csv_path)

        for index, row in classes_df.iterrows():
            image_name = row['filename']
            image_paths = os.path.join(data_paths, image_name)
            self.images_paths.append(image_paths)

            pesplanus_label = row[self.classes[0]]
            if pesplanus_label == 1:
                self.labels.append(pesplanus_label)
            else:
                self.labels.append(0)

        self.transform = transform

    def __len__(self):
        return len(self.images_paths)

    def __getitem__(self, item):
        image = cv2.imread(self.images_paths[item])
        if self.transform:
            image = self.transform(image)
        label = self.labels[item]
        return image, label


