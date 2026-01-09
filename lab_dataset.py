import numpy as np
import pandas as pd
import os
import cv2
import torch
from torch.utils.data import Dataset
from fp_dataset import FPDataset
from white_balance import shades_of_grey_wb

class LabDataset(FPDataset):    
    def __getitem__(self, idx):
        orig_path = self.orig_files[idx]
        img_path = self.get_processed_path(orig_path)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.white_balance:
            img = shades_of_grey_wb(img, p=6)
            img = (img * 255).astype(np.uint8)
        if self.transform:
            img = self.transform(img)
        lab = self.lab_values[idx]
        lab = torch.tensor(lab, dtype=torch.float)
        return img, lab

    def mskcc_init(self, files):
        metadata_path = 'data/mskcc/s7.csv' # use s7 instead of metadata.csv to get the L* a* b* values
        images_dir = 'data/mskcc/images'
        df = pd.read_csv(metadata_path)
        df['image_path'] = df['isic_id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))
        df = df[df['image_path'].apply(os.path.exists)]
        if files is not None:
            df = df[df['image_path'].isin(files)]
        self.orig_files = df['image_path'].tolist()

        ls = df['average_l'].tolist()
        as_ = df['average_a'].tolist()
        bs = df['average_b'].tolist()

        self.lab_values = list(zip(ls, as_, bs))

        self.ensure_processed_images(self.orig_files, self.processed_dir)
        print(f"[LabDataset] {len(self.orig_files)} processed files found for mskcc out of {len(df)} total entries.")
