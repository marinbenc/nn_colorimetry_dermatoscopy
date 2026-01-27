import numpy as np
import pandas as pd
import os
import glob
import cv2
import torch
from torch.utils.data import Dataset
from fp_dataset import FPDataset
from lab_dataset import LabDataset
from white_balance import shades_of_grey_wb


class InferenceDatasetMixin:    
    def isic2020_init(self, files):
        metadata_path = 'inference_data/isic2020/train-metadata.csv'
        images_dir = 'inference_data/isic2020/train-image/image'
        df = pd.read_csv(metadata_path)
        df['image_path'] = df['isic_id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))
        df = df[df['image_path'].apply(os.path.exists)]
        if files is not None:
            df = df[df['image_path'].isin(files)]
        self.orig_files = df['image_path'].tolist()
        self.patient_ids = df['patient_id'].tolist()
        self.ensure_processed_images(self.orig_files, self.processed_dir)
        print(f"[InferenceDataset] {len(self.orig_files)} processed files found for isic2020 out of {len(df)} total entries.")
    
    def milk10k_init(self, files):
        metadata_path = 'inference_data/milk10k/MILK10k_Training_Metadata.csv'
        images_dir = 'inference_data/milk10k/MILK10k_Training_Input'
        df = pd.read_csv(metadata_path)
        # Extract all image files recursively using glob
        image_files = glob.glob(os.path.join(images_dir, '**', '*.jpg'), recursive=True)
        image_files_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in image_files}
        # Map ISIC IDs to image paths
        df['image_path'] = df['isic_id'].map(image_files_dict)
        df = df.dropna(subset=['image_path'])
        if files is not None:
            df = df[df['image_path'].isin(files)]
        self.orig_files = df['image_path'].tolist()
        self.patient_ids = df['lesion_id'].tolist()  # Use lesion_id as pseudo patient_id
        self.ensure_processed_images(self.orig_files, self.processed_dir)
        print(f"[InferenceDataset] {len(self.orig_files)} processed files found for milk10k out of {len(df)} total entries.")

    def isic2020_test_init(self, files):
        metadata_path = 'inference_data/isic2020_test/challenge-2020-test_metadata_2026-01-14.csv'
        images_dir = 'inference_data/isic2020_test/test-resized'
        df = pd.read_csv(metadata_path)
        # Build image paths from isic_id column
        df['image_path'] = df['isic_id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))
        df = df[df['image_path'].apply(os.path.exists)]
        if files is not None:
            df = df[df['image_path'].isin(files)]
        self.orig_files = df['image_path'].tolist()
        # Test metadata may not have patient_id; fall back to None for each entry
        if 'patient_id' in df.columns:
            self.patient_ids = df['patient_id'].tolist()
        else:
            self.patient_ids = [None] * len(self.orig_files)
        self.ensure_processed_images(self.orig_files, self.processed_dir)
        print(f"[InferenceDataset] {len(self.orig_files)} processed files found for isic2020_test out of {len(df)} total entries.")


class InferenceFPDataset(FPDataset, InferenceDatasetMixin):
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
        return img


class InferenceLabDataset(LabDataset, InferenceDatasetMixin):
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
        return img
