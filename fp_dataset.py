import numpy as np
import pandas as pd
import os
import glob
import re
import cv2
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from torchvision import transforms
from white_balance import shades_of_grey_wb

def bin_FP_by_mel(mel_range, mel):
    # find the category that mel falls into
    # if mel is less than the first mid point, return the first category
    if mel < mel_range['mid'].iloc[0]:
        return 1
    # if mel is greater than the last mid point, return the last category
    if mel > mel_range['mid'].iloc[-1]:
        return mel_range.index[-1]
    # find the category that mel falls into
    for i in range(len(mel_range) - 1):
        if mel_range['mid'].iloc[i] <= mel < mel_range['mid'].iloc[i + 1]:
            return mel_range.index[i]
    return None

class FPDataset(Dataset):
    def __new__(cls, dataset_name, files, blur_amount=21, white_balance=False):
        if '+' in dataset_name:
            dataset_names = dataset_name.split('+')
            return CombinedFPDataset(dataset_names, files, blur_amount, white_balance)
        return super().__new__(cls)

    def __init__(self, dataset_name, files, blur_amount=21, white_balance=False):
        self.dataset_name = dataset_name
        self.blur_amount = blur_amount
        self.white_balance = white_balance
        self.orig_files = []  # Always store original file paths
        self.fps = []
        self.processed_dir = f"{dataset_name}_processed"
        
        # Dynamically call the appropriate init method
        init_method_name = f"{dataset_name}_init"
        if not hasattr(self, init_method_name):
            raise ValueError(f"Unknown dataset_name: {dataset_name}. No method '{init_method_name}' found.")
        
        init_method = getattr(self, init_method_name)
        init_method(files)
        
        # Build transform pipeline, remove blur if blur_amount is 0
        if self.blur_amount == 0:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.GaussianBlur(self.blur_amount, sigma=5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def get_processed_path(self, orig_path):
        # Always keep subdir structure relative to the dataset root
        if self.dataset_name == 'ssynth':
            rel_path = os.path.relpath(orig_path, start='data/output_10k')
        elif self.dataset_name == 'mskcc':
            rel_path = os.path.relpath(orig_path, start='data/mskcc/images')
        elif self.dataset_name == 'fp17k':
            # fp17k images have no extension, add .jpg for processed files
            base = os.path.basename(orig_path)
            if '.' not in base:
                base += '.jpg'
            rel_path = base
        else:
            rel_path = os.path.basename(orig_path)
        return os.path.join(self.processed_dir, rel_path)

    def ensure_processed_images(self, orig_files, processed_dir, resize=(224, 224)):
        processed_files = [self.get_processed_path(f) for f in orig_files]
        for proc_path in processed_files:
            proc_dir = os.path.dirname(proc_path)
            if not os.path.exists(proc_dir):
                os.makedirs(proc_dir, exist_ok=True)
        # --- Skip missing files up front ---
        existing_pairs = [(orig, proc) for orig, proc in zip(orig_files, processed_files) if os.path.exists(orig)]
        skipped_missing = len(orig_files) - len(existing_pairs)
        if skipped_missing > 0:
            print(f"[FPDataset] Skipping {skipped_missing} missing files (not present on disk).")
        # Only check is_readable if processed file does not exist
        def is_readable(path):
            img = cv2.imread(path)
            return img is not None
        pairs_to_process = []
        for orig, proc in existing_pairs:
            if not os.path.exists(proc):
                if not is_readable(orig):
                    print(f"[FPDataset] Warning: Could not read {orig}")
                    continue
                pairs_to_process.append((orig, proc))
        if pairs_to_process:
            print(f"[FPDataset] Resizing images for {self.dataset_name} to {resize} and saving to {processed_dir}...")
            def process_one(args):
                orig, proc = args
                img = cv2.imread(orig)
                if img is None:
                    print(f"[FPDataset] Warning: Could not read {orig}")
                    return
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
                cv2.imwrite(proc, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            with ThreadPoolExecutor() as executor:
                list(tqdm(executor.map(process_one, pairs_to_process), total=len(pairs_to_process), desc=f"Resizing {self.dataset_name}"))
            print(f"[FPDataset] Done resizing images for {self.dataset_name}.")
        else:
            print(f"[FPDataset] Found all resized images for {self.dataset_name} in {processed_dir}.")
        return [self.get_processed_path(f) for f in orig_files]

    def ssynth_init(self, files):
        root_dir = 'data/output_10k'
        all_files = glob.glob(os.path.join(root_dir, '**', 'image.png'), recursive=True)
        if files is not None:
            all_files = [f for f in all_files if f in files]
        self.orig_files = sorted(all_files)
        mel_range = pd.read_csv('mel_range.csv', index_col=0)
        self.mels = []
        for f in self.orig_files:
            m = re.search(r'mel_([0-9]+\.[0-9]+)', f)
            self.mels.append(float(m.group(1)))
        self.mels = np.array(self.mels)
        self.fps = [bin_FP_by_mel(mel_range, mel) for mel in self.mels]
        self.ensure_processed_images(self.orig_files, self.processed_dir)
        print(f"[FPDataset] {len(self.orig_files)} processed files found for ssynth out of {len(all_files)} total entries.")

    def mskcc_init(self, files):
        metadata_path = 'data/mskcc/metadata.csv'
        images_dir = 'data/mskcc/images'
        df = pd.read_csv(metadata_path)
        df['image_path'] = df['isic_id'].apply(lambda x: os.path.join(images_dir, f"{x}.jpg"))
        df = df[df['image_path'].apply(os.path.exists)]
        if files is not None:
            df = df[df['image_path'].isin(files)]
        self.orig_files = df['image_path'].tolist()
        fp_map = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5, 'VI': 6}
        self.fps = [fp_map.get(str(fp).strip().upper(), 0) for fp in df['fitzpatrick_skin_type']]
        self.patient_ids = df['patient_id'].tolist()
        self.ensure_processed_images(self.orig_files, self.processed_dir)
        print(f"[FPDataset] {len(self.orig_files)} processed files found for mskcc out of {len(df)} total entries.")

    def diverse_init(self, files):
        import random
        metadata_path = 'data/diverse/ddi_metadata.csv'
        images_dir = 'data/diverse'
        df = pd.read_csv(metadata_path)
        df['image_path'] = df['DDI_file'].apply(lambda x: os.path.join(images_dir, x))
        df = df[df['image_path'].apply(os.path.exists)]
        if files is not None:
            df = df[df['image_path'].isin(files)]
        self.orig_files = df['image_path'].tolist()
        # Map skin_tone to 1/2, 3/4, 5/6 randomly
        def map_skin_tone(st):
            st = str(st)
            if st == '12':
                return random.choice([1, 2])
            elif st == '34':
                return random.choice([3, 4])
            elif st == '56':
                return random.choice([5, 6])
            else:
                raise ValueError(f"Unknown skin tone: {st}")
        self.fps = [map_skin_tone(st) for st in df['skin_tone']]
        # Filter out any samples with label 0 (invalid label)
        filtered = [(f, fp) for f, fp in zip(self.orig_files, self.fps) if fp in [1,2,3,4,5,6]]
        if len(filtered) < len(self.orig_files):
            print(f"[FPDataset] Filtered out {len(self.orig_files) - len(filtered)} diverse samples with invalid FP labels (not 1-6).")
        self.orig_files, self.fps = zip(*filtered) if filtered else ([],[])
        self.orig_files = list(self.orig_files)
        self.fps = list(self.fps)
        self.ensure_processed_images(self.orig_files, self.processed_dir)
        print(f"[FPDataset] {len(self.orig_files)} processed files found for diverse out of {len(df)} total entries.")

    def fp17k_init(self, files):
        metadata_path = 'data/fp17k/fitzpatrick17k.csv'
        images_dir = 'data/fp17k/images'
        df = pd.read_csv(metadata_path)
        df['image_path'] = df['md5hash'].apply(lambda x: os.path.join(images_dir, x))
        # Check for missing files
        missing_files = df[~df['image_path'].apply(os.path.exists)]['image_path'].tolist()
        if missing_files:
            print(f"Missing image files in fp17k: {missing_files[:5]}... (total missing: {len(missing_files)})")
        if files is not None:
            df = df[df['image_path'].isin(files)]
        self.orig_files = df['image_path'].tolist()
        # Use fitzpatrick_scale as label, ensure int and valid
        invalid_count = 0
        def check_fp(fp):
            nonlocal invalid_count
            if not str(fp).isdigit() or not (1 <= int(fp) <= 6):
                invalid_count += 1
                return 0
            return int(fp)
        self.fps = [check_fp(fp) for fp in df['fitzpatrick_scale']]
        if invalid_count > 0:
            print(f"[FPDataset] {invalid_count} invalid fitzpatrick_scale labels found in fp17k.")
        # Filter out any samples with label 0 (invalid label) or missing processed image
        filtered = [(f, fp) for f, fp in zip(self.orig_files, self.fps)
                    if fp in [1,2,3,4,5,6] and os.path.exists(self.get_processed_path(f))]
        if len(filtered) < len(self.orig_files):
            print(f"[FPDataset] Filtered out {len(self.orig_files) - len(filtered)} fp17k samples with invalid FP labels or missing processed images.")
        self.orig_files, self.fps = zip(*filtered) if filtered else ([],[])
        self.orig_files = list(self.orig_files)
        self.fps = list(self.fps)
        print(f"[FPDataset] {len(self.orig_files)} processed files found for fp17k out of {len(df)} total entries.")

    def scin_init(self, files):
        # Load both metadata files
        labels_path = 'data/scin/dataset/scin_labels.csv'
        cases_path = 'data/scin/dataset/scin_cases.csv'
        images_dir = 'data/scin/dataset/images'
        df_labels = pd.read_csv(labels_path)
        df_cases = pd.read_csv(cases_path)
        # Only keep images with sufficient quality
        df_labels = df_labels[df_labels['dermatologist_gradable_for_skin_condition_1'] == 'DEFAULT_YES_IMAGE_QUALITY_SUFFICIENT']
        # Merge on case_id
        df = pd.merge(df_labels, df_cases, on='case_id', how='inner', suffixes=('', '_case'))
        # --- Vectorized filtering for valid FP labels ---
        fp_cols = [f'dermatologist_fitzpatrick_skin_type_label_{i}' for i in range(1, 4)]
        def valid_fp_label(x):
            if pd.isnull(x) or str(x).strip() == '':
                return False
            s = str(x).strip().upper()
            return s.startswith('FST') and s[3:].isdigit() and 1 <= int(s[3:]) <= 6
        valid_mask = df[fp_cols].apply(lambda row: any(valid_fp_label(x) for x in row), axis=1)
        df = df[valid_mask]
        if df.empty:
            print("[FPDataset] No valid images found in scin after merging metadata.")
            self.orig_files = []
            self.fps = []
            return
        # Collect all image paths for each case (image_1_path, image_2_path, image_3_path)
        image_cols = ['image_1_path', 'image_2_path', 'image_3_path']
        # Stack all image paths into a single Series, keeping index for row reference
        img_paths = df[image_cols].stack().reset_index(level=1, drop=True)
        img_paths = img_paths[img_paths.notnull() & (img_paths.astype(str).str.strip() != '')]
        # Build full paths and keep only those that exist
        full_paths = img_paths.apply(lambda p: os.path.join('data/scin', p) if not str(p).startswith('data/scin') else p)
        exists_mask = full_paths.apply(os.path.exists)
        valid_img_paths = full_paths[exists_mask]
        valid_indices = valid_img_paths.index
        self.orig_files = valid_img_paths.tolist()
        # For each valid image, get the corresponding row for FP label voting
        df_valid = df.loc[valid_indices]
        def parse_fp_row(row):
            labels = [row[c] for c in fp_cols]
            labels = [str(l).strip().upper() for l in labels if pd.notnull(l) and str(l).strip() != '']
            valid = [l for l in labels if l.startswith('FST') and l[3:].isdigit() and 1 <= int(l[3:]) <= 6]
            if not valid:
                print(f"[FPDataset] Warning: No valid fitzpatrick label for case_id {row['case_id']}")
                return 0
            counts = {k: valid.count(k) for k in set(valid)}
            majority = max(counts, key=counts.get)
            return int(majority[3:])
        self.fps = df_valid.apply(parse_fp_row, axis=1).tolist()
        # Filter out any samples with label 0 (invalid label)
        filtered = [(f, fp) for f, fp in zip(self.orig_files, self.fps) if fp in [1,2,3,4,5,6]]
        if len(filtered) < len(self.orig_files):
            print(f"[FPDataset] Filtered out {len(self.orig_files) - len(filtered)} scin samples with invalid FP labels (not 1-6).")
        self.orig_files, self.fps = zip(*filtered) if filtered else ([],[])
        self.orig_files = list(self.orig_files)
        self.fps = list(self.fps)
        self.ensure_processed_images(self.orig_files, self.processed_dir)
        print(f"[FPDataset] {len(self.orig_files)} processed files found for scin after merging metadata.")

    def pad_ufes_20_init(self, files):
        # Metadata and image locations
        metadata_path = 'data/pad_ufes_20/zr7vgbcyr2-1/metadata.csv'
        image_dirs = [
            'data/pad_ufes_20/zr7vgbcyr2-1/images/imgs_part_1',
            'data/pad_ufes_20/zr7vgbcyr2-1/images/imgs_part_2',
            'data/pad_ufes_20/zr7vgbcyr2-1/images/imgs_part_3',
        ]
        df = pd.read_csv(metadata_path)
        # Only keep rows with valid Fitzpatrick (fitspatrick) label 1-6
        def parse_fp(fp):
            try:
                val = int(float(fp))
                if 1 <= val <= 6:
                    return val
            except Exception:
                pass
            return 0
        df['fp'] = df['fitspatrick'].apply(parse_fp)
        df = df[df['fp'].isin([1,2,3,4,5,6])]
        # Build image paths
        def find_img_path(img_id):
            for d in image_dirs:
                p = os.path.join(d, img_id)
                if os.path.exists(p):
                    return p
            return None
        df['img_path'] = df['img_id'].apply(find_img_path)
        df = df[df['img_path'].notnull()]
        if files is not None:
            df = df[df['img_path'].isin(files)]
        self.orig_files = df['img_path'].tolist()
        self.fps = df['fp'].tolist()
        self.ensure_processed_images(self.orig_files, self.processed_dir)
        print(f"[FPDataset] {len(self.orig_files)} processed files found for pad_ufes_20 out of {len(df)} total entries.")

    def mra_midas_init(self, files):
        # Metadata and image locations
        metadata_path = 'data/mra-midas/release_midas.xlsx'
        images_dir = 'data/mra-midas'
        df = pd.read_excel(metadata_path)
        # Only use rows where midas_distance == 'dscope'
        df = df[df['midas_distance'].astype(str).str.lower() == 'dscope']
        # Parse Fitzpatrick from midas_fitzpatrick (should contain I-VI or 1-6)
        def parse_fp(fp):
            if pd.isnull(fp):
                return 0
            s = str(fp).strip().lower()
            # Use regex to match the roman numeral at the start
            match = re.match(r'^(vi|iv|iii|ii|v|i)\b', s)
            roman_map = {'i': 1, 'ii': 2, 'iii': 3, 'iv': 4, 'v': 5, 'vi': 6}
            if match:
                roman = match.group(1)
                return roman_map.get(roman, 0)
            return 0
        df['fp'] = df['midas_fitzpatrick'].apply(parse_fp)
        df = df[df['fp'].isin([1,2,3,4,5,6])]
        # Build image paths
        df['img_path'] = df['midas_file_name'].apply(lambda x: os.path.join(images_dir, x) if pd.notnull(x) and os.path.exists(os.path.join(images_dir, x)) else None)
        df = df[df['img_path'].notnull()]
        if files is not None:
            df = df[df['img_path'].isin(files)]
        self.orig_files = df['img_path'].tolist()
        self.fps = df['fp'].tolist()
        self.ensure_processed_images(self.orig_files, self.processed_dir)
        print(f"[FPDataset] {len(self.orig_files)} processed files found for mra-midas out of {len(df)} total entries.")

    def __len__(self):
        return len(self.orig_files)

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
        fp = self.fps[idx] - 1
        fp = torch.tensor(fp, dtype=torch.long)
        return img, fp

class CombinedFPDataset(Dataset):
    def __init__(self, dataset_names, files, blur_amount=21, white_balance=False):
        self.datasets = []
        self.orig_files = []
        self.fps = []
        self.file_dataset = []
        self.blur_amount = blur_amount
        self.white_balance = white_balance
        
        for name in dataset_names:
            ds = FPDataset(name, files, blur_amount=blur_amount, white_balance=white_balance)
            self.datasets.append(ds)
            self.orig_files.extend(ds.orig_files)
            self.fps.extend(ds.fps)
            self.file_dataset.extend([name] * len(ds.orig_files))
            
        self.transform = self.datasets[0].transform if self.datasets else None
        self.processed_dir = '+'.join(dataset_names) + '_processed'
        
    def __len__(self):
        return len(self.orig_files)
    
    def __getitem__(self, idx):
        # Find which dataset this idx belongs to
        running = 0
        for ds in self.datasets:
            if idx < running + len(ds):
                return ds[idx - running]
            running += len(ds)
        raise IndexError('Index out of range in CombinedFPDataset')
