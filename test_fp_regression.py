import torch
import numpy as np
import pandas as pd
from coral_pytorch.dataset import corn_label_from_logits
from model_factory import get_vgg11_bn_coral

def test_fp_regression(model, test_dataset, device='cuda', batch_size=1, tqdm_cls=None):
    # Remove blur if blur_amount is 0
    if hasattr(test_dataset, 'blur_amount') and test_dataset.blur_amount == 0:
        from torchvision import transforms
        test_dataset.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()
    fps = []
    preds = []
    iterator = test_loader
    if tqdm_cls is not None:
        iterator = tqdm_cls(test_loader, desc='Testing', unit='batch')
    for i, (img, fp) in enumerate(iterator):
        img = img.to(device)
        fp = fp.to(device)
        with torch.no_grad():
            pred = model(img)
            pred = corn_label_from_logits(pred)
        # Support batch size > 1
        fps.extend(fp.cpu().numpy() + 1)
        preds.extend(pred.cpu().numpy() + 1)
    fps = np.array(fps)
    preds = np.array(preds)
    df = pd.DataFrame({'fp': fps, 'pred': preds, 'file': test_dataset.orig_files})
    df.to_csv('fp_regression_test_results.csv', index=False)
    return df
