import torch
from torch.utils.data import DataLoader
from dataloader import SessioniDataset, val_transform
from model import TwoStreamCNNLSTM
import pandas as pd
import os
from tqdm import tqdm

def generate_submission(model_path='./outputs/models/best_model.pth', split='test'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    data_directory = '../data'
    dataset = SessioniDataset(root_dir=f'{data_directory}/{split}', split=split, num_frames=16, transform=val_transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    # Load model
    model = TwoStreamCNNLSTM(num_classes=126).to(device) ## 0.96575
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    assert os.path.exists(model_path), f"Model checkpoint not found: {model_path}"
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Generate predictions
    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Generating {split} predictions'):
            sample_ids = batch['sample_id']
            rgb = batch['rgb'].to(device)
            rdm = batch['rdm'].to(device)

            outputs = model(rgb, rdm)
            _, preds = outputs.max(1)

            for sample_id, pred in zip(sample_ids, preds):
                predictions.append({'id': sample_id.item(), 'Pred': pred.item()})

    # Create submission CSV
    os.makedirs('./outputs/submissions', exist_ok=True)
    df = pd.DataFrame(predictions)
    output_file = f'./outputs/submissions/submission_{split}.csv'
    df.to_csv(output_file, index=False)
    print(f'[✔] Submission file saved: {output_file}')

if __name__ == '__main__':
    # Use best model and change split to 'test' or 'val' as needed
    # generate_submission(model_path='./outputs/models/best_model.pth', split='test')
    generate_submission(model_path='./outputs/models/best_model.pth', split='val')
