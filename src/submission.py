import torch
from torch.utils.data import DataLoader
from dataloader import SessioniDataset, val_transform
from model import TwoStreamCNNLSTM, SwinTwoStreamModel, AdvancedTwoStreamModel, UltraAdvancedTwoStreamModel
import pandas as pd
import os
from tqdm import tqdm

def generate_submission(model_path='./outputs/models/best_model.pth', split='test'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    data_directory = '../data'
    dataset = SessioniDataset(root_dir=f'{data_directory}/{split}', split=split, num_frames=16, transform=val_transform)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=4)

    ## Load model
    # model = TwoStreamCNNLSTM(num_classes=126).to(device) ## 0.96575 (val) ## 0.96575 (test) 

    ###----------------------------------------------------------------------------------------
    model = AdvancedTwoStreamModel(num_classes=126).to(device) 
    
    ''' 
    ## Best accuracy model
    ```
    |    Val    |   Test    |
    |-----------|-----------|
    | 0.98594   | 0.98594   |
    | 0.98752   | 0.99126   |
    | 0.98662   | 0.98994   |
    | 0.98956   | 0.99060   |
    | 0.94240   | 0.94417   |
    | 0.96938   | 0.97341   |
    | 0.99365   | 0.99444   | Ensemble All Model (TwoStreamCNNLSTM, AdvancedTwoStreamModel, UltraAdvancedTwoStreamModel, SwinTwoStreamModel)
    ```

    ''' 
    ###----------------------------------------------------------------------------------------

    # model = UltraAdvancedTwoStreamModel(num_classes=126).to(device) ## 0.96938  (val) ## 0.97341 (test) 
    # model = SwinTwoStreamModel(num_classes=126).to(device) ## 0.94240 (val) ## 0.94417 (test)

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


def merge_val_test_csv():
    val_csv = './outputs/submissions/submission_val.csv'
    test_csv = './outputs/submissions/submission_test.csv'
    merged_csv = './outputs/submissions/submission_valtest.csv'

    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)
    df_merged = pd.concat([df_val, df_test], ignore_index=True)
    df_merged.to_csv(merged_csv, index=False)
    print(f'[✔] Merged CSV saved: {merged_csv}')
    print(f'[i] Number of rows in merged CSV: {len(df_merged)}')
    if len(df_merged) == 11970:
        print('[✔] Submission has the correct number of rows (11970).')
    else:
        print(f'[!] Warning: Submission has {len(df_merged)} rows, expected 11970.')

if __name__ == '__main__':
    ## Use best model and change split to 'test' or 'val' as needed

    ### Best accuracy model
    generate_submission(model_path='./outputs/models/best_model.pth', split='val')
    generate_submission(model_path='./outputs/models/best_model.pth', split='test')

    ### Best loss model
    # generate_submission(model_path='./outputs/models/best_model_loss.pth', split='val')
    # generate_submission(model_path='./outputs/models/best_model_loss.pth', split='test')


    merge_val_test_csv()
    
