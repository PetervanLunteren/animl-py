'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    Original script from
    2022 Benjamin Kellenberger
'''
import argparse
import yaml
from tqdm import trange
import pandas as pd
import torch
import numpy as np
import os
import csv
import random
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report

from .generator import train_dataloader
from .classifiers import load_model


def test(data_loader, model, device='cpu'):
    '''
        Run trained model on test split

        Args:
            data_loader: test set dataloader
            model: trained model object
            device: run model on gpu or cpu, defaults to cpu
    '''
    model.to(device)
    model.eval()  # put the model into evaluation mode

    pred_labels = []
    true_labels = []
    confidences = []
    filepaths = []

    progressBar = trange(len(data_loader))
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            # forward pass
            data = batch[0]
            data = data.to(device)
            prediction = model(data)
            
            # apply softmax to get the probability distribution
            probabilities = F.softmax(prediction, dim=1)
            
            # get predicted labels
            pred_label = torch.argmax(probabilities, dim=1)
            pred_label_np = pred_label.cpu().detach().numpy()
            pred_labels.extend(pred_label_np)
            
            # get confidence scores (probabilities for the predicted class)
            conf_score = probabilities.max(dim=1).values.cpu().detach().numpy()  # maximum probability
            confidences.extend(conf_score)

            # get ground truth labels
            labels = batch[1]
            labels_np = labels.numpy()
            true_labels.extend(labels_np)
            
            # get file paths
            paths = batch[2]
            filepaths.extend(paths)

            progressBar.update(1)

    return pred_labels, true_labels, confidences, filepaths
    
# # finds the last iteration of a subfolder in experiment_folder and returns its path.
# def find_last_run_dir(experiment_folder: str) -> str:
#     if not os.path.exists(experiment_folder):
#         return None  # folder doesn't exist, no runs yet
    
#     # find existing run_X folders
#     existing_runs = []
#     for folder in os.listdir(experiment_folder):
#         folder_path = os.path.join(experiment_folder, folder)
#         if folder.startswith("run_") and os.path.isdir(folder_path):
#             try:
#                 run_num = int(folder.split("_")[1])
#                 existing_runs.append((run_num, folder_path))
#             except ValueError:
#                 pass  # ignore non-numeric run directories

#     if not existing_runs:
#         return None  # No runs found
    
#     # get the folder with the highest run number
#     last_run_path = max(existing_runs, key=lambda x: x[0])[1]
    
#     return last_run_path

def main():
    '''
    Command line function

    Example usage:
    > python train.py --config configs/exp_resnet18.yaml
    '''
    parser = argparse.ArgumentParser(description='Test species classifier model.')
    parser.add_argument('--config', help='Path to config file')
    args = parser.parse_args()

    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    crop = cfg.get('crop', False)

    # check if GPU is available
    device = cfg.get('device', 'cpu')
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        device = 'cpu'

    # read run dir from config location
    run_dir = os.path.dirname(args.config)

    # initialize model and get class list
    active_model = os.path.join(run_dir, 'best.pt')
    model, classes = load_model(active_model, cfg['class_file'], device=device, architecture=cfg['architecture'])
    categories = dict([[x["class"], x["id"]] for _, x in classes.iterrows()])

    # initialize data loaders for training and validation set 
    if 'test_set' not in cfg: print('No test set specified in config file. Will continue to test on validation set.')
    test_set = cfg.get('test_set', cfg['validate_set'])
    test_dataset = pd.read_csv(test_set).reset_index(drop=True)
    dl_test = train_dataloader(test_dataset, categories, batch_size=cfg['batch_size'], workers=cfg['num_workers'], crop=crop)

    # get predictions
    pred, true, conf, paths = test(dl_test, model, device)
    pred = np.asarray(pred)
    true = np.asarray(true)
    conf = np.asarray(conf)
    corr = (pred == true).tolist()

    # print accuracy
    oa = np.mean((pred == true))
    print(f"Test accuracy: {oa}")

    # save test results
    results = pd.DataFrame({'FilePath': paths,
                            'Ground Truth': true,
                            'Predicted': pred,
                            'Confidence': conf,
                            'Correct': corr})
    results.to_csv(run_dir + "/test_results.csv")
    print(f"Creating: .\\test_results.csv")

if __name__ == '__main__':
    main()
