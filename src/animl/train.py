'''
    Training script. Here, we load the training and validation datasets (and
    data loaders) and the model and train and validate the model accordingly.

    Original script from
    2022 Benjamin Kellenberger
'''

# TODO: get scheduler bool from config file... now it is hard coded.

import argparse
import yaml
from tqdm import trange
import pandas as pd
import random
import torch.nn as nn
import os
import csv
import torch
from torch.backends import cudnn
from torch.optim import SGD
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score
from torch.optim.lr_scheduler import ReduceLROnPlateau, ExponentialLR
from sklearn.utils.class_weight import compute_class_weight


from .generator import train_dataloader, find_optimal_batch_size
from .classifiers import save_model, load_model

# # log values using comet ml (comet.com)
# from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model
# experiment = Experiment()


def init_seed(seed):
    '''
        Initalizes the seed for all random number generators used. This is
        important to be able to reproduce results and experiment with different
        random setups of the same code and experiments.
    '''
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True
        cudnn.deterministic = True


def train(data_loader, model, optimizer, device='cpu', class_weights_tensor=None):
    '''
        Our actual training function.
    '''
    model.to(device)
    model.train()  # put the model into training mode

    # loss function
    if class_weights_tensor is not None and class_weights_tensor.numel() > 0:
        criterion = nn.CrossEntropyLoss(weight = class_weights_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    # log the loss and overall accuracy (OA)
    loss_total, oa_total = 0.0, 0.0

    progressBar = trange(len(data_loader))
    for idx, batch in enumerate(data_loader):
        # put data and labels on device
        data = batch[0]
        labels = batch[1]
        data, labels = data.to(device), labels.to(device)
        # forward pass
        prediction = model(data)
        # reset gradients to zero
        optimizer.zero_grad()

        loss = criterion(prediction, labels)
        # calculate gradients of current batch
        loss.backward()
        # apply gradients to model parameters
        optimizer.step()

        loss_total += loss.item()

        pred_label = torch.argmax(prediction, dim=1)

        oa = torch.mean((pred_label == labels).float())
        oa_total += oa.item()

        progressBar.set_description(
            '[Train] Loss: {:.2f}; OA: {:.2f}%'.format(
                loss_total/(idx+1),
                100*oa_total/(idx+1)
            )
        )
        progressBar.update(1)

    # end of epoch
    progressBar.close()
    loss_total /= len(data_loader)
    oa_total /= len(data_loader)

    return loss_total, oa_total


def validate(data_loader, model, device="cpu"):
    '''
        Validation function. Note that this looks almost the same as the training
        function, except that we don't use any optimizer or gradient steps.
    '''
    model.to(device)
    model.eval()  # put the model into evaluation mode

    criterion = nn.CrossEntropyLoss()

    # log the loss and overall accuracy (OA)
    loss_total, oa_total = 0.0, 0.0

    # create empty lists for true and predicted labels
    true_labels = []
    pred_labels = []

    progressBar = trange(len(data_loader))
    with torch.no_grad():  # gradients not necessary for validation
        for idx, batch in enumerate(data_loader):
            data = batch[0]
            labels = batch[1]
            data, labels = data.to(device), labels.to(device)

            # add true labels to the true labels list
            labels_np = labels.cpu().detach().numpy()
            true_labels.extend(labels_np)

            # forward pass
            prediction = model(data)

            # loss
            loss = criterion(prediction, labels)

            # log statistics
            loss_total += loss.item()

            pred_label = torch.argmax(prediction, dim=1)
            oa = torch.mean((pred_label == labels).float())
            oa_total += oa.item()

            # add predicted labels to the predicted labels list
            pred_label_np = pred_label.cpu().detach().numpy()
            pred_labels.extend(pred_label_np)

            progressBar.set_description(
                '[Val  ] Loss: {:.2f}; OA: {:.2f}%'.format(
                    loss_total/(idx+1),
                    100*oa_total/(idx+1)
                )
            )
            progressBar.update(1)

    # end of epoch; finalize
    progressBar.close()
    loss_total /= len(data_loader)
    oa_total /= len(data_loader)

    # calculate precision and recall
    precision = precision_score(true_labels, pred_labels, average="weighted")
    recall = recall_score(true_labels, pred_labels, average="weighted")

    return loss_total, oa_total, precision, recall

def append_to_history_csv(history_csv, epoch, train_loss, train_acc, val_loss, val_acc, val_precision, val_recall, learning_rate):
    # Write or append history to CSV file
    file_exists = os.path.isfile(history_csv)
    with open(history_csv, 'a', newline='') as csvfile:
        fieldnames = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'val_precision', 'val_recall', 'learning_rate']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header only if the file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'learning_rate': learning_rate
        })

# plot training metrics
def plot_training_metrics(csv_file, session_dir):
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)
    
    # Extract data from the DataFrame
    epochs_range = df['epoch']
    acc = df['train_acc']
    val_acc = df['val_acc']
    loss = df['train_loss']
    val_loss = df['val_loss']
    
    plt.figure(figsize=(8, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig(os.path.join(session_dir, 'training_metrics.png'))
    plt.close()

# finds the last iteration of a subfolder in experiment_folder and creates the next one
def create_next_run_dir(experiment_folder: str) -> str:
    
    # find existing run_{d} folders
    os.makedirs(experiment_folder, exist_ok=True)
    existing_runs = []
    for folder in os.listdir(experiment_folder):
        folder_path = os.path.join(experiment_folder, folder)
        if folder.startswith("run_") and os.path.isdir(folder_path):
            try:
                run_num = int(folder.split("_")[1])
                existing_runs.append(run_num)
            except ValueError:
                pass  # ignore non-numeric run directories
    
    # determine next run number
    next_run_num = max(existing_runs, default=0) + 1
    next_run_folder = os.path.join(experiment_folder, f"run_{next_run_num}")
    os.makedirs(next_run_folder, exist_ok=True)
    
    return next_run_folder

def main():
    '''
    Command line function

    Example usage :
    > python train.py --config configs/exp_resnet18.yaml
    '''
    parser = argparse.ArgumentParser(description='Train deep learning model.')
    parser.add_argument('--config', help='Path to config file', default='exp_resnet18.yaml')
    args = parser.parse_args()

    # load config
    print(f'\n\nUsing config "{args.config}"')
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    # ensure 'experiment_folder' key exists
    if 'experiment_folder' not in cfg:
        raise KeyError("Config file must contain 'experiment_folder' key.")

    # create next run dir
    run_dir = create_next_run_dir(cfg['experiment_folder'])
    print(f'Created dir "{run_dir}" to store files for this run')

    # save config to store for later use
    used_config_fpath = os.path.join(run_dir, 'used-config.yml')
    with open(used_config_fpath, 'w') as f:
        yaml.safe_dump(cfg, f)

    # init random number generator seed (set at the start)
    init_seed(cfg.get('seed', None))
    crop = cfg.get('crop', False)

    # code to save training history
    history_csv = os.path.join(run_dir, 'training_history.csv')

    # check if GPU is available
    device = cfg.get('device', 'cpu')
    if device != 'cpu' and not torch.cuda.is_available():
        print(f'WARNING: device set to "{device}" but CUDA not available; falling back to CPU...')
        device = 'cpu'

    # initialize model and get class list
    model, classes, current_epoch = load_model(run_dir, cfg['class_file'], device=device, architecture=cfg['architecture'])
    categories = dict([[x["class"], x["id"]] for _, x in classes.iterrows()])

    # load datasets
    train_dataset = pd.read_csv(cfg['training_set']).reset_index(drop=True)
    validate_dataset = pd.read_csv(cfg['validate_set']).reset_index(drop=True)
    
    # compute optimal batch size
    if 'batch_size' in cfg:
        batch_size = cfg['batch_size']
    else:
        batch_size = find_optimal_batch_size(dataset = train_dataset,
                                            categories = categories,
                                            num_workers = cfg['num_workers'],
                                            crop = crop,
                                            resize_height = cfg['image_size'][0],
                                            resize_width = cfg['image_size'][1],
                                            augment = cfg.get('augment', False),
                                            device = device,
                                            model = model)
    
    # initialize data loaders for training and validation set
    dl_train = train_dataloader(train_dataset,
                                categories,
                                batch_size=batch_size,
                                workers=cfg['num_workers'],
                                crop=crop,
                                resize_height=cfg['image_size'][0],
                                resize_width=cfg['image_size'][1],
                                augment=cfg.get('augment', False))
    dl_val = train_dataloader(validate_dataset,
                              categories,
                              batch_size=batch_size,
                              workers=cfg['num_workers'],
                              crop=crop,
                              resize_height=cfg['image_size'][0],
                              resize_width=cfg['image_size'][1],
                              augment=False)

    # calculate class weights
    use_class_weights = cfg.get('use_class_weights', False)
    if use_class_weights:
        print("Using class weights")
        y_train = train_dataset['species'].map(dl_train.dataset.categories).values
        class_indices = list(dl_train.dataset.categories.values())
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=class_indices,
            y=y_train)
        scaling_factor = len(train_dataset) / sum(class_weights)
        class_weights_tensor = torch.tensor(class_weights * scaling_factor, dtype=torch.float32).to(device)
    else:
        class_weights_tensor = None

    # set up model optimizer
    optim = SGD(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    
    # initialize scheduler
    use_scheduler = cfg.get('use_scheduler', False)
    if use_scheduler:
        scheduler_patience = int(cfg.get('patience', 20) / 2) - 1 # give the scheduler the chance to lower the learning rate twice before early stopping
        print(f"Using learning rate scheduler ReduceLROnPlateau (factor=0.5, patience={scheduler_patience})")
        scheduler = ReduceLROnPlateau(optim, mode='min', factor=0.5, patience=scheduler_patience)
        # print(f"Using learning rate scheduler ExponentialLR (gamma=0.5)") # DEBUG
        # gamma = 0.95
        # scheduler = ExponentialLR(optim, gamma=gamma)


    # initialize training arguments
    numEpochs = cfg['num_epochs']
    if 'patience' in cfg:
        patience = cfg['patience']
        early_stopping = True
        best_val_loss = float('inf')
        epochs_no_improve = 0
        print(f"Early stopping enabled with a patience of {patience} epochs")
    else:
        early_stopping = False

    # training loop
    while current_epoch < numEpochs:
        current_epoch += 1
        print(f'Epoch {current_epoch}/{numEpochs}')
        print(f"Using learning rate : {scheduler.get_last_lr()[0]}")

        loss_train, oa_train = train(dl_train, model, optim, device, class_weights_tensor)
        loss_val, oa_val, precision, recall = validate(dl_val, model, device)

        # combine stats and save
        stats = {
            'num_classes': len(classes),
            'loss_train': loss_train,
            'loss_val': loss_val,
            'oa_train': oa_train,
            'oa_val': oa_val,
            'precision': precision,
            'recall': recall,
            'image_size': cfg['image_size'],
            'architecture': cfg['architecture'],
            'categories': categories,
            'epoch': current_epoch   
        }
        
        # print stats
        print(f"       train loss : {loss_train:.5f}")
        print(f"         train OA : {oa_train:.5f}")
        print(f"         val loss : {loss_val:.5f}")
        print(f"           val OA : {oa_val:.5f}")
        print(f"    val precision : {precision:.5f}")
        print(f"       val recall : {recall:.5f}")
        
        # Write history and update plots
        append_to_history_csv(history_csv,
                              current_epoch,
                              loss_train,
                              oa_train,
                              loss_val,
                              oa_val,
                              precision,
                              recall,
                              scheduler.get_last_lr()[0] if use_scheduler else cfg['learning_rate'])
        plot_training_metrics(history_csv, run_dir)

        # <current_epoch>.pt checkpoint saving every *checkpoint_frequency* epochs
        checkpoint = cfg.get('checkpoint_frequency', 10)
        # experiment.log_metrics(stats, step=current_epoch)
        if current_epoch % checkpoint == 0:
            save_model(run_dir, current_epoch, model, stats)
        
        # if user specified early stopping
        if early_stopping:
            
            # best.pt saving
            if loss_val < best_val_loss:
                best_val_loss = loss_val
                epochs_no_improve = 0
                save_model(run_dir, 'best', model, stats)
                print(f"\nCurrent best model saved at epoch {current_epoch} with ...")
                print(f"         val loss : {best_val_loss:.5f}")
                print(f"           val OA : {oa_val:.5f}")
                print(f"    val precision : {precision:.5f}")
                print(f"       val recall : {recall:.5f}\n")

            else:
                epochs_no_improve += 1

            # last.pt saving
            save_model(run_dir, 'last', model, stats)
        
            # check patience
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement.")
                break
        
        # step scheduler
        if use_scheduler:
            scheduler.step(loss_val) # DEBUG
            # scheduler.step() # DEBUG
    
    # after training, directly test and compute metrics on the test set
    subprocess.run(['python', '-m', 'animl.test', f'--config={used_config_fpath}'], check=True)
    subprocess.run(['python', 'C:\\Peter\\training-utils\\scripts\\val-test-set.py', used_config_fpath], check=True)

if __name__ == '__main__':
    main()
