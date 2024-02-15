"""
@author: <nktoan163@gmail.com>
"""
import os
from model import CNN
from dataset import PesplanusDataset
import argparse
from tqdm.autonotebook import tqdm
import shutil
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings("ignore")


def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="viridis")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def args_parser():
    parser = argparse.ArgumentParser(description='Pesplanus training train script')
    parser.add_argument('-r', '--root', type=str, default='Pesplanus.v2i.multiclass', help='Root')
    parser.add_argument('-b', '--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('-s', '--image_size', type=int, default=224, help='image size (default:224)')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('-c', '--checkpoint_path', type=str, default=None, help='checkpoint path')
    parser.add_argument('-t', '--tensorboard_path', type=str, default="tensorboard_logs", help='tensorboard path')
    parser.add_argument('-tr', '--trained_path', type=str, default="trained_model", help='trained model path')
    args = parser.parse_args()
    return args


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = Compose([
        ToTensor(),
        Resize((args.image_size, args.image_size)),
    ])

    # Define training data
    train_set = PesplanusDataset(root=args.root, train=True, transform=transform)
    valid_set = PesplanusDataset(root=args.root, train=False, transform=transform)

    # Dataloader
    train_params = {
        'batch_size': args.batch_size,
        'num_workers': 6,
        'shuffle': True,
        'drop_last': True
    }

    valid_params = {
        'batch_size': args.batch_size,
        'num_workers': 6,
        'shuffle': False,
        'drop_last': True
    }

    train_loader = DataLoader(train_set, **train_params)
    valid_loader = DataLoader(valid_set, **valid_params)

    # Model
    model = CNN(num_classes=len(PesplanusDataset().classes)).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    # Define learning rate at specific steps
    scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

    # Load saved model (if any)
    if args.checkpoint_path and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
    else:
        start_epoch = 0
        best_accuracy = 0

    # Tensorboard
    if os.path.isdir(args.tensorboard_path):
        shutil.rmtree(args.tensorboard_path)
    os.mkdir(args.tensorboard_path)

    # Train checkpoint
    if not os.path.isdir(args.trained_path):
        os.mkdir(args.trained_path)

    writer = SummaryWriter(args.tensorboard_path)
    num_iters = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        # Train
        model.train()
        losses_train = []
        progress_bar = tqdm(train_loader, colour='green')

        for iter, (images, labels) in enumerate(progress_bar):
            # Move tensor to configured device:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            predictions = model(images)
            loss = criterion(predictions, labels)

            # Backward
            loss.backward()
            optimizer.step()
            loss_t = loss.item()
            progress_bar.set_description('Epoch: {}/{}. Loss_value: {:.4f}'.format(epoch + 1, args.epochs, loss_t))
            losses_train.append(loss_t)
            writer.add_scalar("Train/Loss", np.mean(losses_train), epoch * num_iters + iter)

        # Validate
        model.eval()
        losses_valid = []
        all_predictions = []
        all_gts = []
        with torch.no_grad():
            for iter, (images, labels) in enumerate(valid_loader):
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images)

                # Foward pass
                _, max_idx = torch.max(predictions, 1)

                loss_v = criterion(predictions, labels)
                losses_valid.append(loss_v.item())
                all_gts.extend(labels.tolist())
                all_predictions.extend(max_idx.tolist())

        writer.add_scalar("Val/Loss", np.mean(losses_valid), epoch)
        acc = accuracy_score(all_gts, all_predictions)
        writer.add_scalar("Val/Accuracy", acc, epoch)
        conf_matrix = confusion_matrix(all_gts, all_predictions)
        plot_confusion_matrix(writer, conf_matrix, [i for i in train_set.classes], epoch)

        # Dictionary checkpoint
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_accuracy": best_accuracy,
            "batch_size": args.batch_size
        }

        # Save model
        torch.save(checkpoint, os.path.join(args.trained_path, 'last.pt'))
        if acc > best_accuracy:
            torch.save(checkpoint, os.path.join(args.trained_path, 'best.pt'))
            best_accuracy = acc

        scheduler.step()

if __name__ == '__main__':
    args = args_parser()
    train(args)
