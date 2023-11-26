import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import glob
from datetime import datetime
import argparse

import torch
import torch.nn as nn
from torchvision.transforms import v2
import torchvision.transforms.functional as TF
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

from utils.utilities import plot
from utils.preprocessing import read_data, Normalize
from utils.training_utils import AMDataset, transforms, train_one_epoch, validation

from model import Le_net_AMD_classifier


def main(args):

    ADMISSIBLE_LABELS = ['A','N']
    print('starting: ')

    images, labels = read_data(args.train_dir, admissible_labels=ADMISSIBLE_LABELS)
    print('data has been read')

    images = Normalize(images, path_load=args.normalize_config_load, path_save=args.normalize_config_save)
    print('data has been normalized')

    images_train, images_val, labels_train, labels_val = train_test_split(images, labels,test_size=0.15,random_state=666)
    training_data = AMDataset(images_train,labels_train, transforms, train=True)
    validation_data = AMDataset(images_val,labels_val,transforms, train=False)

    train_dataloader = DataLoader(training_data, batch_size=args.batch_size,shuffle=True)
    valid_dataloader = DataLoader(validation_data, batch_size=args.batch_size,shuffle=False)

    model = Le_net_AMD_classifier(args.dropout_rate)
    if args.model_trained_path:
        model.load_state_dict(torch.load(args.model_trained_path))

    adam = torch.optim.Adam(model.parameters(),lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/amd_classifier_{}'.format(timestamp))

    best_vloss = 1_000_000.

    for epoch in range(args.epochs):
        model.train(True)
        avg_loss, avg_acc = train_one_epoch(model, train_dataloader, epoch ,loss_fn, adam, writer, device)

        avg_vloss, avg_vacc, _, _ = validation(model, valid_dataloader, loss_fn, device)
        print(f"Epoch {epoch} training performance: Loss={avg_loss}   Accuracy={avg_acc}")
        print(f"Epoch {epoch} validation performance: Loss={avg_vloss}   Accuracy={avg_vacc}")

        writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch + 1)
        writer.flush()

        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'version_fixed/model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)

    if args.test_dir:
        images_test, labels_test = read_data(args.test_dir, admissible_labels=ADMISSIBLE_LABELS)
        images_test = Normalize(images_test, path_load=args.normalize_config_load)
        
        test_data = AMDataset(images_test,labels_test,transforms, train=False)
        test_dataloader = DataLoader(test_data, batch_size=args.batch_size,shuffle=False)

        avg_tloss, avg_tacc, _, _ = validation(model, test_dataloader, loss_fn, device)
        print(print(f"Test performance: Loss={avg_tloss}   Accuracy={avg_tacc}"))

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('train_dir', type=str, help='Input data dir')
    parser.add_argument('--test-dir', type=str, help='Input data dir')
    parser.add_argument('--model-trained-path', '-m', type=str, default=None, help='trained model path')
    parser.add_argument('--batch-size', '-b', type=int, default=8)
    parser.add_argument('--epochs', '-e', type=int, default=10)
    parser.add_argument('--dropout-rate', '-d', type=float, default=0.1)
    parser.add_argument('--learning-rate', type=float, default=0.0005)
    parser.add_argument('--weight-decay', type=float, default=0.001)
    parser.add_argument('--normalize-config-load', '-nl',type=str, default=None, help='path of normalizing config')
    parser.add_argument('--normalize-config-save', '-ns',type=str, default=None, help='path where we want to save normalizing config')

    # Parse the command-line arguments
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
