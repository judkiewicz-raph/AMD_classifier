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
from utils.training_utils import get_accuracy

from sklearn.metrics import RocCurveDisplay, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay

from model import Le_net_AMD_classifier


def main(args):

    ADMISSIBLE_LABELS = ['A','N']
    print('starting: ')

    loss_fn = nn.BCELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Le_net_AMD_classifier(0.)
    if args.model_trained_path:
        model.load_state_dict(torch.load(args.model_trained_path))
        print("model has been loaded")

    images_test, labels_test = read_data(args.test_dir, admissible_labels=ADMISSIBLE_LABELS)
    images_test = Normalize(images_test, path_load=args.normalize_config_load)
    
    test_data = AMDataset(images_test,labels_test,transforms, train=False)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size,shuffle=False)

    avg_tloss, avg_tacc, t_labels, t_probs = validation(model, test_dataloader, loss_fn, device, threshold=args.prob_treshold)

    t_labels = [int(l[0]) for l in t_labels]
    t_probs = [p[0] for p in t_probs]
    t_preds = (np.array(t_probs)>args.prob_treshold).astype(float)
    precision, recall,f1_score,_ = precision_recall_fscore_support(t_labels, t_preds, average='weighted')
    cm = confusion_matrix(t_labels, t_preds)

    print(f"The test loss is {avg_tloss}.")
    print(f"The accuracy on the test dataset is {avg_tacc} and is achieved for a probability threshold of {args.prob_treshold}.")
    print(f"The precision is {precision}, the recall is {recall} and the F1 score is {f1_score}")

    if args.show_plots:
        RocCurveDisplay.from_predictions(
            t_labels,
            t_probs,
            name=f"AMD vs healthy",
            color="darkorange",
            plot_chance_level=True,
        )
        plt.axis("square")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("AMD vs Healthy")
        plt.legend()
        plt.show()

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('test_dir', type=str, help='Input data dir')
    parser.add_argument('--model-trained-path', '-m', type=str, default=None, help='trained model path')
    parser.add_argument('--batch-size', '-b', type=int, default=8)
    parser.add_argument('--normalize-config-load', '-nl',type=str, default=None, help='path of normalizing config')
    parser.add_argument('--prob-treshold', '-pt',type=float, default=0.5, help="The probability threshold used for predicting the label based on the probability.")
    parser.add_argument('--show-plots', '-sp',type=bool, default=True, help="Show performance plots.")

    # Parse the command-line arguments
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)