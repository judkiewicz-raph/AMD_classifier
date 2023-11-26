from typing import Sequence
import random
import numpy as np

import torch
import torchvision.transforms.functional as TF
import torch.nn as nn
from torchvision.transforms import v2
from torch.utils.data import Dataset
from tqdm import tqdm

class MyRotateTransform:
    """
    Custom rotation transform class smapling rotation angle from a list.

    Args:
        angles (Sequence[int]): A sequence of integers representing the angles to choose from for rotation.

    Returns:
        Callable: A callable object that applies a random rotation to the input.

    Example:
        transform = MyRotateTransform([0, 90, 180, 270])
        rotated_image = transform(image)
    """
    def __init__(self, angles: Sequence[int]):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)
    

class AMDataset(Dataset):
    """
    Custom dataset class for Age-related Macular Degeneration (AMD) images.

    Args:
    images (list): A list of input images.
    labels (list): A list of corresponding labels.
    transform (callable, optional): A callable object that applies transformations to the input images. Default is None.
    train (bool, optional): Whether the dataset is used for training. Default is None.

    Returns:
    tuple: A tuple containing the image and its corresponding label.

    Example:
    dataset = AMDataset(images, labels, transform=transforms, train=True)
    image, label = dataset[0]
    """
    def __init__(self, images, labels, transform=None, train=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = int(self.labels[idx]=='A')
        if self.transform:
            image = self.transform(self.train)(image)
        return image, label
    

def transforms(train):
    """
    Get the data augmentation transforms based on the train flag.

    Args:
        train (bool): Whether the transforms are for training or not.

    Returns:
        torchvision.transforms.Compose: A composition of data augmentation transforms.
        """
    if train:
        return  v2.Compose([           
        MyRotateTransform([90,180,270]),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ElasticTransform(300,20),
        v2.Normalize((0.5,), (0.5,))
    ])
    else:
        return v2.Compose([
        v2.Normalize((0.5,), (0.5,))    
    ])

def get_accuracy(probs, labels, thresh=0.5, from_tensor=True):
    """
    Calculate the accuracy based on predicted probabilities and true labels.

    Args:
        probs (list or ndarray): A list or ndarray of predicted probabilities.
        labels (list or ndarray): A list or ndarray of true labels.
        thresh (float): The threshold value for classification.
        from_tensor (bool): True if accuracy computed from tensor, False if computed from ndarray.

    Returns:
        float: The accuracy value.
    """
    # preds = (np.array(probs)>thresh)
    if from_tensor:
        preds = (probs>thresh).type(torch.DoubleTensor)
    else:
        preds = (np.array(probs)>thresh)
    acc = (preds==labels).sum()/len(labels)
    return acc.item()

def train_one_epoch(model, train_dataloader, epoch_idx,loss_fn, optimizer, tb_writer, device):
    """
    Train the model for one epoch using the provided dataloader.

    Args:
        model (nn.Module): The model to train.
        train_dataloader (DataLoader): The dataloader for training data.
        epoch_idx (int): The index of the current epoch.
        loss_fn: The loss function to use for training.
        optimizer: The optimizer to use for training.
        tb_writer: The tensorboard writer for logging.
        device: The device to use for training.

    Returns:
        tuple: A tuple containing the last loss and the last accuracy.
    """

    running_loss = 0.
    last_loss = 0.
    running_acc = 0.
    last_acc = 0.

    with tqdm(train_dataloader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch_idx}")
        for i,(inputs,labels) in enumerate(tepoch):

            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.type(torch.float).unsqueeze(1)
            optimizer.zero_grad()

            outputs = model(inputs)

            sig = nn.Sigmoid()
            probs = sig(outputs)

            loss = loss_fn(probs, labels)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()

            acc = get_accuracy(probs,labels)
            running_acc+= acc

            tepoch.set_postfix(loss=loss.item(), accuracy=acc)
             
            if i % 3 == 2:
                last_loss = running_loss / 3 # loss per batch
                last_acc = running_acc / 3 # loss per batch
                print('batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_idx * len(train_dataloader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                tb_writer.add_scalar('Accuracy/train', last_acc, tb_x)
                running_loss = 0.
                running_acc = 0.
    
    return last_loss, last_acc

def validation(model, dataloader, loss_fn, device, threshold=0.5):
    """
    Perform validation on the model using the provided dataloader.

    Args:
        model (nn.Module): The model to validate.
        dataloader (DataLoader): The dataloader for validation data.
        loss_fn: The loss function to use for validation.
        device: The device to use for validation.

    Returns:
        tuple: A tuple containing the average validation loss, average validation accuracy, labels and predictions returned by the model.
    """
    running_vloss = 0.0
    running_vacc = 0.0
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()

    v_labels = []
    v_preds = []

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, (vinputs, vlabels) in enumerate(dataloader):
            vinputs, vlabels = vinputs.to(device), vlabels.to(device)
            vlabels = vlabels.type(torch.float).unsqueeze(1)
            v_labels.extend(vlabels.numpy())
            voutputs = model(vinputs)
            vprobs = nn.Sigmoid()(voutputs)
            v_preds.extend(vprobs.numpy())
            vloss = loss_fn(vprobs, vlabels)
            running_vloss += vloss
            vacc = get_accuracy(vprobs,vlabels, thresh=threshold)
            running_vacc+= vacc

    avg_vloss = running_vloss / (i + 1)
    avg_vacc = running_vacc / (i + 1)

    return avg_vloss, avg_vacc, v_labels, v_preds
