"""
包含基于位图的草图分类方法
"""
# Standard Library Imports
import os
import random
import argparse

# Third-party Libraries
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# PyTorch and Torchvision Imports
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, densenet121, mobilenet_v3_large, efficientnet_b0, vgg16, inception_v3
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms


def initialize_model(model_name, num_classes):
    if model_name == "ResNet18":
        model = models.resnet18()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "ResNet50":
        model = models.resnet50()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "DenseNet121":
        model = models.densenet121()
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "DenseNet169":
        model = models.densenet169()
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "MobileNetV3":
        model = models.mobilenet_v3_large()
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "EfficientNetB0":
        model = models.efficientnet_b0()
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "VGG16":
        model = models.vgg16()
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "VGG19":
        model = models.vgg19()
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    else:
        raise ValueError("Invalid model name. Expected one of: ResNet18, ResNet50, DenseNet121, DenseNet169, MobileNetV3, EfficientNetB0, VGG16, VGG19.")
    return model

