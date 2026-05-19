import torch
from msdflow.model.resnet_classifier import ResNetMini
from tqdm import tqdm
import inspect
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from scipy.linalg import sqrtm
import numpy as np
import torch.nn as nn

import torch.nn.functional as F
from torch.distributions import kl_divergence
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
T = 300
epochs = 50
batch_size = 128


def load_classifier(model_path="checkpoints/classifier_mnist_resnet.pth", device=None):
    """
    Initializes the ResNetMini architecture and loads saved weights.
    
    Args:
        model_path (str): Path to the .pth file.
        device (str): 'cuda' or 'cpu'. Auto-detects if None.
        
    Returns:
        torch.nn.Module: The loaded model in eval mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # 1. Initialize the architecture
    model = ResNetMini()

    # 2. Load the state dictionary
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model from {model_path}")
    except FileNotFoundError:
        print(f"Error: {model_path} not found. Did you run train_classifier.py first?")
        return None

    # 3. Move to device and set to evaluation mode
    model.to(device)
    model.eval()
    
    return model

    

def Inception_score(classifier, sample_images):
    classifier.eval()

    # Compute p_dis(.|x_i): PDF of labels conditioned on images
    with torch.no_grad():
        res = classifier(sample_images)
        prob_yx = F.softmax(res, dim=1) 

    # Estimate the marginal integral
    py_marginal = prob_yx.mean(dim=0, keepdim=True)

    # Compute KL-divergence and exponentiate to get Inception score
    kl_div = (prob_yx*(prob_yx.log()-py_marginal.log())).sum(dim=1)
    avg_kl = kl_div.mean()
    Inception_score = torch.exp(avg_kl)
    return Inception_score


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = F.relu(self.model.bn1(self.model.conv1(x)))
        out = self.model.layer1(out)
        out = self.model.layer2(out)
        out = self.model.layer3(out)
        out = F.avg_pool2d(out, out.size()[2:])  # same as forward()
        out = out.view(out.size(0), -1)          # now (N, 10)
        return out  # stop before linear


def FID(classifier, sample_images, ref_images):
    # Remove classification layer and extract features
    feature_extractor = FeatureExtractor(classifier)
    feature_extractor.eval()

    with torch.no_grad():
        real_features = feature_extractor(ref_images)
        gen_features = feature_extractor(sample_images)

        # Flatten to 2D
        real_features = real_features.view(real_features.shape[0], -1).cpu().numpy()
        gen_features = gen_features.view(gen_features.shape[0], -1).cpu().numpy()

    # Calculate mean and covariance for fit to Gaussian
    mu_r, sigma_r = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu_g, sigma_g = gen_features.mean(axis=0),  np.cov(gen_features,  rowvar=False)

    # Model the feature distributions as Gaussian and calculate the Wasserstein-2 distance
    diff = mu_r - mu_g
    covmean = sqrtm(sigma_r @ sigma_g)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return fid
