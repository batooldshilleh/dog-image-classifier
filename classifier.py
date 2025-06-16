import time
import argparse
import os
from PIL import Image, UnidentifiedImageError
import torch
from torchvision import models, transforms
from torchvision.models import VGG16_Weights, ResNet18_Weights, AlexNet_Weights


def classifier(image_path, model_name):
    """
    Classifies an image using a pretrained CNN model (vgg, resnet, alexnet).
    Returns a string label or None if image is unreadable.
    """
    model_name = model_name.lower()

    if model_name == 'vgg':
        weights = VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
    elif model_name == 'resnet':
        weights = ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    elif model_name == 'alexnet':
        weights = AlexNet_Weights.DEFAULT
        model = models.alexnet(weights=weights)
    else:
        return "invalid model"

    model.eval()
    preprocess = weights.transforms()

    try:
        image = Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        print(f"⚠️ Skipping unsupported image: {image_path} → {e}")
        return None

    image_tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = outputs.max(1)

    class_names = weights.meta["categories"]
    return class_names[predicted.item()].lower()
