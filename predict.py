import os
import torch
import time
import numpy as np

from road_segmentation.utils import *
from road_segmentation.models import UNet

from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = torch.device(device)

CHECKPOINT_NAME = "unet-marwan2.pt"

IMAGE_SRC_DIR = "data/training/images/"

TEST_SET_FOLDER = os.path.abspath("data/test_set_images/")
SUBMISSION_NAME = f"submission_{time.time()}.csv"

SUBMISSION_PATH = os.path.abspath("submissions/" + SUBMISSION_NAME)

model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load(f"checkpoints/{CHECKPOINT_NAME}", device))

IMAGES_SRC = np.asarray(read_all_images(IMAGE_SRC_DIR))
means, stds = IMAGES_SRC.mean(axis=(0, 1, 2)), IMAGES_SRC.std(axis=(0, 1, 2))

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means, stds)
])

# Use best contrast and threshold values
make_submission(model, device, img_transform, SUBMISSION_PATH, TEST_SET_FOLDER, constrast=2, threshold=0.1)