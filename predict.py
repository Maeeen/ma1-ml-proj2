import os
import torch
import time

from road_segmentation.models import UNet
from road_segmentation.utils import make_submission

CHECKPOINT_NAME = "UNet_145.pth"

TEST_SET_FOLDER = os.path.abspath("data/test_set_images/")
SUBMISSION_NAME = f"submission_{time.time()}.csv"

SUBMISSION_PATH = os.path.abspath("submissions/" + SUBMISSION_NAME)

model = UNet(n_channels=3, n_classes=1)
model.load_state_dict(torch.load(f"checkpoints/{CHECKPOINT_NAME}"))


make_submission(model, SUBMISSION_PATH, TEST_SET_FOLDER)