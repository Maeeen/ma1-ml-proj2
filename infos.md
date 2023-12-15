# Submissions and pre-trained models

CSV files for the different submissions.

* UNET1: `submissions/unet-marwan1.csv`
  * Submission date: 01/12/2023 @ 3:35
  * Model: `UNet`
  * Model weights (PyTorch CUDA 12.1): `checkpoints/unet-marwan1.pt`
  * Submission file: `submission_marwan1.csv`
  * Submission threshold: `0.25`
  * AIcrowd F1: `0.899`
  * Test F1: `0.848`
  * AIcrowd rank: `3`
  * Applied constrast for test/validation set: coefficient `2×`
  * Training epochs: between 250 and 350
  * Augmentation function used: `augment_data` (see `data_augmentation.py`)

* UNET2: `submissions/unet-marwan2.csv`
  * Submission date: 05/12/2023 @ 3:25
  * Model: `UNet`
  * Model weights (PyTorch CUDA 12.1): `checkpoints/unet-marwan2.pt`
  * Submission file: `submission_marwan2.csv`
  * Submission threshold: `0.1`
  * AIcrowd F1: `0.902`
  * AIcrowd rank: `2`
  * Applied constrast for test/validation set: coefficient `2×`
  * Training epochs: between 250 and 350
  * Augmentation function used: `huge_augment_data` (see `data_augmentation.py`)

* DinkNET: `submissions/dinknet.csv`
  * Submission date: 13/12/2023 @ 23:39
  * Model: `DinkNet`
  * Model weights (PyTorch CUDA 12.1): `checkpoints/dinknet.pt`
  * Submission file: `dinknet.csv`
  * Submission threshold: `0.136`
  * AIcrowd F1: `0.901`
  * AIcrowd rank: `6`
  * Applied constrast for test/validation set: coefficient `1×`
  * Training epochs: 222
  * Augmentation function used: `augment_data` (see `data_augmentation.py`)