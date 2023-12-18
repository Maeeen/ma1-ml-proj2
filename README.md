# Project Road Segmentation

## Git LFS

This project uses Git LFS. Make sure to install it before cloning the repository. See [Git LFS](https://git-lfs.com/).

## Project architecture
The project is organized as follows
```
/checkpoints                # Saved model checkpoints for different architectures
/data                       # Train and test data
/notebooks                  # Past experiments
/src/road_segmentation      # Source code
  /models                   # Models definitions
  /utils                    # Utility functions
  mask_to_submission.py     # Somes functions related to submissions
  submission_to_mask.py     # Somes functions related to submissions
/submissions                # Generated submission files for AIcrowd
infos.md                    # Submissions and checkpoints descriptions
main.ipynb                  # Training code
predict.py                  # Generate predictions on the test set using a trained model. Using Unet by default.
```

## Setting up the environment

First, make sure to clone the repository and to place you on the `main` branch.

### Conda

Dependencies can be managed by conda in the following way. This ensures reproducibility of the environment at the libraries level.

```
conda install -n road-segmentation python=3.10
conda activate road-segmentation
conda env update --file environment.yml --prune

pip install -e .
```

**Note**: make sure to install pytorch with the right version for your system. You can follow the instructions [here](https://pytorch.org/get-started/locally/).

### List of dependencies

* `pytorch`
* `torchvision`
* `numpy`
* `scikit-learn`
* `PIL`
* `scipy`

## Credits

* [DinkNet](https://github.com/zlckanata/DeepGlobe-Road-Extraction-Challenge/blob/master/networks/dinknet.py)
* [Dice Loss](https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch)
* [Pytorch-UNET](https://github.com/milesial/Pytorch-UNet)