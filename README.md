# Project Road Segmentation

## Git LFS

This project uses Git LFS. Make sure to install it before cloning the repository. See [Git LFS](https://git-lfs.com/).

## Project architecture

The project is organized as follows:
* The main part for training is inside the `main.ipynb` notebook.
* `predict.py` is used to generate predictions on the test set, with the best model (UNet, see `submissions/README.md`).
* Content of the project is in the `src/road_segmentation` folder.
  * `models` contains the different models used.
  * `utils` contains the different utilities used.
  * `root` of `src/road_segmentation` contains diverse functions.

## Submissions and pre-trained models

About the submissions, see the `submissions/README.md` file. Here lies as well pre-trained models.

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