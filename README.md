# Project Road Segmentation

## Project architecture

The project is organized as follows:
* The main part is inside the `main.ipynb` notebook.
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