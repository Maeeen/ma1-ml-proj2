# Project Road Segmentation

The main part is inside the `main.ipynb` notebook.

## Submissions

About the submissions, see the `submissions/README.md` file. You will find pre-trained models.

## Setting up the environment

First, make sure to clone the repository and to place you on the `main` branch.

## Conda

Dependencies can be managed by conda in the following way. This ensures reproducibility of the environment at the libraries level.

```
conda install -n road-segmentation python=3.10
conda activate road-segmentation
conda env update --file environment.yml --prune

pip install -e .
```