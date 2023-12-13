# Submissions

CSV files for the different submissions.

When graded on AICrowd, try to rename them with their corresponding scores (and submission date).

* `submission_marwan1.csv`: 01/12/2023 @ 3:35, model weights: marwan1.pth, UNet, submission threshold: .25, score: 0.899, applied constrast for test set (coeff 2*), 250-350 epochs
* `0.902 (submission very augmented dataset).csv`: 05/12/2023 @ 3:25, model weights: same name, UNet, huge augmentation, submission threshold: .1, post-contrast=2
  * The model was trained on a very augmented dataset (see `data_augmentation.py`), with 250-350 epochs.
  * However, the `.pth` file seems to be corrupted.