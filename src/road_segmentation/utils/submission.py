from road_segmentation.mask_to_submission import masks_to_submission
from road_segmentation.augmentation import augment_contrast
from .image import img_float_to_uint8, load_image

from PIL import Image
import torch.nn as nn
import numpy as np

NUMBER_OF_TEST = 50

def make_submission(model, device, img_transform, file_path, test_folder, constrast=2, threshold=0.1):
    for i in range(1, NUMBER_OF_TEST+1):
        img = load_image(test_folder + "/test_" + str(i) + "/" + "test_" + str(i) + ".png")
        display(Image.open(test_folder + "/test_" + str(i) + "/" + "test_" + str(i) + ".png"))
        img, _ = augment_contrast([img], None, constrast)
        img = img[0]
        img = img_transform(img).to(device)
        model.eval()
        output = nn.Sigmoid()(model.to(device)(img.unsqueeze(0)).cpu().detach()).numpy()
        output = img_float_to_uint8(output.squeeze())
        output_image = Image.fromarray(output)
        display(output_image)
        output_image.save("submissions/temp/" + str(i) + ".png")

    masks_to_submission(file_path, *["submissions/temp/" + str(i) + ".png" for i in range(1, NUMBER_OF_TEST+1)])