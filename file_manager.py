import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np

Image = image


def open_image(path: str) -> Image:
    return Image.imread(f'dataset/{path}')


def write_image(img: Image, output_file_name: str):
    image_path = f'output/{output_file_name}'

    normalized_image = np.array((img - np.min(img)) / (np.max(img) - np.min(img)))
    plt.imsave(image_path, normalized_image)
