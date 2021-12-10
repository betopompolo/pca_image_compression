import matplotlib.image as image

Image = image


def open_image(path: str) -> Image:
    return Image.imread(f'dataset/{path}')
