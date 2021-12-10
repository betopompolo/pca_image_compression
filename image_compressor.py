import numpy as np
from sklearn.decomposition import PCA

from file_reader import open_image, Image


class PCAImageCompressor:
    def __init__(self, n_components=32):
        self.original_image_shape = None
        self.pca = PCA(n_components)

    def compress(self, image_path: str) -> Image:
        image = open_image(image_path)
        image_channel_count = 3
        self.original_image_shape = image.shape
        reshaped_image = np.reshape(image, (image.shape[0], image.shape[1] * image_channel_count))
        compressed_image = self.pca.fit_transform(reshaped_image)
        return compressed_image

    def decompress(self, image: Image) -> Image:
        inverse_transformation = self.pca.inverse_transform(image)
        decompressed_image = np.reshape(inverse_transformation, self.original_image_shape)
        return decompressed_image
