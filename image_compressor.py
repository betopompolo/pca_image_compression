from functools import reduce

import numpy as np
from sklearn.decomposition import PCA

from file_manager import open_image, Image


class PCAImageCompressor:
    def __init__(self):
        self.original_image_shape = None
        self.pca = None

    def compress(self, image_path: str, n_components: int) -> Image:
        self.pca = PCA(n_components)

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

    def get_compression_rate(self, compressed_shape: tuple[int, int]) -> float:
        original_image_size = reduce(lambda x, y: x * y, self.original_image_shape)
        compressed_image_size = reduce(lambda x, y: x * y, compressed_shape)
        return 1 - (compressed_image_size / original_image_size)

    def get_variance_ratio(self):
        return np.sum(self.pca.explained_variance_ratio_)
