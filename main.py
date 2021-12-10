from image_compressor import PCAImageCompressor

if __name__ == '__main__':
    image_path = 'opengenus_logo.png'

    image_compressor = PCAImageCompressor()

    compressed = image_compressor.compress(image_path)
    decompressed = image_compressor.decompress(compressed)

    print(image_compressor.original_image_shape)
    print(compressed.shape)
    print(decompressed.shape)
