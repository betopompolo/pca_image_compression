from file_manager import write_image
from image_compressor import PCAImageCompressor

if __name__ == '__main__':
    image_paths = [
        'logo.png',
        'apple_event.png'
    ]
    n_components_list = [
        8,
        16,
        32,
        64
    ]

    for image_path in image_paths:
        msg = f'{image_path}'
        for n_components in n_components_list:
            image_compressor = PCAImageCompressor()
            compressed = image_compressor.compress(image_path, n_components=n_components)
            decompressed = image_compressor.decompress(compressed)

            compression_rate = image_compressor.get_compression_rate(compressed.shape)
            variance_ratio = image_compressor.get_variance_ratio()
            msg += f"""
    number of components: {n_components}
    compression rate: {compression_rate}
    variance_ratio: {variance_ratio}
---------------"""
            image_file_name = image_path.replace('.png', '')
            write_image(decompressed, output_file_name=f'{image_file_name}{n_components}.png')
        print(msg + '\n')

