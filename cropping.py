import os.path
from PIL import Image
import itertools
import matplotlib
from matplotlib import pyplot as plt, image as mpimg
matplotlib.use('TkAgg')  # Use a different backend, e.g., 'TkAgg'
import fireClassification
from PIL import Image, ImageDraw


def plot_images():
    # List of image file names

    image_names = os.listdir(os.path.join('cropped_file_data', 'test', 'Fire'))

    # Set up subplots
    num_images = len(image_names)
    num_rows = 1
    num_cols = num_images

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4))

    # Iterate through images and plot them
    for i, image_name in enumerate(image_names):
        image_path = os.path.join(os.path.join('cropped_file_data', 'test', 'Fire'),
                                  image_name)  # Replace 'path_to_images_folder' with the actual path
        img = mpimg.imread(image_path)
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(image_name)

    plt.show()

def main():
    horizontal_windows_amount = 6
    vertical_windows_amount = 4
    input_images_dir = os.path.join('fire_data')
    for data_type in os.listdir(input_images_dir):
        for class_type in os.listdir(os.path.join(input_images_dir, data_type)):
            for image_name in os.listdir(os.path.join(input_images_dir, data_type, class_type)):
                image_path = os.path.join(input_images_dir, data_type, class_type, image_name)
                crop_image_and_save_windows(image_path, horizontal_windows_amount, vertical_windows_amount,
                                            data_type, class_type, image_name)


def crop_image_and_save_windows(image_path, horizontal_windows_amount, vertical_windows_amount,
                                data_type, class_type, image_name):
    im = Image.open(image_path)
    if im.mode != 'RGB':
        im = im.convert('RGB')

    window_width = im.size[0] / horizontal_windows_amount
    window_height = im.size[1] / vertical_windows_amount
    for i, j in itertools.product(range(horizontal_windows_amount), range(vertical_windows_amount)):
        box = (i * window_width, j * window_height,
               (i + 1) * window_width, (j + 1) * window_height)
        cropped_image = im.crop(box)
        # cropped_image.show()
        cropped_image.save(os.path.join('cropped_file_data', data_type, class_type,
                                        '{}_{}_{}.jpg'.format(image_name.split('.')[0], i, j)))




if __name__ == '__main__':
    # plot_images()
    main()

