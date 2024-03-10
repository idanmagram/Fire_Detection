import torch
from IPython.display import display, HTML
import numpy as np
from PIL import Image
import matplotlib
from segment_anything.segment_anything import sam_model_registry, SamPredictor
matplotlib.use('TkAgg')  # Use the TkAgg backend
import matplotlib.pyplot as plt

# TODO: function return how many points user picked and [x,y] of each point
from matplotlib import patches

def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height

def choosing_points_using_click_Idan_code(length, width, rows, columns, input_array, sliding_windows):
    input_labels = np.array([])
    input_points = np.array([]).reshape(0, 2)
    # square_width = width / rows
    # square_length = length / columns

    i = 0
    j = 0


    for window in sliding_windows:
        if j < len(input_array[0]) - 1:
            j += 1
        elif i < len(input_array) - 1:
            i += 1
            j = 0
        x1, y1, x2, y2 = window
        # cv2.rectangle(image, (x1, y1), (x2, y2), color.tolist(), 2)
        middle_x = (x1 + x2) // 2
        middle_y = (y1 + y2) // 2
        input_points = np.vstack([input_points, [middle_x, middle_y]])
        if input_array[i][j] == 1:
            input_labels = np.append(input_labels, 1)
        else:
            input_labels = np.append(input_labels, 0)
    return input_points, input_labels


    # for i in range(rows):
    #     for j in range(columns):
    #         center_x = (j + 0.5) * square_length
    #         center_y = (i + 0.5) * square_width
    #         input_points = np.vstack([input_points, [center_x, center_y]])
    #         if input_array[i][j] == 1:
    #             input_labels = np.append(input_labels, 1)
    #         else:
    #             input_labels = np.append(input_labels, 0)
    # return input_points, input_labels


def choosing_point_using_box():
    input_box = np.array([0, 40, 100, 0])
    return input_box


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def segement(fire_vector, horizontal_windows_amount, vertical_windows_amount, image_path, image, windows):
    input_labels = np.array([])
    input_points = np.array([]).reshape(0, 2)

    # image_width, image_height = get_image_dimensions(image_path)
    im = Image.open(image_path)
    image_width = im.size[0]
    image_height = im.size[1]
    # def choosing_points_using_click_Idan_code(length, width, rows, columns, input_array):
    input_points, input_labels = choosing_points_using_click_Idan_code(image_width,image_height ,
                                                        horizontal_windows_amount, vertical_windows_amount, fire_vector, windows)
    print(input_points)
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    predictor = SamPredictor(sam)
    image_np = np.array(image)

    predictor.set_image(image_np)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_points, input_labels, plt.gca())
    plt.axis('on')
    plt.show()

    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=True,
        # FIXME: for uncomment next line
        # box=box
    )

    masks.shape  # (number_of_masks) x H x W

    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_points, input_labels, plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()