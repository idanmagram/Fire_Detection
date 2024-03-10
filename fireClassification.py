import itertools
import shutil
from multiprocessing import freeze_support

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib
matplotlib.use('TkAgg')  # Use a different backend, e.g., 'TkAgg'
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from PIL import Image, ImageDraw
import sam


def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def train_model(model, criterion, optimizer, scheduler,
                dataset_sizes, dataloaders, device, num_epochs):
    since = time.time()

    # Create a temporary directory to save training checkpoints
    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                # deep copy the model
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        # load best model weights
        model.load_state_dict(torch.load(best_model_params_path))
        shutil.copy(best_model_params_path, 'final_params.pt')

    return model


def visualize_model(model, dataloaders, device, class_names, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def test_model(model, dataloaders, device, dataset_sizes, criterion):
    model.eval()  # Set model to evaluate mode
    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        # optimizer.zero_grad()

        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / dataset_sizes['test']
    epoch_acc = running_corrects.double() / dataset_sizes['test']

    print(f'test Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


def visualize_model_predictions(model, data_transforms, device, class_names, img_path):
    was_training = model.training
    model.eval()

    img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        ax = plt.subplot(2, 2, 1)
        ax.axis('off')
        ax.set_title(f'Predicted: {class_names[preds[0]]}')
        imshow(img.cpu().data[0])

        model.train(mode=was_training)

def predict_image(model, data_transforms, device, class_names, img):
    was_training = model.training
    model.eval()

    # img = Image.open(img_path)
    img = data_transforms['val'](img)
    img = img.unsqueeze(0)
    img = img.to(device)

    with torch.no_grad():
        outputs = model(img)
        _, preds = torch.max(outputs, 1)

        return class_names[preds[0]]

def construct_borders(model, data_transforms, device, class_names, horizontal_windows_amount, vertical_windows_amount):
    input_images_dir = os.path.join('fire_data')
    for data_type in os.listdir(input_images_dir):
        for class_type in os.listdir(os.path.join(input_images_dir, data_type)):
            for image_name in os.listdir(os.path.join(input_images_dir, data_type, class_type)):
                image_path = os.path.join(input_images_dir, data_type, class_type, image_name)
                im = Image.open(image_path)
                image_copy = im.copy()
                fire_vector = np.zeros((vertical_windows_amount, horizontal_windows_amount))
                if image_copy.mode != 'RGB':
                    image_copy = image_copy.convert('RGB')

                window_width = image_copy.size[0] / horizontal_windows_amount
                window_height = image_copy.size[1] / vertical_windows_amount
                for i, j in itertools.product(range(horizontal_windows_amount), range(vertical_windows_amount)):

                    cropped_image = os.path.join('cropped_file_data', data_type, class_type,
                                                    '{}_{}_{}.jpg'.format(image_name.split('.')[0], i, j))
                    pred = predict_image(model, data_transforms, device, class_names, cropped_image)
                    if pred == 'Fire':
                        print(i,j)
                        fire_vector[j][i] = 1
                        draw = ImageDraw.Draw(image_copy)
                        # Draw a rectangle around the specified area
                        draw.rectangle([i * window_width, j * window_height, (i + 1) * window_width, (j + 1) * window_height], outline="green")
                image_copy.save(os.path.join('fire_data_detection', data_type, class_type,
                                                '{}_detect.jpg'.format(image_name.split('.')[0])))
                print(fire_vector)
                sam.segement(fire_vector, horizontal_windows_amount, vertical_windows_amount, image_path, im)
                return

def construct_borders_one_image(model, data_transforms, device, class_names, horizontal_windows_amount, vertical_windows_amount, im, image_path):
    image_height, image_width = get_image_dimensions(image_path)
    window_width = (int)(im.size[0] / (horizontal_windows_amount))
    window_height = (int)(im.size[1] / (vertical_windows_amount))

    height_offset = (int)(window_height / 4)
    width_offset = (int)(window_width / 4)


    windows = []
    k = 0
    w = 0
    for i in range(0, image_width - window_width + width_offset*2, width_offset):
        for j in range(0, image_height - window_height + height_offset*2, height_offset):
            window = (i, j, i + window_width, j + window_height)
            windows.append(window)
            k += 1
        w += 1
    draw_sliding_windows(im, windows, image_path)

    fire_vector = np.zeros((w, (int)(k/w)))
    i = 0
    j = 0
    for window in windows:
        cropped_image = im.crop(window)
        if j < (k/w - 1):
            j += 1
        elif i < (w - 1):
            i += 1
            j = 0

    # window_width = im.size[0] / (horizontal_windows_amount)
    # window_height = im.size[1] / (vertical_windows_amount)
    # for i, j in itertools.product(range(w), range(k/w)):
    #
    #     box = (i * window_width, j * window_height,
    #            (i + 1) * window_width, (j + 1) * window_height)
    #     cropped_image = im.crop(box)

        pred = predict_image(model, data_transforms, device, class_names, cropped_image)
        if pred == 'Fire':
            print(i,j)
            fire_vector[i][j] = 1
            # draw = ImageDraw.Draw(im)
            # Draw a rectangle around the specified area
            # draw.rectangle([i * window_width, j * window_height, (i + 1) * window_width, (j + 1) * window_height], outline="green")

    # print(fire_vector.transpose())


    fire_vector_new = process_matrix(fire_vector)
    # print(" ")
    # print(fire_vector_new)

    sam.segement(fire_vector_new, vertical_windows_amount, horizontal_windows_amount, image_path, im, windows)
    return




def main(already_trained=True):
    cudnn.benchmark = True
    plt.ion()

    image_path = 'fire_data\\train\\Fire\\F_13.jpg'

    #img = Image.open(image_path)
    # If you want to display the image, you can add the following line
    #img.show()

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'fire_data'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                              data_transforms[x])
                      for x in ['train', 'val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                 shuffle=True, num_workers=4)
                  for x in ['train', 'val', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['train']))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    # Uncomment the line below if you want to visualize the batch
    # imshow(out, title=[class_names[x] for x in classes])

    model_ft = models.resnet18(weights='IMAGENET1K_V1')
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    if not already_trained:
        print('Start training model')
        model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                               dataset_sizes, dataloaders, device, num_epochs=4)
    else:
        print('Did not train model, load final_params')
        model_ft.load_state_dict(torch.load('final_params.pt'))

    # visualize_model(model_ft, dataloaders, device, class_names)

    plt.ioff()
    plt.show()
    # visualize_model_predictions(model_ft, data_transforms, device, class_names,
    #                              'cropped_file_data\\test\\Fire\\F_1006_1_0_Non_Fire.jpg')


    # Go test
    # test_model(model_ft, dataloaders, device, dataset_sizes, criterion)
    horizontal_windows_amount = 6
    vertical_windows_amount = 4
    # construct_borders(model_ft, data_transforms, device, class_names, horizontal_windows_amount, vertical_windows_amount)
    image_path = 'fire_data\\train\\Fire\\F_561.jpg'
    im = Image.open(image_path)

    construct_borders_one_image(model_ft, data_transforms, device, class_names, horizontal_windows_amount,
                                vertical_windows_amount, im, image_path)




def get_image_dimensions(image_path):
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def generate_sliding_windows(image_height, image_width):
    window_size_height = (int)(image_height / 2)
    window_size_width = (int)(image_width//2)

    height_offset = (int)(image_height / 4)
    width_offset = (int)(image_width / 4)

    # print("image_height:", image_height)
    # print("image_width:", image_width)
    # print("window_size_height:", window_size_height)
    # print("window_size_width:", window_size_width)

    windows = []
    k = 0
    for i in range(0, image_width - window_size_width + 1, width_offset):
        for j in range(0, image_height - window_size_height + 1, height_offset):
            window = (i, j, i + window_size_width, j + window_size_height)
            windows.append(window)
            k += 1

    return windows

def draw_sliding_windows(image, sliding_windows, image_path):
    i= 0
    remember_color = None
    image = plt.imread(image_path)
    plt.imshow(image)


    input_points = np.array([]).reshape(0, 2)
    for window in sliding_windows:
        x1, y1, x2, y2 = window
        color = np.random.randint(0, 255, size=3)  # Generate a random color
        # cv2.rectangle(image, (x1, y1), (x2, y2), color.tolist(), 2)
        middle_x = (x1 + x2) // 2
        middle_y = (y1 + y2) // 2
        # cv.circle(image, (middle_x, middle_y), 10, color.tolist(), 2)
        circle = plt.Circle((middle_x, middle_y), 5, color='green', fill=False)
        plt.gca().add_patch(circle)

        input_points = np.vstack([input_points, [middle_x, middle_y]])

    print(input_points)
    print("!!!" ,sliding_windows[0])
    print("len", len(sliding_windows))
    plt.show()

    # cv2.rectangle(image, (sliding_windows[0][0], sliding_windows[0][1]), (sliding_windows[0][2], sliding_windows[0][3]), remember_color.tolist(), 2)



def process_matrix(matrix):
    # Create a copy of the matrix to avoid modifying the original
    new_matrix = np.zeros_like(matrix)

    # Perform a deep copy of matrix into new_matrix
    new_matrix[:] = matrix
    print("new materix\n")
    print(new_matrix)

    # Define the eight possible directions around a cell
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

    # Iterate over each cell in the matrix
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            # If the cell is a 1, check its neighbors
            if matrix[i][j] == 1:
                found_zero_neighbor = False
                for dx, dy in directions:
                    nx, ny = i + dx, j + dy
                    # If any neighbor is out of bounds or a 0, mark that we found a zero neighbor
                    if nx < 0 or nx >= len(matrix) or ny < 0 or ny >= len(matrix[0]) or matrix[nx][ny] == 0:
                        found_zero_neighbor = True
                        break
                # If we found a zero neighbor, set the current cell to 0
                if found_zero_neighbor:
                    new_matrix[i][j] = 0

    return new_matrix



if __name__ == '__main__':
    freeze_support()
    main(already_trained=True)
    input("idan magram")



