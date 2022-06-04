import cv2
import numpy as np
import matplotlib.pyplot as plt

import skimage.color as color
import skimage.filters as filters
from skimage.measure import label
import skimage.segmentation as seg

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy

import segmentation_models as sm
from segmentation_models.utils import set_trainable

from datacube import Datacube
from dea_spatialtools import xr_rasterize


## EDA functions


## Data processing functions

def clean_name(name):
    '''
    Process sourcename string format to match image and its label.
    '''

    if name is None:
        res = None
    else:
        if name.upper()[-4::] == ".JPG":
            res = name.upper()[:-4].replace(' ','_')
        else:
            res = name.upper().replace(' ','_')
    return res


def laplace_of_gaussian(gray_img, sigma=1., kappa=0.75, pad=False):
    '''
    Applies Laplacian of Gaussians to a grayscale image.
    '''

    assert len(gray_img.shape) == 2
    img = cv2.GaussianBlur(gray_img, (0, 0), sigma) if 0. < sigma else gray_img
    img = cv2.Laplacian(img, cv2.CV_64F)
    rows, cols = img.shape[:2]

    # min/max of 3x3-neighbourhoods
    min_map = np.minimum.reduce(list(img[r:rows-2+r, c:cols-2+c]
                                     for r in range(3) for c in range(3)))
    max_map = np.maximum.reduce(list(img[r:rows-2+r, c:cols-2+c]
                                     for r in range(3) for c in range(3)))

    # bool matrix for image value positiv (w/out border pixels)
    pos_img = 0 < img[1:rows-1, 1:cols-1]

    # bool matrix for min < 0 and 0 < image pixel
    neg_min = min_map < 0
    neg_min[1 - pos_img] = 0

    # bool matrix for 0 < max and image pixel < 0
    pos_max = 0 < max_map
    pos_max[pos_img] = 0

    # sign change at pixel?
    zero_cross = neg_min + pos_max

    # values: max - min, scaled to 0--255; set to 0 for no sign change
    value_scale = 255. / max(1., img.max() - img.min())
    values = value_scale * (max_map - min_map)
    values[1 - zero_cross] = 0.

    # optional thresholding
    if 0. <= kappa:
        thresh = float(np.absolute(img).mean()) * kappa
        values[values < thresh] = 0.

    log_img = values.astype(np.uint8)

    if pad:
        log_img = np.pad(log_img, pad_width=1, mode='constant', constant_values=0)

    return log_img


## Visualization functions


def show_filters(image, true_mask, abs_threshold=150, perc_threshold=0.4):
    '''
    Visualizes various segmentation methods applied on a given image.
    '''

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(16, 9), sharex=True, sharey=True)

    # Original image
    ax[0][0].set_title('Original Image')
    ax[0][0].imshow(image)

    # Ground-truth mask
    ax[0][1].set_title('Ground truth mask')
    ax[0][1].imshow(true_mask)

    # Manual threshold
    ax[0][2].set_title('Manual thresholding')
    ax[0][2].imshow(image >= abs_threshold)

    # Adaptive threshold
    local_thr = filters.threshold_local(image, block_size=51, offset=10)
    ax[1][0].set_title('Adaptive thresholding')
    ax[1][0].imshow(image < local_thr)

    # Yen filter
    yen_thr = filters.threshold_yen(image)
    ax[1][1].set_title("Yen's method")
    ax[1][1].imshow(image > yen_thr)

    # Edge-detection and watershed method
    edges = filters.sobel(image)

    # Identify some background and foreground pixels from the intensity values.
    # These pixels are used as seeds for watershed.
    markers = np.zeros_like(image)
    foreground, background = 1, 2
    markers[image < 30.0] = background
    markers[image > 150.0] = foreground

    ws = seg.watershed(edges, markers)
    seg1 = label(ws == foreground)
    ax[1][2].set_title('Edge detection and watershed method')
    ax[1][2].imshow(seg1)

    # Slic filter
    image_slic = seg.slic(image, n_segments=8000, start_label=0)

    # label2rgb replaces each discrete label with the average interior color
    slic_mask = color.label2rgb(image_slic, image, kind='avg', bg_label=-1)
    ax[2][0].set_title('Simple Linear Iterative Clustering')
    ax[2][0].imshow(slic_mask/255 >= perc_threshold)

    # Quickshift method
    image_quick = seg.quickshift(np.repeat(image.reshape(image.shape + (1,)), 3, -1))

    # label2rgb
    quick_mask = color.label2rgb(image_quick, image, kind='avg', bg_label=-1)

    ax[2][1].set_title('Quickshift method')
    ax[2][1].imshow(quick_mask/255 >= perc_threshold)

    # Laplace of Gaussian
    ax[2][2].set_title('Laplacian of Gaussian')
    ax[2][2].imshow(laplace_of_gaussian(image))

    plt.tight_layout()


def show_prediction(model, image, mask, threshold=0.97, img_size=(768, 768), post_processing=True,
                    kernelOpen=np.ones((3, 3)), kernelClose=np.ones((64, 64))):
    '''
    Visualizes the model's prediction on a given image and its ground truth mask.
    '''

    # Check that the images are two-dimensional
    assert len(image.shape) == 2
    assert len(mask.shape) == 2

    # Resize and filter images
    image = cv2.resize(image, img_size, interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(mask, img_size, interpolation=cv2.INTER_NEAREST)
    image = laplace_of_gaussian(image, pad=True)

    # Transform image into the suitable format for prediction
    image = np.expand_dims(np.repeat(image.reshape(image.shape + (1,)), 3, -1), axis=0)

    # Confirm image's shape
    assert len(image.shape) == 4

    # Plot predicted and true mask
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))

    pred = model.predict(image)[0, :, :, 0]
    pred_mask = cv2.inRange(pred, threshold, 1)

    # Apply post-processing methods
    if post_processing == True:

        # denoise the pixels
        pred_mask=cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernelOpen)

        # join the pixels together
        pred_mask=cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernelClose)

    ax[0].imshow(pred_mask)
    ax[0].set_title('Predicted mask')

    ax[1].imshow(mask)
    ax[1].set_title('True mask')



## Training functions


epsilon = 1e-5
smooth = 1

## Model functions

def conv2d_block(input_tensor, n_filters, kernel_size = 3, batchnorm = True):
    '''
    Defines 2D convolutional block to be used for the U-net Light construction.
    '''

    # first layer
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    # second layer
    x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    return x


def get_unet_small(nClasses, input_height=128, input_width=128, n_filters = 16, dropout = 0.1, batchnorm = True, n_channels=3):
    '''
    Defines a small U-net model (U-net Light) to be trained from scratch.
    '''

    input_img = tf.keras.layers.Input(shape=(input_height,input_width, n_channels))

    # Contracting Path
    c1 = conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)
    p1 = tf.keras.layers.Dropout(dropout)(p1)

    c2 = conv2d_block(p1, n_filters = n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

    # Expansive Path
    u3 = tf.keras.layers.Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c2)
    u3 = tf.keras.layers.concatenate([u3, c1])
    u3 = tf.keras.layers.Dropout(dropout)(u3)
    c3 = conv2d_block(u3, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c3)
    model = tf.keras.models.Model(inputs=[input_img], outputs=[outputs])

    return model


def print_model_menu():
    '''
    Prints selection menu to define model and settings for training.
    '''

    # Print main menu

    print("\nChoose a target image size (integer) for resizing and training (Default is 768).")

    size = int(input())
    img_size = (size, size)

    print("\nSelect a model architecture from the list or define a new model from scratch.")
    print("\n\t1) U-net (VGG16)")
    print("\t2) LinkNet (VGG16)")
    print("\t3) FPN (VGG16)")
    print("\t4) PSPNet (VGG16)")
    print("\t5) Small U-net model (U-net Light)")

    selection = int(input())

    # Free up RAM in case the model definition cells were run multiple times
    tf.keras.backend.clear_session()

    if selection == 1:
        model = sm.Unet(backbone_name='vgg16', classes=1, activation='sigmoid',
                        encoder_weights=None, input_shape=img_size + (3,))
    elif selection == 2:
        model = sm.Linknet(backbone_name='vgg16', classes=1, activation='sigmoid',
                           encoder_weights=None, input_shape=img_size + (3,))
    elif selection == 3:
        model = sm.FPN(backbone_name='vgg16', classes=1, activation='sigmoid',
                       encoder_weights=None, input_shape=img_size + (3,))
    elif selection == 4:
        model = sm.PSPNet(backbone_name='vgg16', classes=1, activation='sigmoid',
                          encoder_weights=None, input_shape=img_size + (3,))
    elif selection == 5:
        model = get_unet_small(1, input_height=size, input_width=size, n_channels=3)
    else:
        raise ValueError('Invalid option. Select a number from 1 to 5.')

    print("\nChoose a specific batch size for training (Default = 2).")

    batch_size = int(input())


    return model, img_size, batch_size


def get_training_data(datacube, linescan_datasets, polygons, img_size=(768, 768), filtering=True, include_all=False, plot=True):
    '''
    Collects and transforms the linescan images into a suitable format for training.
    '''

    # Initialize lists to store results
    ids = []
    masks = []
    images = []

    # Collect all the images that directly match in both datasets
    for i in range(0, len(linescan_datasets)):
        fname = linescan_datasets[i].metadata_doc['label']

        if sum(polygons.SourceNameClean == fname):
            ids.append(i)

    # Loop over the whole dataset collecting, transforming images, and getting the corresponding masks
    for idx in ids:
        fname = linescan_datasets[idx].metadata_doc['label']
        src = datacube.load(product='linescan', id=linescan_datasets[idx].id, output_crs='epsg:28355', resolution=(-10,10))

        ob = polygons.loc[polygons.SourceNameClean == fname]
        tgt = xr_rasterize(gdf=ob, da=src)

        img = cv2.resize(src.linescan.values[0], img_size, interpolation=cv2.INTER_NEAREST)

        if filtering == True:
            img = laplace_of_gaussian(img, pad=True)

        img = (np.repeat(img.reshape(img.shape + (1,)), 3, -1))
        images.append(img)

        mask = cv2.resize(tgt.values, img_size, interpolation=cv2.INTER_NEAREST)
        mask = (np.repeat(mask.reshape(mask.shape + (1,)), 3, -1))
        masks.append(mask)

    # Plot an example of an image and its mask
    if plot == True:

        image1 = images[0][:, :, 0]
        mask1 = masks[0][:, :, 0]

        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
        ax[0].imshow(image1)
        ax[0].set_title('Source: Linescan Image')

        ax[1].imshow(mask1)
        ax[1].set_title('Target: Ground Truth Mask')

    # Load collected data into arrays
    if include_all == True:

        x_train = np.array(images)
        y_train = np.array(masks)

    else:

        x_train = np.array(images[0:11] + images[13:15] + images[16:19] + images[24:27] + images[28:30] + images[31:33] + images[34])
        y_train = np.array(masks[0:11] + masks[13:15] + masks[16:19] + masks[24:27] + masks[28:30] + masks[31:33] + masks[34])

    return x_train, y_train



## Loss functions

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dsc(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true, y_pred)



## Evaluation functions


def make_submission(datacube, test_dataset, test_labels, model, img_size=(768, 768),
                    post_processing=True, kernelOpen=np.ones((3, 3)), kernelClose=np.ones((64, 64))):
    '''
    Obtains the model's predictions on the testing set and creates the file for submission.
    '''

    for index, file_stem in enumerate(test_labels):

        # load the linescan data
        src = datacube.load(product='linescan', label=file_stem, output_crs='epsg:28355', resolution=(-10, 10))

        # Make copies
        result = src.linescan.copy()
        original_shape = result.shape

        # Transform image and get predictions
        img = src.linescan.values
        img = cv2.resize(img[0], img_size, interpolation=cv2.INTER_NEAREST)
        img = laplace_of_gaussian(img, pad=True)
        img = np.expand_dims(img, axis=0)
        pred = model.predict(np.repeat(img.reshape(img.shape + (1,)), 3, -1))[0, :, :, 0]

        # Apply post-processing methods
        if post_processing == True:

            pred_mask = cv2.inRange(pred, 0.97, 1)
            maskOpen = cv2.morphologyEx(pred_mask, cv2.MORPH_OPEN, kernelOpen)
            pred = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)

        pred2 = cv2.resize(pred, (original_shape[2], original_shape[1]), interpolation=cv2.INTER_NEAREST)
        pred2 = np.expand_dims(pred2, axis=0)
        result.data = pred2

        # iterate over the coordinates that are required for testing in the current linescan file
        for idx, ob in test_dataset.loc[test_dataset.label==file_stem].iterrows():
            result_tf = result.sel(x=ob.x, y=ob.y, method='nearest').values[0]
            result_10 = int(result_tf >= 0.5)
            test_dataset.loc[(test_dataset.label==file_stem) & (test_dataset.x==ob.x) & (test_dataset.y==ob.y), 'target'] = result_10

    # Clean and denoise predictions
    df = test_dataset.reset_index()[['id', 'target']].copy()

    df['P'] = df['target'].shift(1).fillna(1)
    df['N'] = df['target'].shift(-1).fillna(0)
    df['New'] = np.nan

    for index, row in df.iterrows():
        if (row['P'] < row['target']) and (row['target'] > row['N']):
            df.loc[index, 'New'] = 0.0
        elif (row['P'] > row['target']) and (row['target'] < row['N']):
            df.loc[index, 'New'] = 1.0
        else:
            df.loc[index, 'New'] = df.loc[index, 'target']

    final = df[['id', 'New']].copy()
    final['target'] = final['New'].astype(int)
    final = final.drop('New', axis=1)

    final.to_csv('final_submission.csv', index=None)

    return final
