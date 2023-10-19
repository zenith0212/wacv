import sys
# sys.path.append('../../Keras_VGGFace2_ResNet50/src')
import PIL
import numpy as np
import Keras_VGGFace2_ResNet50.src.config as cg
import cv2

def load_data(path='', shape=None, mode='eval'):

    short_size = 224.0
    crop_size = shape
    img = PIL.Image.open(path)
    im_shape = np.array(img.size)    # in the format of (width, height, *)
    img = img.convert('RGB')

    ratio = float(short_size) / np.min(im_shape)
    img = img.resize(size=(int(np.ceil(im_shape[0] * ratio)),   # width
                           int(np.ceil(im_shape[1] * ratio))),  # height
                     resample=PIL.Image.BILINEAR)

    x = np.array(img)  # image has been transposed into (height, width)
    newshape = x.shape[:2]
    if mode == 'eval':    # center crop
        h_start = (newshape[0] - crop_size[0])//2
        w_start = (newshape[1] - crop_size[1])//2
    else:
        raise IOError('==> unknown mode.')
    x = x[h_start:h_start+crop_size[0], w_start:w_start+crop_size[1]]
    x = x[:, :, ::-1] - cg.mean
    return x

def crop_image(img, shape):
    short_size = 224.0
    crop_size = shape
    im_shape = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ratio = float(short_size) / np.min(im_shape)
    resizeX = int(np.ceil(im_shape[1] * ratio))
    resizeY = int(np.ceil(im_shape[0] * ratio))
    img = cv2.resize(img, (resizeX, resizeY), interpolation=cv2.INTER_LINEAR)
    im_shape = img.shape[:2]
    y_start = (im_shape[0] - crop_size[0])//2
    x_start = (im_shape[1] - crop_size[1])//2
    # print(img.shape, y_start, x_start)
    img = img[y_start: y_start + crop_size[0], x_start: x_start + crop_size[1]]
    img = img[:,:,::-1] - cg.mean
    return img