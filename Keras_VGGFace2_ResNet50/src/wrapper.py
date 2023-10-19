'''/**
 * @author Rahul Sharma
 * @email rahul.sharma@usc.edu
 * @create date 2023-02-04 11:38:53
 * @modify date 2023-02-04 11:38:53
 * @desc [description]
 */
'''
from __future__ import absolute_import
from __future__ import print_function
import os
import sys
import pdb
import argparse
import Keras_VGGFace2_ResNet50.src.utils as ut
import numpy as np

def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

def initialize_model(weights_path='../Keras_VGGFace2_ResNet50/weights/weights.h5'):
    import Keras_VGGFace2_ResNet50.src.model as model
    model_eval = model.Vggface2_ResNet50(mode='eval')
    weights_path = weights_path
    model_eval.load_weights(weights_path, by_name=True)
    return model_eval

def image_encoding(model, face_track_images):
    FEATURE_DIM = 512
    BATCH_SIZE = 512
    num_faces = len(face_track_images)
    face_feats = np.empty((num_faces, FEATURE_DIM))
    idxes = list(range(num_faces))
    imgchunks = list(chunks(idxes, BATCH_SIZE))

    for c, imgs in enumerate(imgchunks):
        im_array = np.array([ut.crop_image(face_track_images[idx], shape=(224, 224, 3))\
             for idx in imgs])
        f = model.predict(im_array, batch_size=BATCH_SIZE)
        start = c * BATCH_SIZE
        end = min((c + 1) * BATCH_SIZE, num_faces)
        face_feats[start:end] = f
    return face_feats
