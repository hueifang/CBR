#!/usr/bin/env python
"""
This code extracts the features from a network for the training samples
and then computes the target codes of relevant samples.

The obtained target codes are written back to the network as the weights.

"""
#import os
import sys
# Make sure that caffe is on the python path:
caffe_root = '/path/to/caffe/'
sys.path.insert(0, caffe_root + 'python')

import argparse
import time
#import leveldb
import math
import caffe
import numpy as np
import cv2



def prepare_im(image, crop_size):
    '''
    Get a center crop of a 256x256 image.
    '''

    image = cv2.resize(image, (256, 256))

    # Take center crop
    center = np.array(image.shape[:2]) / 2.0
    crop = np.tile(center, (1, 2))[0] + np.concatenate([
	       -np.array([crop_size, crop_size]) / 2.0,
	       np.array([crop_size, crop_size]) / 2.0])
    crop = crop.astype(int)
    input_ = image[crop[0]:crop[2], crop[1]:crop[3], :]

    return input_


def main(argv):
    '''
    Compute target codes and write the target codes into the network as weights
    '''
    #pycaffe_dir = os.path.dirname(__file__)
    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
    parser.add_argument(
        "IN_weights",
        help="Input network weight model."
    )
    parser.add_argument(
        "OUT_weights",
        help="Output network weight model."
    )
    parser.add_argument(
        "DEPLOY",
        help="Deploy prototxt."
    )
    parser.add_argument(
        "INPUT_data",
        help="Input data image list."
    )
    parser.add_argument(
        "INPUT_label",
        help="Input data label list."
    )
    args = parser.parse_args()

    caffe.set_mode_gpu()
    caffe.set_device(0)

    # load network model
    net = caffe.Net(args.DEPLOY, args.IN_weights, caffe.TEST)

    # add preprocessing
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    mean = np.array([104, 117, 123])
    transformer.set_mean('data', mean) #### subtract mean ####
    transformer.set_transpose('data', (2, 0, 1)) # height*width*channel -> channel*height*width
    transformer.set_raw_scale('data', 255) # pixel value range
    transformer.set_channel_swap('data', (2, 1, 0)) # RGB -> BGR

    # get input data blob
    data_blob_shape = net.blobs['data'].data.shape
    data_blob_shape = list(data_blob_shape)
    batchsize = data_blob_shape[0]

    # get crop size
    crop_size = data_blob_shape[2]

    print 'Reading image list ...'

    with open(args.INPUT_data) as f:
        image_files = [im_f.rstrip('\n')
                       for im_f in f.readlines()]

    with open(args.INPUT_label) as f:
        labels = [int(im_f.rstrip('\n'))
                  for im_f in f.readlines()]

    feat_blob_shape = net.blobs['abstract_layer'].data.shape
    feat_blob_shape = list(feat_blob_shape)

    num_labels = max(labels) + 1
    avg_feat = [0]*(num_labels)
    label_counter = [0]*num_labels

    num_images = len(image_files)
    num_batches = int(math.ceil(float(num_images)/batchsize))

    start = time.time()
    for i in range(0, num_batches):
        start_idx = batchsize*i
        end_idx = min(batchsize*(i+1), num_images)
        net.blobs['data'].reshape(end_idx - start_idx, data_blob_shape[1],
                                  data_blob_shape[2], data_blob_shape[3])
        net.blobs['data'].data[...] = map(lambda x: transformer.preprocess('data', prepare_im(caffe.io.load_image(x), crop_size)), image_files[start_idx:end_idx])
        net.forward()
        feat = net.blobs['abstract_layer'].data

        for j in range(0, feat.shape[0]):
            ll = labels[start_idx + j]
            avg_feat[ll] += feat[j]
            label_counter[ll] += 1

        print "processed: ", (i+1), "/", num_batches, "batches"

    for i in range(0, num_labels):
        avg_feat[i] /= label_counter[i]

    print "Feature extraction done in %.2f s." % (time.time() - start)
#===============================================================================================

    # update model
    print "Updating model"

    params = ['retrieval']
    fc_params = {pr: (net.params[pr][0].data) for pr in params}

    for i in range(0, num_labels):
        fc_params['retrieval'][i] = avg_feat[i]

    net.save(args.OUT_weights)

if __name__ == '__main__':
    main(sys.argv)
