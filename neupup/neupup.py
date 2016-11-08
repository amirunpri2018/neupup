#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import sys
import re
import urllib
import urlparse, os
import shutil
import os
from faceswap import doalign
import faceswap.core
import random
from subprocess import call
import glob

from plat.utils import anchors_from_image, offset_from_string, get_json_vectors
from plat.grid_layout import create_mine_grid
from plat import zoo

# discgen related imports
import numpy as np
from PIL import Image
from scipy.misc import imread, imsave, imresize
import theano
import hashlib
import time

# returns True if file not found and can be processed
def check_recent(infile, recentfile):
    try:
        with open(recentfile) as f :
            content = f.readlines()
    except EnvironmentError: # parent of IOError, OSError
        # that's ok
        print("No cache of recent files not found ({}), will create".format(recentfile))
        return True

    md5hash = hashlib.md5(open(infile, 'rb').read()).hexdigest().encode('utf-8')
    known_hashes = [line.split('\t', 1)[0] for line in content]
    if md5hash in known_hashes:
        return False
    else:
        return True

def add_to_recent(infile, comment, recentfile, limit=500):
    if os.path.isfile(recentfile):
        shutil.copyfile(recentfile, "{}.bak".format(recentfile))

    try:
        with open(recentfile) as f :
            content = f.readlines()
    except EnvironmentError: # parent of IOError, OSError
        content = []

    md5hash = hashlib.md5(open(infile, 'rb').read()).hexdigest()
    newitem = u"{}\t{}\n".format(md5hash, comment)
    content.insert(0, newitem)
    content = content[:limit]

    with open(recentfile, "w") as f:
        f.writelines(content)

max_allowable_extent = 180
min_allowable_extent = 1
# reized input file
resized_input_file = "temp_files/resized_input_file.png"
# the input image file is algined and saved
aligned_file = "temp_files/aligned_file.png"
# the reconstruction is also saved
recon_file = "temp_files/recon_file.png"
# the reconstruction is also saved
transformed_file = "temp_files/transformed.png"
# reconstruction is swapped into original
swapped_file = "temp_files/swapped_file.png"
# used to save surprising failures
debug_file = "temp_files/debug.png"
# the interpolated sequence is saved into this directory
sequence_dir = "temp_files/image_sequence/"
# template for output png files
generic_sequence = "{:03d}.png"
samples_sequence_filename = sequence_dir + generic_sequence

def make_or_cleanup(local_dir):
    # make output directory if it is not there
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # and clean it out if it is there
    filelist = [ f for f in os.listdir(local_dir) ]
    for f in filelist:
        del_path = os.path.join(local_dir, f)
        if os.path.isdir(del_path):
            shutil.rmtree(del_path)
        else:
            os.remove(del_path)

max_extent = 720
def resize_to_a_good_size(infile, outfile):
    image_array = imread(infile, mode='RGB')

    im_shape = image_array.shape
    h, w, _ = im_shape

    # maximum twitter aspect ratio is 239:100
    max_width = int(h * 230 / 100)
    if w > max_width:
        offset_x = (w - max_width)/2
        print("cropping from {0},{1} to {2},{1}".format(w,h,max_width))
        image_array = image_array[:,offset_x:offset_x+max_width,:]
        w = max_width
    # minimum twitter aspect ratio is maybe 1:2
    max_height = int(w * 2)
    if h > max_height:
        offset_y = (h - max_height)/2
        print("cropping from {0},{1} to {0},{2}".format(w,h,max_height))
        image_array = image_array[offset_y:offset_y+max_height,:,:]
        h = max_height

    scale_down = None
    if w >= h:
        if w > max_extent:
            scale_down = float(max_extent) / w
    else:
        if h > max_extent:
            scale_down = float(max_extent) / h

    if scale_down is not None:
        new_w = int(scale_down * w)
        new_h = int(scale_down * h)
    else:
        new_w = w
        new_h = h

    new_w = new_w - (new_w % 4)
    new_h = new_h - (new_h % 4)

    if new_w >= 1.5 * new_h:
        wide_image = True
    else:
        wide_image = False

    print("resizing from {},{} to {},{}".format(w, h, new_w, new_h))
    image_array_resized = imresize(image_array, (new_h, new_w))
    imsave(outfile, image_array_resized)
    return True, wide_image

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def do_convert(raw_infile, outfile, dmodel, do_smile, smile_offsets, image_size, min_span, initial_steps=1, recon_steps=1, offset_steps=2, check_extent=True):
    failure_return_status = False, False, False

    infile = resized_input_file;

    did_resize, wide_image = resize_to_a_good_size(raw_infile, infile)
    if not did_resize:
        return failure_return_status

    # first align input face to canonical alignment and save result
    try:
        if not doalign.align_face(infile, aligned_file, image_size, min_span=min_span, max_extension_amount=0):
            return failure_return_status
    except Exception as e:
        # get_landmarks strangely fails sometimes (see bad_shriek test image)
        print("faceswap: doalign failure {}".format(e))
        return failure_return_status

    # go ahead and cache the main (body) image and landmarks, and fail if face is too big
    try:
        body_image_array = imread(infile, mode='RGB')
        print(body_image_array.shape)
        body_rect, body_landmarks = faceswap.core.get_landmarks(body_image_array)
        max_extent = faceswap.core.get_max_extent(body_landmarks)
    except faceswap.core.NoFaces:
        print("faceswap: no faces in {}".format(infile))
        return failure_return_status
    except faceswap.core.TooManyFaces:
        print("faceswap: too many faces in {}".format(infile))
        return failure_return_status
    if check_extent and max_extent > max_allowable_extent:
        print("face to large: {}".format(max_extent))
        return failure_return_status
    elif check_extent and max_extent < min_allowable_extent:
        print("face to small: {}".format(max_extent))
        return failure_return_status
    else:
        print("face not too large: {}".format(max_extent))

    # read in aligned file to image array
    _, _, anchor_images = anchors_from_image(aligned_file, image_size=(image_size, image_size))

    # encode aligned image array as vector, apply offset
    encoded = dmodel.encode_images(anchor_images)[0]

    if smile_offsets is not None:
        smile_vector = smile_offsets[0]
        smile_score = np.dot(smile_vector, encoded)
        smile_detected = (smile_score > 0)
        print("Smile vector detector:", smile_score, smile_detected)
        if do_smile is None:
            has_smile = smile_detected
        else:
            has_smile = not str2bool(do_smile)

        if has_smile:
            print("Smile detected, removing")
            chosen_anchor = [encoded, encoded - smile_vector]
        else:
            print("Smile not detected, providing")
            chosen_anchor = [encoded, encoded + smile_vector]
    else:
        has_smile = False
        chosen_anchor = [encoded, encoded]

    z_dim = dmodel.get_zdim()

    # TODO: fix variable renaming
    anchors, samples_sequence_dir, movie_file = chosen_anchor, sequence_dir, outfile

    # these are the output png files
    samples_sequence_filename = samples_sequence_dir + generic_sequence

    # prepare output directory
    make_or_cleanup(samples_sequence_dir)

    # generate latents from anchors
    z_latents = create_mine_grid(rows=1, cols=offset_steps, dim=z_dim, space=offset_steps-1, anchors=anchors, spherical=True, gaussian=False)
    samples_array = dmodel.sample_at(z_latents)
    print("Samples array: ", samples_array.shape)

    # save original file as-is
    for i in range(initial_steps):
        filename = samples_sequence_filename.format(1 + i)
        imsave(filename, body_image_array)
        print("original file: {}".format(filename))

    # build face swapped reconstruction
    sample = samples_array[0]
    try:
        # face_image_array = (255 * np.dstack(sample)).astype(np.uint8)
        face_image_array = (255 * np.dstack(sample)).astype(np.uint8)
        imsave(recon_file, face_image_array)
        # face_landmarks = faceswap.core.get_landmarks(face_image_array)
        # faceswap.core.do_faceswap_from_face(infile, face_image_array, face_landmarks, swapped_file)
        faceswap.core.do_faceswap(infile, recon_file, swapped_file)
        print("swapped file: {}".format(swapped_file))
        recon_array = imread(swapped_file, mode='RGB')
    except faceswap.core.NoFaces:
        print("faceswap: no faces when generating swapped file {}".format(infile))
        imsave(debug_file, face_image_array)
        return failure_return_status
    except faceswap.core.TooManyFaces:
        print("faceswap: too many faces in {}".format(infile))
        return failure_return_status

    # now save interpolations to recon
    for i in range(1,recon_steps):
        frac_orig = ((recon_steps - i) / (1.0 * recon_steps))
        frac_recon = (i / (1.0 * recon_steps))
        interpolated_im = frac_orig * body_image_array + frac_recon * recon_array
        filename = samples_sequence_filename.format(i+initial_steps)
        imsave(filename, interpolated_im)
        print("interpolated file: {}".format(filename))

    final_face_index = len(samples_array) - 1
    for i, sample in enumerate(samples_array):
        try:
            cur_index = i + initial_steps + recon_steps
            stack = np.dstack(sample)
            face_image_array = (255 * np.dstack(sample)).astype(np.uint8)
            # if i == final_face_index:
            #     imsave(transformed_file, face_image_array)
            face_rect, face_landmarks = faceswap.core.get_landmarks(face_image_array)
            filename = samples_sequence_filename.format(cur_index)
            imsave(transformed_file, face_image_array)
            # faceswap.core.do_faceswap_from_face(infile, face_image_array, face_landmarks, filename)
            faceswap.core.do_faceswap(infile, transformed_file, filename)
            print("generated file: {}".format(filename))
        except faceswap.core.NoFaces:
            print("faceswap: no faces in {}".format(infile))
            return failure_return_status
        except faceswap.core.TooManyFaces:
            print("faceswap: too many faces in {}".format(infile))
            return failure_return_status

    last_sequence_index = initial_steps + recon_steps + offset_steps - 1
    last_filename = samples_sequence_filename.format(last_sequence_index)
    shutil.copyfile(last_filename, outfile)

    return True, has_smile, wide_image

def check_lazy_initialize(args, dmodel, smile_offsets):
    # debug: don't load anything...
    # return dmodel, smile_offsets

    # first get model ready
    if dmodel is None and (args.model is not None or args.model_file is not None):
        print('Finding saved model...')
        dmodel = zoo.load_model(args.model, args.model_file, args.model_type)

    # get attributes
    if smile_offsets is None and args.anchor_offset is not None:
        offsets = get_json_vectors(args.anchor_offset)
        dim = len(offsets[0])
        offset_indexes = args.anchor_indexes.split(",")
        offset_vector = offset_from_string(offset_indexes[0], offsets, dim)
        for n in range(1, len(offset_indexes)):
            offset_vector += offset_from_string(offset_indexes[n], offsets, dim)
        smile_offsets = [offset_vector]

    return dmodel, smile_offsets

def main(args=None):
    # argparse
    parser = argparse.ArgumentParser(description='Perform neural puppet transformations on images')
    parser.add_argument('--do-smile', default=None,
                        help='Force smile on/off (skip classifier) [1/0]')
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="name of model in plat zoo")
    parser.add_argument("--model-file", dest='model_file', type=str, default=None,
                        help="path to the saved model")
    parser.add_argument("--model-type", dest='model_type', type=str, default=None,
                        help="the type of model (usually inferred from filename)")
    parser.add_argument('--anchor-offset', dest='anchor_offset', default=None,
                        help="use json file as source of each anchors offsets")
    parser.add_argument('--anchor-indexes', dest='anchor_indexes', default="31,21",
                        help="smile_index,open_mouth_index")
    parser.add_argument("--image-size", dest='image_size', type=int, default=64,
                        help="size of (offset) images")
    parser.add_argument("--min-span", dest='min_span', type=int, default=None,
                        help="minimum width for detected face (default: image-size/4")

    parser.add_argument('--sort-inputs', dest='sort_inputs',
                        help='Sort inputs before processing', default=False, action='store_true')
    parser.add_argument("--input-directory", dest='input_directory', default="inputs",
                        help="directory for input files")
    parser.add_argument("--output-directory", dest='output_directory', default="outputs",
                        help="directory for output files")
    parser.add_argument("--offset", type=int, default=0,
                        help="data offset to skip")
    parser.add_argument("--input-file", dest='input_file', default=None,
                        help="single file input (overrides input-directory)")
    parser.add_argument("--output-file", dest='output_file', default="output.png",
                        help="single file output")

    args = parser.parse_args()

    # initialize and then lazily load
    dmodel = None
    smile_offsets = None

    make_or_cleanup("temp_files")

    if args.min_span is not None:
        min_span = args.min_span
    else:
        min_span = args.image_size / 4

    dmodel, smile_offsets = check_lazy_initialize(args, dmodel, smile_offsets)

    if args.input_file is not None:
        result, had_smile, is_wide = do_convert(args.input_file, args.output_file, dmodel, args.do_smile, smile_offsets, args.image_size, min_span, check_extent=False)
        print("result: {}, had_smile: {}".format(result, had_smile))
        exit(0)

    # read input files
    files = glob.glob("{}/*.*".format(args.input_directory))
    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    if args.sort_inputs:
        files = sorted(files)

    if args.offset > 0:
        print("Trimming from {} to {}".format(len(files), args.offset))
        files = files[args.offset:]

    for infile in files:
        outfile = os.path.join(args.output_directory, os.path.basename(infile))
        # always save as png
        outfile = "{}.png".format(os.path.splitext(outfile)[0])
        result, had_smile, is_wide = do_convert(infile, outfile, dmodel, args.do_smile, smile_offsets, args.image_size, min_span, check_extent=False)

if __name__ == "__main__":
    main()
