#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import sys
import re
import urllib
import urlparse, os
from shutil import copyfile
import os
import doalign
import random
from subprocess import call

from plat.utils import anchors_from_image, offset_from_string, get_json_vectors
from plat.grid_layout import create_mine_grid

# discgen related imports
from discgen.interface import DiscGenModel
import faceswap
import numpy as np
from PIL import Image
from scipy.misc import imread, imsave, imresize
import theano
import hashlib
import time

# tweet_suffix = u""
# tweet_suffix = u" #test_hashtag"
# tweet_suffix = u" #nuclai16"
tweet_suffix = u" #NeuralPuppet"

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
        copyfile(recentfile, "{}.bak".format(recentfile))

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
min_allowable_extent = 60
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
# this is the final swapped image
final_image = "temp_files/final_image.png"
# the interpolated sequence is saved into this directory
sequence_dir = "temp_files/image_sequence/"
# template for output png files
generic_sequence = "{:03d}.png"
samples_sequence_filename = sequence_dir + generic_sequence
# template for ffmpeg arguments
ffmpeg_sequence = "%3d.png"
ffmpeg_sequence_filename = sequence_dir + ffmpeg_sequence

def make_or_cleanup(local_dir):
    # make output directory if it is not there
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # and clean it out if it is there
    filelist = [ f for f in os.listdir(local_dir) ]
    for f in filelist:
        os.remove(os.path.join(local_dir, f))

archive_text = "metadata.txt"
archive_resized = "input_resized.png"
archive_aligned = "aligned.png"
archive_recon = "reconstruction.png"
archive_transformed = "transformed.png"
archive_swapped = "swapped.png"
archive_final_image = "final_image.png"
archive_final_movie = "final_movie.mp4"

def archive_post(subdir, posted_id, original_text, post_text, respond_text, downloaded_basename, downloaded_input, final_movie, archive_dir="archives"):
    # setup paths
    archive_dir = "{}/{}".format(archive_dir, subdir)
    archive_input_path = "{}/{}".format(archive_dir, downloaded_basename)
    archive_text_path = "{}/{}".format(archive_dir, archive_text)
    archive_resized_path = "{}/{}".format(archive_dir, archive_resized)
    archive_aligned_path = "{}/{}".format(archive_dir, archive_aligned)
    archive_recon_path = "{}/{}".format(archive_dir, archive_recon)
    archive_transformed_path = "{}/{}".format(archive_dir, archive_transformed)
    archive_swapped_path = "{}/{}".format(archive_dir, archive_swapped)
    archive_final_image_path = "{}/{}".format(archive_dir, archive_final_image)
    archive_final_movie_path = "{}/{}".format(archive_dir, archive_final_movie)

    # prepare output directory
    make_or_cleanup(archive_dir)

    # save metadata
    with open(archive_text_path, 'a') as f:
        f.write(u"posted_id\t{}\n".format(posted_id))
        f.write(u"original_text\t{}\n".format(original_text))
        # these might be unicode. what a PITA
        f.write(u'\t'.join([u"post_text", post_text]).encode('utf-8').strip())
        f.write(u"\n")
        f.write(u'\t'.join([u"respond_text", respond_text]).encode('utf-8').strip())
        f.write(u"\n")
        f.write(u"subdir\t{}\n".format(subdir))

    # save input, a few working files, outputs
    copyfile(downloaded_input, archive_input_path)
    copyfile(resized_input_file, archive_resized_path)
    copyfile(aligned_file, archive_aligned_path)
    copyfile(recon_file, archive_recon_path)
    copyfile(transformed_file, archive_transformed_path)
    copyfile(swapped_file, archive_swapped_path)
    copyfile(final_image, archive_final_image_path)
    copyfile(final_movie, archive_final_movie_path)

max_extent = 720
def resize_to_a_good_size(infile, outfile):
    image_array = imread(infile, mode='RGB')

    # this is believed to no longer be necessary because imread is coercing to rgb
    # im_shape = image_array.shape
    # if len(im_shape) == 2:
    #     h, w = im_shape
    #     print("converting from 1 channel to 3")
    #     image_array = np.array([image_array, image_array, image_array])
    # else:
    #     h, w, _ = im_shape

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

def do_convert(raw_infile, outfile, dmodel, classifier, do_smile, smile_offsets, image_size, initial_steps=10, recon_steps=10, offset_steps=20, end_bumper_steps=10, check_extent=True, wraparound=True):
    failure_return_status = False, False, False

    infile = resized_input_file;

    did_resize, wide_image = resize_to_a_good_size(raw_infile, infile)
    if not did_resize:
        return failure_return_status

    # first align input face to canonical alignment and save result
    try:
        if not doalign.align_face(infile, aligned_file, image_size, max_extension_amount=0):
            return failure_return_status
    except:
        # get_landmarks strangely fails sometimes (see bad_shriek test image)
        return failure_return_status

    # go ahead and cache the main (body) image and landmarks, and fail if face is too big
    try:
        body_image_array = imread(infile, mode='RGB')
        print(body_image_array.shape)
        body_landmarks = faceswap.get_landmarks(body_image_array)
        max_extent = faceswap.get_max_extent(body_landmarks)
    except faceswap.NoFaces:
        print("faceswap: no faces in {}".format(infile))
        return failure_return_status
    except faceswap.TooManyFaces:
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

    # classifiy aligned as smiling or not
    if do_smile is not None:
        has_smile = not str2bool(do_smile)
    else:
        has_smile = random.choice([True, False])

    # encode aligned image array as vector, apply offset
    anchor = dmodel.encode_images(anchor_images)
    if has_smile:
        print("Smile detected, removing")
        chosen_anchor = [anchor[0], anchor[0] + smile_offsets[1]]
    else:
        print("Smile not detected, providing")
        chosen_anchor = [anchor[0], anchor[0] + smile_offsets[0]]

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
        # face_landmarks = faceswap.get_landmarks(face_image_array)
        # faceswap.do_faceswap_from_face(infile, face_image_array, face_landmarks, swapped_file)
        faceswap.do_faceswap(infile, recon_file, swapped_file)
        print("swapped file: {}".format(swapped_file))
        recon_array = imread(swapped_file, mode='RGB')
    except faceswap.NoFaces:
        print("faceswap: no faces when generating swapped file {}".format(infile))
        imsave(debug_file, face_image_array)
        return failure_return_status
    except faceswap.TooManyFaces:
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
            face_landmarks = faceswap.get_landmarks(face_image_array)
            filename = samples_sequence_filename.format(cur_index)
            imsave(transformed_file, face_image_array)
            # faceswap.do_faceswap_from_face(infile, face_image_array, face_landmarks, filename)
            faceswap.do_faceswap(infile, transformed_file, filename)
            print("generated file: {}".format(filename))
        except faceswap.NoFaces:
            print("faceswap: no faces in {}".format(infile))
            return failure_return_status
        except faceswap.TooManyFaces:
            print("faceswap: too many faces in {}".format(infile))
            return failure_return_status

    last_sequence_index = initial_steps + recon_steps + offset_steps - 1
    last_filename = samples_sequence_filename.format(last_sequence_index)
    if wraparound:
        # copy last image back around to first
        first_filename = samples_sequence_filename.format(0)
        print("wraparound file: {} -> {}".format(last_filename, first_filename))
        copyfile(last_filename, first_filename)

    copyfile(last_filename, final_image)

    # also add a final out bumper
    for i in range(last_sequence_index, last_sequence_index + end_bumper_steps):
        filename = samples_sequence_filename.format(i + 1)
        copyfile(last_filename, filename)
        print("end bumper file: {}".format(filename))

    if os.path.exists(movie_file):
        os.remove(movie_file)
    command = "/usr/local/bin/ffmpeg -r 20 -f image2 -i \"{}\" -c:v libx264 -crf 20 -pix_fmt yuv420p -tune fastdecode -y -tune zerolatency -profile:v baseline {}".format(ffmpeg_sequence_filename, movie_file)
    print("ffmpeg command: {}".format(command))
    result = os.system(command)
    if result != 0:
        return failure_return_status
    if not os.path.isfile(movie_file):
        return failure_return_status

    return True, has_smile, wide_image

def check_lazy_initialize(args, dmodel, classifier, smile_offsets):
    # debug: don't load anything...
    # return model, classifier, smile_offsets

    # first get model ready
    if dmodel is None and args.model is not None:
        print('Loading saved model...')
        dmodel = DiscGenModel(filename=args.model)

    # get attributes
    if smile_offsets is None and args.anchor_offset is not None:
        offsets = get_json_vectors(args.anchor_offset)
        dim = len(offsets[0])
        offset_indexes = args.anchor_indexes.split(",")
        smile_offset_smile = offset_from_string(offset_indexes[0], offsets, dim)
        smile_offset_open = offset_from_string(offset_indexes[1], offsets, dim)
        smile_offset_blur = offset_from_string(offset_indexes[2], offsets, dim)
        pos_smile_offset = 1 * smile_offset_open + 1.25 * smile_offset_smile + 1.0 * smile_offset_blur
        neg_smile_offset = -1 * smile_offset_open - 1.25 * smile_offset_smile + 1.0 * smile_offset_blur
        smile_offsets = [pos_smile_offset, neg_smile_offset]

    return dmodel, classifier, smile_offsets

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='Follow account and repost munged images')
    parser.add_argument('-a','--accounts', help='Accounts to follow (comma separated)', default="peopleschoice,NPG")
    parser.add_argument('-d','--debug', help='Debug: do not post', default=False, action='store_true')
    parser.add_argument('-o','--open', help='Open image (when in debug mode)', default=False, action='store_true')
    parser.add_argument('-c','--creds', help='Twitter json credentials1 (smile)', default='creds.json')
    parser.add_argument('--do-smile', default=None,
                        help='Force smile on/off (skip classifier) [1/0]')
    parser.add_argument('-n','--no-update', dest='no_update',
            help='Do not update postion on timeline', default=False, action='store_true')
    parser.add_argument("--input-file", dest='input_file', default=None,
                        help="single image file input (for debugging)")
    parser.add_argument("--archive-subdir", dest='archive_subdir', default=None,
                        help="specific subdirectory for archiving results")
    parser.add_argument("--model", dest='model', type=str, default=None,
                        help="path to the saved model")
    parser.add_argument('--anchor-offset', dest='anchor_offset', default=None,
                        help="use json file as source of each anchors offsets")
    parser.add_argument('--anchor-indexes', dest='anchor_indexes', default="31,21,41",
                        help="smile_index,open_mouth_index,blur_index")
    parser.add_argument("--image-size", dest='image_size', type=int, default=64,
                        help="size of (offset) images")
    parser.add_argument('--no-wrap', dest="wraparound", default=True,
                        help='Do not wraparound last image to front', action='store_false')
    args = parser.parse_args()

    # initialize and then lazily load
    dmodel = None
    classifier = None
    smile_offsets = None

    final_movie = "temp_files/final_movie.mp4"
    final_image = "temp_files/final_image.png"

    if args.archive_subdir:
        archive_subdir = args.archive_subdir
    else:
        archive_subdir = time.strftime("%Y%m%d_%H%M%S")

    # do debug as a special case
    if args.input_file:
        dmodel, classifier, smile_offsets = check_lazy_initialize(args, dmodel, classifier, smile_offsets)
        result, had_smile, is_wide = do_convert(args.input_file, final_movie, dmodel, classifier, args.do_smile, smile_offsets, args.image_size, check_extent=False, wraparound=args.wraparound)
        print("result: {}, had_smile: {}".format(result, had_smile))
        if result and not args.no_update:
            input_basename = os.path.basename(args.input_file)
            archive_post(archive_subdir, "no_id", had_smile, "no_post", "no_respond", input_basename, args.input_file, final_movie, ".")
        exit(0)