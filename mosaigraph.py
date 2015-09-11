#!/usr/bin/python

from __future__ import division

from PIL import Image

import argparse
import getopt
import json
import math
import os
import random
import re
import struct
import sys

# TODO: work on adding more sensible output
# TODO: for "unique" maybe move in this direction - find best overall fit rather than just going through each piece in order (which causes the image to get worse left to right) 
# TODO: add alternative means of comparing images (like something from scikit-img)
# TODO: Keep testing python 3 compatibility
# TODO: add "satisficing" strategy as an option (rather than always trying to find the "best" match)

# for python 2+3 compatibility
# per http://stackoverflow.com/questions/954834/how-do-i-use-raw-input-in-python-3-1
global input
try: input = raw_input
except NameError: pass

# Exception definitions
class ArgumentException(Exception):
    pass


class NoCandidatesException(Exception):
    pass


class TooFewCandidatesException(Exception):
    pass


# Samplers are objects that know how to take samples of sets of pixels and return information about their color. They wrap basic
# information about the sample size and about which pixels to sample. Reusing the same sampler between multiple images gives us
# consistency so that we can make meaningful comparisons between them. 
# 
# Matchers are objects that have access to a pool of "candidate" images and a sampler. If you pass an image to their "get_match" 
# method, they will use their sampler to compare that image to each of their candidates and return the candidate they decide is
# most visually similar to it.
# 
# Class definitions for matchers and samplers below, and a couple of convenience functions for returning the appropriate sampler
# or matcher, depending on the characteristics we want.

def get_sampler(pixelwise, sample_size, width = 1000, height = 1000):
    if pixelwise:
        return PixelwiseSampler(sample_size, width, height)
    else:
        return AverageSampler(sample_size, width, height)

def get_matcher(pixelwise, unique, candidates, sampler, width = 1000, height = 1000):
    # TODO: can width and height be removed as parameters? i.e., does passing the actual width/height of the images improve the result?
    if pixelwise:
        return PixelwiseMatcher(candidates, unique, sampler)
    elif not unique:
        try:
            return KDTreeRgbMatcher(candidates, sampler)
        except ImportError:
            print('scipy.spatial not found, will use a linear search through the candidates...')

    return RgbMatcher(candidates, unique, sampler)


class Sampler(object):
    def __init__(self, sample_size, pts_to_sample, width = 1000, height = 1000):
        #TODO: potentially remove width and height?
        self.sample_size = sample_size
        self.width = width
        self.height = height
        self.pts_to_sample = [(random.randint(0, self.width - 1), random.randint(0, self.height - 1)) for i in range(sample_size)]  # maybe use randrange
    
    def sample_image(self, image):
        # before taking a sample, have to scale every image to the same dimensions and clip them all to the same width / height ratio
        # otherwise, one pixel won't be comparable with another
        image = make_proportional(image, self.width / self.height).resize((self.width, self.height)).convert(mode = 'RGB') 
        return [image.getpixel((x, y)) for x, y in self.pts_to_sample]


class AverageSampler(Sampler):

    def compare(self, imageA, imageB):
        avgA = get_color_avg(self.sample_image(imageA))
        avgB = get_color_avg(self.sample_image(imageB))
        return math.sqrt((avgA['r'] - avgB['r'])**2 + (avgA['g'] - avgB['g'])**2 + (avgA['b'] - avgB['b'])**2)

    # maybe give AverageSampler back its own sample_image - it can take random samples instead of using pts_to_sample every time
    # problem is that won't work in preprocessing


class PixelwiseSampler(Sampler):

    def compare(self, imageA, imageB):
        # TODO: this is kind of redundant with get_distances - fix
        
        pixelsA = self.sample_image(imageA)
        pixelsB = self.sample_image(imageB)
        
        return self.pixel_dist(pixelsA, pixelsB)

    def pixel_dist(self, pixelsA, pixelsB):
        # consider taking this out of class if needed
        diff = 0

        for pixelA, pixelB in zip(pixelsA, pixelsB):
            diff += abs(pixelA[0] - pixelB[0]) + abs(pixelA[1] - pixelB[1]) + abs(pixelA[2] - pixelB[2])

        return diff / (self.sample_size * 3)


class Matcher(object):
    def __init__(self):
        self.unique = False

    def get_distances(self, image, candidate):
        pass

    def get_match(self, image):
        closest_image = None
        least_distance = None

        for n, distance in enumerate(self.get_distances(image)):
            if not least_distance or distance < least_distance:
                closest_index = n
                least_distance = distance

        result = self.candidates[closest_index]

        if self.unique:
            self.candidates.pop(closest_index)  # consider keeping a separate record of what we've used instead

        return result


class PixelwiseMatcher(Matcher):
    def __init__(self, candidates, unique, sampler):
        self.candidates = list(candidates)  # make our own copy since we may mutate the list

        self.sampler = sampler

        self.unique = unique

    def get_distances(self, image):
        pixels = self.sampler.sample_image(image)
        
        for candidate in self.candidates:

            yield self.sampler.pixel_dist(pixels, candidate['pixels'])


class KDTreeRgbMatcher(Matcher):
    def __init__(self, candidates, sampler):
        from scipy import spatial 
        self.candidates = list(candidates)
        self.sampler = sampler
        self.kdtree = spatial.KDTree([(candidate['r'], candidate['g'], candidate['b']) for candidate in candidates])

    def get_match(self, image):
        rgb = get_color_avg(self.sampler.sample_image(image))
        index = self.kdtree.query((rgb['r'], rgb['g'], rgb['b']))[1]
        return self.candidates[index]


class RgbMatcher(Matcher):
    # k-d trees are an efficient data structure for nearest neighbor search. However, using them makes it difficult to exclude 
    # items from the search that we've already used. (At least given scipy's implementation of k-d trees). So 
    # for now only using them when we don't mind reusing images. Also, not everyone has scipy installed. 
    # So, we need a non-KD-tree Rgb Matcher.

    def __init__(self, candidates, unique, sampler):
        self.candidates = list(candidates)
        self.sampler = sampler
        self.unique = unique

    def get_distances(self, image):
        rgb = get_color_avg(self.sampler.sample_image(image))
        for candidate in self.candidates:
            yield (candidate['r'] - rgb['r'])**2 + (candidate['g'] - rgb['g'])**2 + (candidate['b'] - rgb['b'])**2


def get_color_avg(pixels):
    # get the average color in a iterable of rgb pixels

    r, g, b = 0, 0, 0
    num_pixels = 0

    for pix in pixels:
        r += pix[0]
        g += pix[1]
        b += pix[2]
        num_pixels += 1
    
    r = r / num_pixels
    g = g / num_pixels
    b = b / num_pixels

    return {'r': int(r), 'g': int(g), 'b': int(b)}

def get_n_pixels(img, n):
    # return iterator through n randomly selected pixels in an image

    width, height = img.size
    for i in range(0, n):
        x = random.randint(0, width - 1)  # is subtracting 1 correct?
        y = random.randint(0, height - 1) 
        yield img.getpixel((x, y))

def get_best_edge_length(width, height, num_pieces):
    # If we have a rectangle of width and height, and we want to divide it into num_pieces squares of equal size,
    # return how long each square's sides be should to get as close as possible to num_pieces
 
    # get an initial estimate
    # maybe would be simpler to just start with an estimate of 1, then iterate up
    currentEstimate = int(math.sqrt(width * height / num_pieces))

    # get the difference between the number of pieces this gives and the number we want
    signedDiff = (width // currentEstimate) * (height // currentEstimate) - num_pieces
    lastDiff = abs(signedDiff)

    # if our estimate gets us the exact right number of pieces, we're done
    if lastDiff == 0:
        return currentEstimate

    # too many pieces means we may want to increase the edge size, so a positive "step"
    # too few means a negative step
    step = signedDiff // lastDiff  # maybe set this in a way that makes the idea clearer

    # now step through options to see if there's a better edge size
    # if the difference between what we get and what we want increases, we've gone too far
    while True:
        currentEstimate += step
        currentDiff = abs((width // currentEstimate) * (height // currentEstimate) - num_pieces)
        if currentDiff > lastDiff:
            return currentEstimate - step
        lastDiff = currentDiff

def make_proportional(img, ratio = 1):
    # returns a piece of the center of an image with a given width/height ratio
    # ratio defaults to 1

    width, height = img.size
    if width > height * ratio:
        # if the width is too big, cut the sides off the image by enough to make it right
        snipSize = int((width - height * ratio) / 2)
        return img.crop((snipSize, 0, width - snipSize, height))
    elif width < height * ratio:
        # if the height is too big, cut the top/bottom off the image by enough to make it right
        snipSize = int((height - width / ratio) / 2)
        return img.crop((0, snipSize, width, height - snipSize))
    else:
        return img
 
def expand_directories(paths):
    new_paths = []
    for path in paths:
      if os.path.isdir(path):
          new_paths = new_paths + [os.path.join(path, filename) for filename in os.listdir(path)]
      else:
          new_paths.append(path)
    return new_paths


def preprocess_paths(paths, preprocessed_data, sampler):
  if not paths:
      return {}

  result = {}
  
  paths = expand_directories(paths or [])
 
  failed_paths = []
  
  for n, path in enumerate(paths):
      full_path = os.path.abspath(path)

      sys.stdout.write('\rPreprocessing candidate {} out of {}'.format(n + 1, len(paths)))
      sys.stdout.flush()

      result[full_path] = preprocessed_data.get(full_path, {})

      if result[full_path].get('pixels') is None:
          result[full_path] = process_image(full_path, sampler)
          if not result[full_path]:
              del result[full_path]
              failed_paths.append(full_path)

  print('')

  for path in failed_paths:
      print('Note: unable to use file {}'.format(path))

  return result
              
def process_image(path, sampler):
    try:
        image = make_proportional(Image.open(path).convert(mode = 'RGB'))
    except (IOError, struct.error):
        # IOError is usually because the file isn't an image file.
        # TODO: look into what's causing struct.error
        # TODO: check for struct.error elsewhere?
        return None
    
    # maybe add an option to suppress 'pixels' from the result for an "average only" mode
    d = {'pixels': sampler.sample_image(image)}
    d.update(get_color_avg(d['pixels']))
    
    return d

def make_mosaigraph(img, num_pieces, matcher, scaled_piece_size, randomize_order, unique):
    # makes a photomosaic, returning a PIL.Image representing it, which can then be displayed, saved, etc, and a dict describing the images used and where

    # img: the base image
    # num_pieces: the number of square pieces we want the mosaic to have
    #   Actual number of pieces may differ. We are restricted by the fact that the pieces are square-shaped. Also, the lengths of the edges, as well as the number of
    #   divisions we make along the x and y axes, must be integers.
    # matcher: a matcher object that knows how to find images that fit well in a given section of the mosaic
    # scaled_piece_size: the pieces of the mosaic will have edges this long
    # randomize_order: randomize the order in which we add pieces to the mosaic
    # unique: if True, candidate images won't be reused within a given mosaic

    log = []  # record of which images we've used where in the mosaic so we can report back to our caller
    loaded_images = {}  # loading images is a very expensive operation - keep a cache of loaded images to minimize this

    width, height = img.size
    
    division_length = get_best_edge_length(width, height, num_pieces)  # original image will be broken into pieces of this length
    numX = width // division_length
    numY = height // division_length

    actual_num_pieces = numX * numY
    print('Dividing mosaic into {} pieces'.format(actual_num_pieces))

    if unique and actual_num_pieces > len(matcher.candidates):
        raise TooFewCandidatesException

    piece_ids = [(x, y) for x in range(0, numX) for y in range(0, numY)]

    # On "unique" mode, we can run low on good images to use. Randomizing the order in which we add each piece to the mosaic
    # prevents having a large section of the mosaic look worse just because it was made later. (Though can just lead to an
    # image that looks equally bad all over).
    if randomize_order:
        random.shuffle(piece_ids)

    # create a blank image - we'll paste the mosaic onto this as we assemble it, piece by piece
    result_image = Image.new(img.mode, (numX * scaled_piece_size, numY * scaled_piece_size))
    
    print('Beginning to select pieces...')

    # now iterate through each piece in the original image, find matching images among the candidates, and paste matches into result_image
    for current_num, (x, y) in enumerate(piece_ids):
        old_piece = img.crop((x * division_length, y * division_length, (x + 1) * division_length, (y + 1) * division_length))

        match = matcher.get_match(old_piece)
        path = match['path']

        if path in loaded_images:  # TODO: Maybe matcher should handle caching
            new_piece = loaded_images[path]  # get image from the cache if we can
        else:
            new_piece = make_proportional(Image.open(path)).resize((scaled_piece_size, scaled_piece_size))
            if not unique: 
                # no point in caching if unique, since we load a different image each time
                loaded_images[path] = new_piece  # if we must load the image, add it to our cache

        log.append({ 'x': x, 'y': y, 'path': path })
        result_image.paste(new_piece, (x * scaled_piece_size, y * scaled_piece_size))
        sys.stdout.write('\rCompleted piece {} out of {}'.format(current_num + 1, actual_num_pieces))
        sys.stdout.flush()

    print('')
    return result_image, log 

def main(args):
    # Process command line args. Depending on the args, decide whether to construct a mosaic and whether to save the preprocessing results

    save_preprocessing = args.preprocessingfile and not args.discardpreprocessing

    if not (args.baseimage or save_preprocessing):
        raise ArgumentException('Nothing to do!')

    sample_size = args.samplesize

    loaded_preprocessing_data = {}
    loaded_sampling_info = None  # information pulled from a preprocessing file about how the preprocessed images were sampled 

    if args.preprocessingfile:
        print('Loading preprocessing file...')
        try:
            with open(args.preprocessingfile, 'r') as f:
                loaded_preprocessing_data = json.load(f)
            loaded_sampling_info = loaded_preprocessing_data.get('sampling_info')
        except IOError as e:
            print('Note: couldn\'t open specified preprocessing file.') 
            if save_preprocessing:
                print('Will attempt to create it later on.')
                loaded_preprocessing_data = { 'sampling_info': {}, 'image_data': {} }

    # for pixelwise comparisons to make sense, we need to sample the same pixels in each image
    # thus, we need to use the same sample size as what's in the preprocessing file for the file to be useful
    if loaded_sampling_info and args.pixelwise:
        if loaded_sampling_info['sample_size'] != sample_size:
            print('Warning! Not using specified sample size; using the one from the preprocessing file instead.')
            sample_size = loaded_sampling_info['sample_size']

    # also, if we're using samples we stored in a preprocessing file, we need to pull info about 
    # which points we sampled and use the same points for any further comparisons
    sampler = get_sampler(args.pixelwise, sample_size)
    if loaded_sampling_info: 
        sampler.pts_to_sample = loaded_sampling_info['pts_sampled']

    all_image_data = loaded_preprocessing_data.get('image_data', {})
    old_len = len(all_image_data)

    preprocessed_paths = preprocess_paths(args.imagelist, loaded_preprocessing_data.get('image_data'), sampler)
    all_image_data.update(preprocessed_paths)

    if save_preprocessing and len(all_image_data) > old_len:
        data_to_save = {
              'image_data': all_image_data, 
              'sampling_info': loaded_sampling_info or { 'pts_sampled': sampler.pts_to_sample, 'sample_size': sample_size }
              }
        print('Writing preprocessed data to {}...'.format(args.preprocessingfile))
        with open(args.preprocessingfile, 'w') as f:
            # might be better to do more often than just once at the end so that interruptions don't ruin everything during a long preprocessing period
            json.dump(data_to_save, f) 

    candidate_dict = preprocessed_paths if args.explicitonly else all_image_data
    for path in candidate_dict:
        candidate_dict[path]['path'] = path  # we need the paths to be in the result of .values()

    candidate_list = candidate_dict.values()

    if args.baseimage:  # we got an image to turn into a mosaic
        if args.outfile and os.path.exists(args.outfile):
            while True:
                response = input('Overwrite existing file {}? (y/n) '.format(args.outfile))
                if response == 'y':
                    break
                if response == 'n':
                    print('Canceling...')
                    sys.exit(0)

        print('Making mosaic out of {}...'.format(args.baseimage))
       
        if len(candidate_list) == 0:
            raise NoCandidatesException

        print('Using pool of {} candidate images'.format(len(candidate_list)))
        
        input_image = Image.open(args.baseimage)
        matcher = get_matcher(args.pixelwise, args.unique, candidate_list, sampler)
        mosaic, dict = make_mosaigraph(input_image, args.n, matcher, args.edgesize, args.randomize, args.unique)
     
        result_tester = get_sampler(False, 1000)
        print('New mosaic produced with average difference {} from the original image'.format(result_tester.compare(input_image, mosaic)))

        if args.outfile:
            print('Saving mosaic as {}'.format(args.outfile))
            mosaic.save(args.outfile)
        else:
            if not args.nooutput:
                print('Showing output')
                mosaic.show()

        if args.log:
            print('Saving log as {}'.format(args.log))
            with open(args.log, 'w') as f:
                json.dump(dict, f)

def get_arg_parser():
    arg_parser = argparse.ArgumentParser()
 
    arg_parser.add_argument('baseimage', nargs = '?', help = 'the path to the image we\'re making a mosaic of')

    # options impacting mosaic construction

    arg_parser.add_argument('-c', '--explicitonly', action = 'store_true', help = 'confine candidate images to ones explicitly specified in the image list, rather than including all files in the preprocessing file')
    arg_parser.add_argument('-e', '--edgesize', type = int, default = 100, help = 'each square piece of the resulting image will have edges this many pixels long; increase this value if the individual images are too pixelated and hard to make out, even when zoomed')
    arg_parser.add_argument('-i', '--imagelist', metavar = 'IMAGES', nargs = '+', help = 'draw from this list of images in constructing the mosaic')
    arg_parser.add_argument('-n', '--number', dest = 'n', type = int, default = 500, help = 'specifies the approximate number of pieces the mosaic should consist of') 
    arg_parser.add_argument('-r', '--randomize', action = 'store_true', help = 'randomize the order in which pieces get added to the mosaic')
    arg_parser.add_argument('-u', '--unique', action = 'store_true', help = 'don\'t use any image as a piece of the mosaic more than once')
    arg_parser.add_argument('-w', '--pixelwise', action = 'store_true', help = 'use a pixel-by-pixel comparison in deciding on which images fit where in the mosaic, instead of just looking at overall average color; slower, but better results than the default averaging mode')

    # options impacting output
    arg_parser.add_argument('-o', '--outfile', metavar = 'FILE', help = 'save the mosaic as FILE; the format of the output is determined by the file extension used')
    arg_parser.add_argument('-x', '--nooutput', action = 'store_true', help = 'never display mosaic after building it')
    arg_parser.add_argument('-l', '--log', metavar = 'FILE', help = 'produce a json log file FILE that shows which images were used where in the mosaic')

    # preprocessing related
    arg_parser.add_argument('-d', '--discardpreprocessing', action = 'store_true', help = 'skip saving preprocessing to a file, even if a file was specified')
    arg_parser.add_argument('-p', '--preprocessingfile', help = 'if constructing mosaic, construct mosaic using images pointed to by this file; if images are specified that aren\'t already in file, save the preprocessing data here')
    arg_parser.add_argument('-s', '--samplesize', type = int, default = 300, help = 'sample size to use when examining images to decide which to use where in the mosaic')
 
    return arg_parser

if __name__ == '__main__':
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    try:
        main(args)
    except IOError as e:
        print('Failed to access file: {}'.format(e))
        sys.exit(1)
    except ArgumentException as e:
        print(str(e))
        arg_parser.print_help()
        sys.exit(1)
    except NoCandidatesException:
        print('No images given to use as pieces of the mosaic. Use the -i or -p arguments to provide some. If you provided a preprocessing file, make sure you provided the right path.')
        sys.exit(1)
    except TooFewCandidatesException:
        print('\nERROR: Not enough candidate images to use them uniquely! Provide more or make a mosaic with fewer or non-unique pieces.\n')
        sys.exit(1)
