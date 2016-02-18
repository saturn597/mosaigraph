#!/usr/bin/python

from __future__ import division

from PIL import Image

import argparse
import json
import math
import os
import random
import re
import struct
import sys


# for python 2+3 compatibility
# per http://stackoverflow.com/questions/954834/how-do-i-use-raw-input-in-python-3-1  # noqa
global input
try:
    input = raw_input
except NameError:
    pass


######## Exception definitions ########

class ArgumentException(Exception):
    pass


class NoCandidatesException(Exception):
    pass


class TooFewCandidatesException(Exception):
    pass


######## Utility functions ########

def get_color_avg(pixels):
    # Get the average color in an iterable of rgb pixels.

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


def get_best_edge_length(width, height, num_pieces):
    # If we have a rectangle of width and height, and we want to divide it into
    # num_pieces squares of equal size, this function says how long each
    # square's sides should be to accomplish that.

    # We assume that the side length and the number of squares both vertically
    # and horizontally must be integers, so our answer won't yield EXACTLY
    # num_pieces.

    # Note there may be multiple possible answers that get equally close. This
    # function only returns one of them.

    # Also note, finding the exact number algebraically and then rounding
    # wouldn't guarantee the best possible result. So, we use trial-and-error.

    # First, assume a side length of 2 (we check estimate - 1 later anyway).
    estimate = 2

    # Calculate the number of pieces the estimate yields and compare to
    # num_pieces.  Increase the estimate until we exceed num_pieces.
    while (width // estimate) * (height // estimate) > num_pieces:
        estimate += 1

    # Either our estimate is now the best value or (estimate - 1) is.  Find out
    # which one results in a number of pieces that's closest to num_pieces.
    poss1 = (width // estimate) * (height // estimate) - num_pieces
    poss2 = (width // (estimate - 1)) * (height // (estimate - 1)) - num_pieces

    if abs(poss1) > abs(poss2):
        return estimate - 1

    return estimate


def make_proportional(img, ratio=1):
    # Return a modified version of "img" with just enough of its edges cut off
    # to make the width / height ratio whatever is specified.

    width, height = img.size
    if width > height * ratio:
        # In this case, clip the left and right sides of the image to reduce
        # the width.
        snipSize = int((width - height * ratio) / 2)
        return img.crop((snipSize, 0, width - snipSize, height))
    elif width < height * ratio:
        # In this case, clip the top and bottom to reduce the height.
        snipSize = int((height - width / ratio) / 2)
        return img.crop((0, snipSize, width, height - snipSize))
    else:
        return img


def expand_directories(paths):
    # Take a list of paths. Return that list of paths, but with any directories
    # replaced by the files they contain (but don't recursively expand
    # subdirectories).

    new_paths = []
    for path in paths:
        if os.path.isdir(path):
            new_paths.extend(
                [os.path.join(path, file) for file in os.listdir(path)])
        else:
            new_paths.append(path)

    return new_paths


def preprocess_paths(paths, preprocessed_data, sampler):
    # Take a list of paths to images.

    # Within each image, take a random sample of pixels and find the color of
    # each of those pixels.

    # If our caller already has this information for any images, it can include
    # that data in the preprocessed_data argument. This way we won't have to
    # process the data again.

    # The sampler argument is an object of class Sampler we should use to
    # sample the pixels. The sampler determines which pixels to sample in each
    # image.

    # Returns a dictionary whose keys are the paths and whose values contain 1)
    # the colors of each pixel sampled and 2) the "average" color of those
    # pixels.

    paths = expand_directories(paths)

    result = {}
    failed_paths = []
    num_preprocessed = 0

    for n, path in enumerate(paths):
        full_path = os.path.abspath(path)

        sys.stdout.write('\rInspecting candidate {} out of {}'.format(
            n + 1, len(paths)))
        sys.stdout.flush()

        result[full_path] = preprocessed_data.get(full_path, {})

        if result[full_path].get('pixels') is None:
            result[full_path] = process_image(full_path, sampler)
            if not result[full_path]:
                del result[full_path]
                failed_paths.append(full_path)
            else:
                num_preprocessed += 1

    if paths:  # This is to format the output correctly.
        print('')

    print('{} images had to be preprocessed.'.format(num_preprocessed))

    if failed_paths:
        msg = ', '.join(failed_paths)
        print('\nCouldn\'t use these files: {}\n'.format(msg))

    return result


def process_image(path, sampler):
    # Take a path and open and inspect it.

    # If the path leads to an image, take a sample of pixels within the image,
    # and get the color values at each pixel.  Return a dictionary containing
    # those color values and an "average" color (with separate "r", "g" and "b"
    # components).

    # The sampler we're passed determines which pixels we sample.

    # If the path doesn't go to an image, return None.

    try:
        image = make_proportional(Image.open(path).convert(mode='RGB'))
    except (IOError, struct.error):
        # IOError is usually because the file isn't an image file.
        # TODO: look into what's causing struct.error
        return None

    # maybe add an option to suppress 'pixels' from the result for an "average
    # only" mode
    d = {'pixels': sampler.sample_image(image)}
    d.update(get_color_avg(d['pixels']))

    return d


######## Samplers ########

# Samplers are objects that know how to take samples of sets of pixels and
# return information about their color. They wrap basic information about the
# sample size and about which pixels to sample.

# Reusing the same sampler between multiple images gives us consistency,
# allowing us to make meaningful comparisons between the images. We can include
# a "compare" method to make use of this.

# Two sampler subclasses are included below, each of which has a different
# "compare" method.

def get_sampler(pixelwise, sample_size, width=1000, height=1000):
    # Return an appropriate sampler, depending on the intended sample size and
    # on whether it should be a "pixelwise" sampler

    # When images are sampled, they are all scaled to the same size for
    # consistency. The size used doesn't make a huge difference, but can be
    # specified through the "width" and "height" parameters if desired.
    if pixelwise:
        return PixelwiseSampler(sample_size, width, height)
    else:
        return AverageSampler(sample_size, width, height)


class Sampler(object):
    def __init__(self, sample_size, pts_to_sample, width=1000, height=1000):
        self.sample_size = sample_size
        self.width = width
        self.height = height
        self.pts_to_sample = [
            (random.randrange(width - 1), random.randrange(height - 1))
            for i in range(sample_size)]

    def sample_image(self, image):
        # Sample an image and return a list of the colors found in the sampled
        # pixels.

        # Scale and clip images first so that they're all comparable.
        image = make_proportional(image, self.width / self.height).resize(
            (self.width, self.height)).convert(mode='RGB')
        return [image.getpixel((x, y)) for x, y in self.pts_to_sample]


class AverageSampler(Sampler):
    # A sampler that also knows how to compare two images by finding the
    # "average" color of each then finding the distance between the two colors.

    # This is used for the "fast" method of mosaic construction.

    def compare(self, imageA, imageB):
        avgA = get_color_avg(self.sample_image(imageA))
        avgB = get_color_avg(self.sample_image(imageB))
        return math.sqrt(
            (avgA['r'] - avgB['r'])**2 +
            (avgA['g'] - avgB['g'])**2 +
            (avgA['b'] - avgB['b'])**2)


class PixelwiseSampler(Sampler):
    # A sampler that knows how to compare two images via a "pixelwise" method.
    #
    # "Pixelwise" comparison finds the distance between two images in the
    # following way.
    #
    # Take two images with the same dimensions. Take a pixel from one and a
    # pixel from the other. This pair of pixels should be in the same place -
    # so both are at (27, 56) or both are at (942, 458), etc. Find the
    # "distance" between the colors of those two pixels. Now, repeat this
    # process with a number of pixel pairs. The sum of the distances (averaged
    # over the number of pixels tested) tells you the "pixelwise" distance
    # between the images.

    # This is used for the "slow" method of mosaic construction.

    def compare(self, imageA, imageB):
        pixelsA = self.sample_image(imageA)
        pixelsB = self.sample_image(imageB)

        return self.pixel_dist(pixelsA, pixelsB)

    def pixel_dist(self, pixelsA, pixelsB):
        diff = 0

        # Consider squaring instead of abs-ing here.
        for a, b in zip(pixelsA, pixelsB):
            diff += abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

        return diff / (self.sample_size * 3)


######## Matchers ########

# Matchers are objects that have access to a pool of "candidate" images and a
# sampler. If you pass an image to their "get_match" method, they will try to
# find the candidate that most resembles it. The precise meaning of "resembles"
# depends in part on the sampler.

def get_matcher(pixelwise, unique, candidates, sampler):
    # Return an appropriate matcher, depending on the parameters.

    if pixelwise:
        return PixelwiseMatcher(candidates, unique, sampler)
    elif not unique:
        try:
            return KDTreeAverageMatcher(candidates, sampler)
        except ImportError:
            print('scipy.spatial not found. '
                  'Will use a linear search through the candidates...')

    return AverageMatcher(candidates, unique, sampler)


class Matcher(object):
    def get_distances(self, image, candidate):
        # The base get_match method uses this method to find the distances
        # between the candidate images and the base image. Matchers implement
        # it to define their approach to finding matches.
        pass

    def get_match(self, image):
        # Conduct a linear search through the candidates to find the one that
        # is most visually similar to "image." The visual similarity of
        # candidates to the base image is defined by the "get_distances"
        # method.

        closest_image = None
        least_distance = None

        for n, distance in enumerate(self.get_distances(image)):
            if not least_distance or distance < least_distance:
                closest_index = n
                least_distance = distance

        result = self.candidates[closest_index]

        if self.unique:
            self.candidates.pop(closest_index)

        return result


class PixelwiseMatcher(Matcher):
    # This matcher finds matches based on the "pixelwise" method of image
    # comparison, discussed in the PixelwiseSampler discussion above. Used for
    # the "slow" method of mosaic creation.

    def __init__(self, candidates, unique, sampler):
        # make our own copy of the candidates since we may mutate the list
        self.candidates = list(candidates)

        self.sampler = sampler

        self.unique = unique

    def get_distances(self, image):
        pixels = self.sampler.sample_image(image)

        for candidate in self.candidates:

            yield self.sampler.pixel_dist(pixels, candidate['pixels'])


class AverageMatcher(Matcher):
    # AverageMatchers find matches by comparing the average color of the input
    # image with the average color of each of the candidate images, and
    # returning the candidate whose average color is least different from that
    # of the input image.

    def __init__(self, candidates, unique, sampler):
        self.candidates = list(candidates)
        self.sampler = sampler
        self.unique = unique

    def get_distances(self, image):
        i = get_color_avg(self.sampler.sample_image(image))
        r, g, b = i['r'], i['g'], i['b']
        for c in self.candidates:
            yield (r - c['r'])**2 + (g - c['g'])**2 + (b - c['b'])**2


class KDTreeAverageMatcher(Matcher):
    # k-d trees are an efficient data structure for "nearest neightbor" search.
    # If we know the average colors of each candidate image, we can plug them
    # into a k-d tree, and use that to easily find the candidate that's closest
    # to a given color.

    # This uses scipy's implementation of k-d trees. If scipy isn't present,
    # we just have to use AverageMatcher instead, which should yield similar
    # results, just more slowly.

    # Also, using scipy's implementation of k-d trees makes it difficult to
    # exclude items from the search that we've already used. So for now this
    # can only be used when we don't mind reusing images (like if the "-u" flag
    # isn't set).

    def __init__(self, candidates, sampler):
        from scipy import spatial
        self.candidates = list(candidates)
        self.sampler = sampler
        self.kdtree = spatial.KDTree(
            [(c['r'], c['g'], c['b']) for c in candidates])
        self.unique = False

    def get_match(self, image):
        rgb = get_color_avg(self.sampler.sample_image(image))
        index = self.kdtree.query((rgb['r'], rgb['g'], rgb['b']))[1]
        return self.candidates[index]


######## Core functions ########

def make_mosaic(img, n, matcher, scaled_piece_size, random_order, unique):
    # Makes a photomosaic, returning a PIL.Image representing it.

    # Also returns a "log" in the form of a dict describing which images were
    # used where.

    # img: the base image
    # n: the number of square pieces we want the mosaic to have.
    #   Actual number of pieces may differ. We are restricted by the fact that
    #   the pieces are square-shaped. Also, the lengths of the edges, as well
    #   as the number of divisions we make along the x and y axes, must be
    #   integers.
    # matcher: a matcher object that knows how to find images that fit well in
    #   a given section of the mosaic
    # scaled_piece_size: the pieces of the mosaic will have sides this long
    # random_order: randomize the order in which we add pieces to the mosaic
    # unique: if True, candidate images won't be reused within a given mosaic

    log = []  # record of which images we've used where in the mosaic
    loaded_images = {}  # cache of loaded images

    width, height = img.size

    # We'll divide the base image into a series of square sections with a side
    # length of division_length. The mosaic will be "numX" squares wide and
    # "numY" squares tall.
    division_length = get_best_edge_length(width, height, n)
    numX = width // division_length
    numY = height // division_length

    actual_num_pieces = numX * numY
    print('Dividing mosaic into {} pieces'.format(actual_num_pieces))

    if unique and actual_num_pieces > len(matcher.candidates):
        raise TooFewCandidatesException

    # To process all the "pieces" of the mosaic, we'll iterate through these
    # piece_ids.
    piece_ids = [(x, y) for x in range(0, numX) for y in range(0, numY)]

    if random_order:
        random.shuffle(piece_ids)

    # We'll paste the mosaic onto result_image as we construct it
    result_image = Image.new(
        img.mode, (numX * scaled_piece_size, numY * scaled_piece_size))

    print('Beginning to select pieces...')

    for current_num, (x, y) in enumerate(piece_ids):
        # Crop out the piece of the base image we're currently examining
        old_piece = img.crop((
            x * division_length,
            y * division_length,
            (x + 1) * division_length,
            (y + 1) * division_length))

        # Look for a candidate that matches "old_piece" and get its path.
        path = matcher.get_match(old_piece)['path']

        # Load that candidate so we can paste it into the result image.
        if path in loaded_images:
            new_piece = loaded_images[path]  # get image from cache if we can
        else:
            new_piece = make_proportional(Image.open(path)).resize(
                (scaled_piece_size, scaled_piece_size))
            if not unique:
                # cache image load if we might be reusing the image
                loaded_images[path] = new_piece

        log.append({'x': x, 'y': y, 'path': path})
        result_image.paste(
            new_piece, (x * scaled_piece_size, y * scaled_piece_size))
        sys.stdout.write('\rCompleted piece {} out of {}'.format(
            current_num + 1, actual_num_pieces))
        sys.stdout.flush()

    print('')
    return result_image, log


def main(args):
    # Depending on the command-line args, 1) load information from any
    # preprocessing file we were given, 2) preprocess any additional images we
    # were told to preprocess, 3) save any preprocessing data we were told to
    # save, then, 4) if we were given a base image, construct a mosaic.

    if args.loadpreprocessing and args.savepreprocessing:
        raise ArgumentException('Can use either -p or -P, but not both')

    if not (args.baseimage or args.savepreprocessing):
        raise ArgumentException('Nothing to do!')

    loaded_preprocessing_data = {}
    loaded_sampling_info = None  # info on how preprocessed images were sampled

    preprocessing_path = args.loadpreprocessing or args.savepreprocessing

    if preprocessing_path:
        print('Loading preprocessing file...')
        try:
            with open(preprocessing_path, 'r') as f:
                loaded_preprocessing_data = json.load(f)
            loaded_sampling_info = loaded_preprocessing_data.get(
                'sampling_info')
        except IOError as e:
            print('Note: couldn\'t open specified preprocessing file.')
            if args.savepreprocessing:
                print('Will attempt to create it later on.')

    sample_size = args.samplesize

    # For pixelwise comparisons to make sense, we need to sample the same
    # pixels in each image. Thus, we need to use the same sample size as what's
    # in the preprocessing file, if there was a preprocessing file.
    if loaded_sampling_info and args.pixelwise:
        if loaded_sampling_info['sample_size'] != sample_size:
            print('''Warning! Not using specified sample size.
                    Using sample size from the preprocessing file instead.''')
            sample_size = loaded_sampling_info['sample_size']

    sampler = get_sampler(args.pixelwise, sample_size)

    # Also, if we're using samples we stored in a preprocessing file, we need
    # to pull info about which points we sampled and use the same points for
    # any further comparisons
    if loaded_sampling_info:
        sampler.pts_to_sample = loaded_sampling_info['pts_sampled']

    # Two ways to get candidate images. One is from the preprocessing file.
    candidate_dict = loaded_preprocessing_data.get('image_data', {})
    old_len = len(candidate_dict)

    # The other is through the "-i" argument (args.imagelist). Preprocess any
    # additional images and add them to candidate_dict.
    imagelist_data = preprocess_paths(args.imagelist, candidate_dict, sampler)
    candidate_dict.update(imagelist_data)

    # If the len of candidate_dict increased, then we've preprocessed images
    # that weren't in our preprocessing file. Save any new data if the user
    # wants to (i.e., if they've set the "-P" flag).
    if args.savepreprocessing and len(candidate_dict) > old_len:
        data_to_save = {
            'image_data': candidate_dict,
            'sampling_info': loaded_sampling_info or {
                'pts_sampled': sampler.pts_to_sample,
                'sample_size': sample_size}
            }
        print('Writing preprocessed data to {}...'.format(preprocessing_path))
        with open(preprocessing_path, 'w') as f:
            # TODO: might be better to save more often than just once at the
            # end, so that interruptions don't ruin everything.
            json.dump(data_to_save, f)

    if args.explicitonly:
        candidate_dict = imagelist_data

    for path in candidate_dict:
        # About to convert candidate_dict to a list - make sure we'll still
        # know the paths.
        candidate_dict[path]['path'] = path

    candidate_list = candidate_dict.values()

    # If we got a "baseimage" arg, then the user wants to make a new mosaic.
    if args.baseimage:
        if args.outfile and os.path.exists(args.outfile):
            while True:
                response = input(
                    'Overwrite existing file {}? (y/n) '.format(
                        args.outfile))
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
        matcher = get_matcher(
            args.pixelwise, args.unique, candidate_list, sampler)
        mosaic, log = make_mosaic(
            input_image, args.n, matcher, args.edgesize, args.randomize,
            args.unique)

        result_tester = get_sampler(False, 1000)
        print(
            ('New mosaic produced with average difference {} from the '
                'original image').format(
                result_tester.compare(input_image, mosaic)))

        if args.outfile:
            print('Saving mosaic as {}'.format(args.outfile))
            mosaic.save(args.outfile)

        if not args.nooutput:
            print('Showing output')
            mosaic.show()

        if args.log:
            print('Saving log as {}'.format(args.log))
            with open(args.log, 'w') as f:
                json.dump(log, f)


def get_arg_parser():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('baseimage', nargs='?', help='''the path to the
            image we\'re making a mosaic of''')

    # options impacting mosaic construction

    arg_parser.add_argument(
        '-c', '--explicitonly', action='store_true',
        help='''confine candidate images to ones explicitly specified in the
        image list (argument to "-i") rather than including all files in the
        preprocessing file''')

    arg_parser.add_argument(
        '-e', '--edgesize', type=int, default=100,
        help='''each square piece of the resulting image will have edges this
        many pixels long; increase this value if the individual images are too
        pixelated and hard to make out, even when zoomed''')

    arg_parser.add_argument(
        '-i', '--imagelist', metavar='IMAGES', default=[],
        nargs='+',
        help='draw from this list of images in constructing the mosaic')

    arg_parser.add_argument(
        '-n', '--number', dest='n', type=int,
        default=500,
        help='''specifies the approximate number of pieces the mosaic should
        consist of''')

    arg_parser.add_argument(
        '-r', '--randomize', action='store_true',
        help='''randomize the order in which pieces get added to the mosaic''')

    arg_parser.add_argument(
        '-u', '--unique', action='store_true',
        help='''don\'t use any image as a piece of the mosaic more than
        once''')

    arg_parser.add_argument(
        '-w', '--pixelwise', action='store_true',
        help='''use a pixel-by-pixel comparison in deciding on which images fit
        where in the mosaic, instead of just looking at overall average color;
        slower, but better results than the default averaging mode''')

    # options impacting output
    arg_parser.add_argument(
        '-o', '--outfile', metavar='FILE',
        help='''save the mosaic as FILE; the format of the output is determined
        by the file extension used''')

    arg_parser.add_argument(
        '-x', '--nooutput', action='store_true',
        help='never display mosaic after building it')

    arg_parser.add_argument(
        '-l', '--log', metavar='FILE',
        help='''produce a json log file FILE that shows which images were used
        where in the mosaic''')

    # preprocessing related
    arg_parser.add_argument(
        '-P', '--savepreprocessing',
        help='save preprocessing data to this file')

    arg_parser.add_argument(
        '-p', '--loadpreprocessing',
        help='''if constructing mosaic, construct mosaic using images pointed
        to by this file''')

    arg_parser.add_argument(
        '-s', '--samplesize', type=int, default=300,
        help='''sample size to use when examining images to decide which to use
        where in the mosaic''')

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
        sys.exit(1)
    except NoCandidatesException:
        print('No images given to use as pieces of the mosaic. Use the -i or '
              '-p arguments to provide some. If you provided a preprocessing '
              'file, make sure you provided the right path.')
        sys.exit(1)
    except TooFewCandidatesException:
        print('\nERROR: Not enough candidate images to use them uniquely! '
              'Provide more or make a mosaic with fewer or non-unique pieces.'
              '\n')
        sys.exit(1)
