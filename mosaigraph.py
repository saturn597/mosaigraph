#!/usr/bin/python

from __future__ import division

from PIL import Image

import argparse
import getopt
import json
import math
import random
import os
import re
import struct
import sys

# TODO: work on adding more sensible output
# TODO: for "unique" maybe move in this direction - find best overall fit rather than just going through each piece in order (which causes the image to get worse left to right) 
# TODO: add alternative means of comparing images (like something from scikit-img)
# TODO: add option for changing sample sizes - include option for sampling ALL pixels
# TODO: Work on python 3 compatibility
# TODO: add "satisficing" strategy as an option (rather than always trying to find the "best" match)
# TODO: get rid of dataset

class TooFewCandidatesException(Exception):
    pass

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

def get_sampler(pixelwise, sample_size, width = 1000, height = 1000):
    if pixelwise:
        return PixelwiseSampler(sample_size, width, height)
    else:
        return AverageSampler(sample_size, width, height)

class Sampler(object):
    def __init__(self, sample_size, pts_to_sample, width = 1000, height = 1000):
        #TODO: potentially remove width 
        self.sample_size = sample_size
        self.width = width
        self.height = height
        self.pts_to_sample = [(random.randint(0, self.width - 1), random.randint(0, self.height - 1)) for i in range(sample_size)]  # maybe use randrange
    
    def sample_image(self, image):
        image = make_proportional(image, self.width / self.height).resize((self.width, self.height)).convert(mode = 'RGB')  # maybe avoid actually resizing/reproportioning the image
        return [image.getpixel((x, y)) for x, y in self.pts_to_sample]

class AverageSampler(Sampler):

    def compare(self, imageA, imageB):
        avgA = get_color_avg(self.sample_image(imageA))
        avgB = get_color_avg(self.sample_image(imageB))
        return math.sqrt((avgA['r'] - avgB['r'])**2 + (avgA['g'] - avgB['g'])**2 + (avgA['b'] - avgB['b'])**2)

    # maybe give AverageSampler back its own sample_image - it can take random samples instead of using pts_to_sample every time

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
        #self.sample_size = sample_size  # this only matters to the sample we take of images we're trying to find a match for; the candidates have already been sampled
        self.sampler = sampler
        self.unique = unique

    def get_distances(self, image):
        rgb = get_color_avg(self.sampler.sample_image(image))
        for candidate in self.candidates:
            yield (candidate['r'] - rgb['r'])**2 + (candidate['g'] - rgb['g'])**2 + (candidate['b'] - rgb['b'])**2


def collect_candidates(dbtable, paths, sampler):
    candidates = []

    if dbtable:
        candidates = list(dbtable.all())  # TODO: do I need the list conversion here?

    paths = paths or []
    for path in paths:
        full_path = unicode(os.path.abspath(path), sys.getfilesystemencoding()) # TODO: Does this need a unicode conversion?
        new_candidate = dict(path = full_path)
        candidates.append(new_candidate)

    total_num = len(candidates)
    print('{} candidates to process'.format(total_num))

    unique_candidates = []  # we'll build a new list with duplicates eliminated so we don't do extra processing
    checked_files = set()

    # TODO: possible to do all of the following while building the candidates list, to avoid having to go through the list 2x?
    for candidate_num, candidate in enumerate(candidates):
        if candidate['path'] in checked_files:
            continue
        
        checked_files.add(candidate['path'])
        sys.stdout.write('\rProcessing candidate {} out of {}'.format(candidate_num + 1, total_num))
        sys.stdout.flush()

        if candidate.get('pixels') is not None:
            candidate['pixels'] = json.loads(candidate['pixels'])

        if candidate.get('pixels') is None:
            print('needed to process!')  # TODO: take out this line
            result = process_image(candidate['path'], sampler)
            if result:
                candidate.update(result)
            else:
                continue

        unique_candidates.append(candidate)

    print('')
    return unique_candidates

def process_image(path, sampler):
    try:
        image = make_proportional(Image.open(path).convert(mode = 'RGB'))
    except (IOError, struct.error):
        # IOError is usually because the file isn't an image file.
        # TODO: look into what's causing struct.error
        # TODO: check for struct.error elsewhere?
        return None
    
    d = dict(path = path)

    d['pixels'] = sampler.sample_image(image)
    # maybe add an option to suppress 'pixels' from the result 

    avg = get_color_avg(d['pixels'])
    d.update(avg)

    return d

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
    # If we have a rectangle of width and height, and we want to divide it into num_pieces squares of equal size
    # how long should each square's sides be to get as close as possible to num_pieces?
 
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

def preprocess(image_files, db, sampler):
    # process a given iterable of image_files, finding and storing the "average" color of the images in 
    # the given database table, or samples of pixels

    rows_to_insert = [] 

    num_files = len(image_files)
    print('{} files to process'.format(num_files))

    skipped = 0
    for file_num, filename in enumerate(image_files):

        path = os.path.abspath(unicode(filename, sys.getfilesystemencoding()))

        # if we already have a row in the db for this path, find it
        original_row = db['image_data'].find_one(path = path)
        if original_row:  
            # if we've already gotten the data we need for this path, don't do any more calculations
            if original_row.get('pixels') is not None:
                skipped += 1
                continue

        new_row = process_image(path, sampler)

        if not new_row:
            print('{} couldn\'t be processed'.format(path))
            continue

        sys.stdout.write('processed file number {}\r'.format(file_num + 1))
        sys.stdout.flush()
        
        if 'pixels' in new_row:
            new_row['pixels'] = json.dumps(new_row['pixels'])  # sqlite can't store lists - maybe switch backends? Or just go all out and json or pickle everything instead of using dataset
        
        if original_row:
            # if we already have a row in the db for this path, update it rather than adding a new row
            original_row.update(new_row)
            db['image_data'].update(original_row, ['path'])
        else:
            # otherwise, add it to our lists of rows to insert
            rows_to_insert.append(new_row)
    
    print('')

    if skipped:
        print('Skipped {} files that were already in the database'.format(skipped))

    db['image_data'].insert_many(rows_to_insert)  # might be better to do more often than just once at the end so that interruptions don't ruin everything during a long preprocessing period

def make_mosaigraph(img, num_pieces, matcher, edge_length, randomize_order, unique):
    # makes a photomosaic, returning a PIL.Image representing it, which can then be displayed, saved, etc, and a dict describing the images used and where

    # img: the base image
    # num_pieces: the number of square pieces we want the mosaic to have
    #   Actual number of pieces may differ. We are restricted by the fact that the pieces are square-shaped. Also, the lengths of the edges, as well as the number of
    #   divisions we make along the x and y axes, must be integers.
    # matcher: a matcher object that knows how to find images that fit well in a given section of the mosaic
    # edge_length: the pieces of the mosaic will have edges this long
    # randomize_order: randomize the order in which we add pieces to the mosaic
    # unique: if True, candidate images won't be reused within a given mosaic

    # TODO: subjective observation, randomize_order seems to make it slower. Is this correct?

    log = []  # record of which images we've used where in the mosaic so we can report back to our caller
    loaded_images = {}  # loading images is a very expensive operation - keep a cache of loaded images to minimize this

    width, height = img.size
    
    edge_length_orig = get_best_edge_length(width, height, num_pieces)  # original image will be broken into pieces of this length
    numX = width // edge_length_orig
    numY = height // edge_length_orig

    actual_num_pieces = numX * numY
    print('Dividing mosaic into {} pieces'.format(actual_num_pieces))

    if unique and actual_num_pieces > len(matcher.candidates):
        raise TooFewCandidatesException

    piece_ids = [(x, y) for x in range(0, numX) for y in range(0, numY)]

    # On "unique" mode, we can run low on good images to use. Randomizing the order in which we add each piece to the mosaic
    # prevents having one section of the mosaic look worse just because it was made later. (Though can just lead to an
    # image that looks equally bad all over).
    if randomize_order:
        random.shuffle(piece_ids)

    # create a blank image - we'll paste the mosaic onto this as we assemble it, piece by piece
    result_image = Image.new(img.mode, (numX * edge_length, numY * edge_length))

    # now iterate through each piece in the original image, find matching images among the candidates, and paste matches into result_image
    for current_num, (x, y) in enumerate(piece_ids):
        # TODO: now that I'm switching to using matching objects, compare with older versions to see if I lost any quality in resulting mosaics
        old_piece = img.crop((x * edge_length_orig, y * edge_length_orig, (x + 1) * edge_length_orig, (y + 1) * edge_length_orig))

        match = matcher.get_match(old_piece)
        path = match['path']

        if path in loaded_images:  # TODO: Maybe matcher should handle caching
            new_piece = loaded_images[path]  # get image from the cache if we can
        else:
            new_piece = make_proportional(Image.open(path)).resize((edge_length, edge_length))
        
        if not unique: 
            # no point in caching if unique, since we load a different image each time
            loaded_images[path] = new_piece  # if we must load the image, add it to our cache

        log.append({ 'x': x, 'y': y, 'path': path })
        result_image.paste(new_piece, (x * edge_length, y * edge_length))
        sys.stdout.write('\rCompleted piece {} out of {}'.format(current_num + 1, actual_num_pieces))
        sys.stdout.flush()

    print('')
    return result_image, log 

def main():
    # Process command line args. Depending on the result, either make a new photomosaic and display or save it (mosaic mode), or simply
    # preprocess images and save them to a database file (preprocessing mode)
    arg_parser = get_arg_parser()
    args = arg_parser.parse_args()

    sample_size = 300  # TODO: make it so this can be set by the user

    db = None
    loaded_sample_info = None
    pts_to_sample = None

    pixelwise = args.pixelwise or args.preprocess

    if args.dbfile:
        try:
            import dataset  # TODO: maybe just get rid of dataset, using json would be just as good and wouldn't add this dependency
        except ImportError:
            print('ERROR: dataset required to save a preprocessing file; you can install dataset with "pip install dataset"')
            sys.exit(1)
        db = dataset.connect('sqlite:///' + args.dbfile)
        loaded_sample_info = db['sample_info'].find_one(id = 0)

    if loaded_sample_info and pixelwise:
        # we need to sample the same spots in each image for pixelwise comparisons to be valid
        # so if we're using samples we stored in a db file, we need to pull info about 
        # which points we sampled and use the same points for any further comparisons

        pts_to_sample = json.loads(loaded_sample_info['pts_sampled'])

        if loaded_sample_info['sample_size'] != sample_size:
            # TODO: make this warning meaningful by allowing user to set sample size
            print('Warning! Not using specified sample size; using the one from the preprocessing file instead.')
            sample_size = loaded_sample_info['sample_size']

    sampler = get_sampler(pixelwise, sample_size)
    if pts_to_sample:
        sampler.pts_to_sample = pts_to_sample

    if args.preprocess:
        if not db:
            print('Must specify a filename for saving the preprocessing data. Use the -d argument.')
            sys.exit(1)
        # if we hadn't stored a sample size and set of points to sample in the db, write one now
        if not loaded_sample_info:
            db['sample_info'].insert({ 'id': 0, 'pts_sampled': json.dumps(sampler.pts_to_sample), 'sample_size': sample_size })
        preprocess(args.preprocess, db, sampler, pixelwise)

    elif args.filename:
        if args.outfile and os.path.exists(args.outfile):
            while True:
                input = raw_input('Overwrite existing file {}? (y/n) '.format(args.outfile))
                if input == 'y':
                    break
                if input == 'n':
                    print('Canceling...')
                    sys.exit(0)

        print('Making mosaic out of {}...'.format(args.filename))
       
        image_data = None
        if db:
            image_data = db['image_data']

        candidates = collect_candidates(image_data, args.imagelist, sampler)

        print('Using pool of {} candidate images'.format(len(candidates)))
        
        input_image = Image.open(args.filename)

        # TODO: should we pass width/height hints (i.e., edge_length_orig x edge_length_orig) to get_matcher?
        matcher = get_matcher(pixelwise, args.unique, candidates, sampler)

        try:
            mosaic, dict = make_mosaigraph(input_image, args.n, matcher, args.piecesize, args.randomize, args.unique)
        except TooFewCandidatesException:
            print('Not enough candidate images to use them uniquely! Provide more or make a mosaic with fewer or non-unique pieces.')
            sys.exit(1) 
      
        # maybe this should always be pixelwise
        print('New mosaic produced with average difference {}'.format(sampler.compare(input_image, mosaic)))

        if args.outfile:
            print('Saving mosaic as {}'.format(args.outfile))
            mosaic.save(args.outfile)
        else:
            if not args.nooutput:
                print('Showing output')
                mosaic.show()
        if args.json:
            print('Saving json output as {}'.format(args.json))
            with open(args.json, 'w') as f:
                json.dump(dict, f)

    else:
        print('Nothing to do!\n')
        arg_parser.print_help()
        sys.exit(0)

def get_arg_parser():
    arg_parser = argparse.ArgumentParser()
 
    # options/argument for mosaic mode only
    arg_parser.add_argument('-i', '--imagelist', metavar = 'IMAGES', nargs = '+', help = 'draw from this set of images in constructing mosaic')
    arg_parser.add_argument('-j', '--json', metavar = 'FILE', help = 'produce a json file FILE that shows which images were used where in the mosaic')
    arg_parser.add_argument('-n', '--number', dest = 'n', type = int, default = 500, help = 'the mosaic should consist of about this many pieces')
    arg_parser.add_argument('-o', '--outfile', metavar = 'FILE', help = 'save the mosaic as FILE; if not specified, will attempt to display the image in a default image viewer, but won\'t save it; format of output file is determined by extension (so .jpg for jpg, .png for png)')
    arg_parser.add_argument('-r', '--randomize', action = 'store_true', help = 'randomize the order in which pieces get added to the mosaic')
    arg_parser.add_argument('-u', '--unique', action = 'store_true', help = 'don\'t use any image as a piece of the mosaic more than once')
    arg_parser.add_argument('-x', '--nooutput', action = 'store_true', help = 'don\'t show mosaic file after it\'s built')
    arg_parser.add_argument('-z', '--piecesize', type = int, default = 100, help = 'each square piece of the resulting image will have edges this many pixels long; increase this value if the individual images are too pixelated and hard to make out, even when zoomed')

    arg_parser.add_argument('filename', nargs = '?', help = 'the filename of the image we\'re making a mosaic of')

    # option to turn on preprocessing mode
    arg_parser.add_argument('-p', '--preprocess', metavar = 'IMAGE', nargs = '+', help = 'switch to preprocessing mode; preprocess specified image file[s], adding to the pool of potential images in our database; options and arguments other than -g and -d will be ignored')

    # options usable in both mosaic mode and preprocessing mode
    arg_parser.add_argument('-d', '--dbfile', help = 'in mosaic mode, construct mosaic using images pointed to by this database file; in preprocessing mode, save the data to this file')
    arg_parser.add_argument('-w', '--pixelwise', action = 'store_true', help = 'in mosaic mode, a pixel-by-pixel comparison of candidate images with the base image, instead of just looking at overall average rgb color; in preprocessing mode, store samples of multiple pixels in each candidate image, useful for making pixelwise mosaics; slower, but better results than the default averaging mode')
 
    return arg_parser

if __name__ == '__main__':
    main()


