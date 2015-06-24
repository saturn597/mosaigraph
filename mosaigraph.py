from __future__ import division

from PIL import Image
from scipy import spatial

import argparse
import dataset
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
# TODO: Make this give appropriate errors in case the user requires uniqueness but doesn't have a big enough pool of images
# TODO: add alternative means of comparing images (like something from scikit-img)
# TODO: add option for changing sample sizes - include option for sampling ALL pixels
# TODO: Test in python 3
# TODO: add "satisficing" strategy as an option (rather than always trying to find the "best" match)
# TODO: Use "" so I don't have to escape '
# TODO: Need to handle the case where a database has pixelwise info but not average rgb info and vice versa - though maybe also I can always do an everage when preprocessing (easy enough to 
#   average the pixels we've already sampled in pixelwise preprocessing

def get_matcher(pixelwise, unique, candidates, sample_size, width = 1000, height = 1000, sampler = None):
    # TODO: can width and height be removed as parameters? i.e., does passing the actual width/height of the images improve the result?
    if pixelwise:
        return PixelwiseMatcher(candidates, sample_size, unique, width, height, sampler)
    elif unique:
        return RgbMatcher(candidates, sample_size, unique)
    else:
        return KDTreeRgbMatcher(candidates, sample_size)


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

class PixelwiseSampler(object):
    def __init__(self, sample_size, width = 1000, height = 1000):
        # TODO: potentially remove width and height
        self.sample_size = sample_size
        self.width = width
        self.height = height
        self.pts_to_sample = [(random.randint(0, self.width - 1), random.randint(0, self.height - 1)) for i in range(sample_size)]  # maybe use randrange

    def sample_path(self, path):
        # Returns None when image load fails
        try: 
            return self.sample_image(Image.open(path))
        except IOError:  # TODO: do exception checking wherever we have an Image.open
            return None

    def sample_image(self, image):
        image = make_proportional(image, self.width / self.height).resize((self.width, self.height)).convert(mode = 'RGB')  # maybe avoid actually resizing/reproportioning the image
        return [image.getpixel((x, y)) for x, y in self.pts_to_sample]

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

class PixelwiseMatcher(Matcher):
    def __init__(self, candidates, sample_size, unique, width = 1000, height = 1000, sampler = PixelwiseSampler(300)):
        # TODO: Can omit width/height as parameters if arbitrary resizing works just as well (in TODO below)
        self.candidates = list(candidates)  # make our own copy since we may mutate the list

        self.sampler = sampler

        self.unique = unique

    def get_distances(self, image):
        pixels = self.sampler.sample_image(image)
        
        for candidate in self.candidates:

            yield self.sampler.pixel_dist(pixels, candidate['pixels'])


class KDTreeRgbMatcher(Matcher):
    def __init__(self, candidates, sample_size):
        self.candidates = list(candidates)
        self.sample_size = sample_size
        self.kdtree = spatial.KDTree([(candidate['r'], candidate['g'], candidate['b']) for candidate in candidates])

    def get_match(self, image):
        rgb = get_color_avg(get_n_pixels(image, self.sample_size))
        index = self.kdtree.query((rgb['r'], rgb['g'], rgb['b']))[1]
        return self.candidates[index]


class RgbMatcher(Matcher):
    # k-d trees are an efficient data structure for nearest neighbor search. However, using them makes it difficult to exclude 
    # items from the search that we've already used. (At least given scipy's implementation of k-d trees). So 
    # for now only using them when we don't mind reusing images - so we need a non-KD-tree Rgb Matcher.
    def __init__(self, candidates, sample_size, unique = False):
        self.candidates = list(candidates)
        self.sample_size = sample_size  # this only matters to the sample we take of images we're trying to find a match for; the candidates have already been sampled
        self.unique = unique

    def get_distances(self, image):
        rgb = get_color_avg(get_n_pixels(image, self.sample_size))
        for candidate in self.candidates:
            yield (candidate['r'] - rgb['r'])**2 + (candidate['g'] - rgb['g'])**2 + (candidate['b'] - rgb['b'])**2


class ImageData(object):
    def __init__(self, path = None, image = None, x = None, y = None, rgb = None):
        self.path = path
        self.image = image
        self.rgb = rgb
        self.x = x
        self.y = y
        self.closest_distance = None
        self.match = None

    def get_image(self):
        if self.image:
            return self.image

        try:
            self.image = make_proportional(Image.open(self.path).convert(mode = "RGB"))
        except (IOError, struct.error):
            # IOError is usually because the file isn't an image file.
            # TODO: look into what's causing struct.error
            pass
        
        return self.image

    def get_rgb(self):
        if self.rgb:
            return self.rgb

        avg = get_color_avg(get_n_pixels(self.get_image(), 300))
        self.rgb = (avg['r'], avg['g'], avg['b'])  

        return self.rgb

def collect_candidates(dbtable, paths, rgb_needed, pixels_needed, sampler):
    # work on ability to use paths AND db
    # work on what happens if a db has rgb but not pixels and vice versa

    candidates = []

    if dbtable:
        candidates = list(dbtable.all())

    paths = paths or []
    for path in paths:
        path = unicode(path, sys.getfilesystemencoding()) # TODO: Does this need a unicode conversion?
        full_path = os.path.abspath(path)  
        new_candidate = dict(path = full_path)

        candidates.append(new_candidate)
   
    total_num = len(candidates)
    print('{} candidates to process'.format(total_num))

    for candidate_num, candidate in enumerate(candidates):
        sys.stdout.write("\rProcessing candidate {} out of {}".format(candidate_num + 1, total_num))
        sys.stdout.flush()

        if candidate.get('pixels') is not None:
            candidate['pixels'] = json.loads(candidate['pixels'])

        if (rgb_needed and candidate.get('r') is None) or (pixels_needed and candidate.get('pixels') is None):
            result = process_image(candidate['path'], rgb_needed, pixels_needed, sampler)
            if result:
                candidate.update(result)
            else:
                continue

    print("")
    return candidates

def collect_candidates2(dbtable, paths = [], rgb_needed = False):
    candidates = collect_candidates(dbtable, paths = [], rgb_needed = False)

    for candidate in candidates:
        yield ImageData(path = candidate['path'], rgb = (candidate['r'], candidate['g'], candidate['b']))

def get_matches(images, candidates, pixelwise = False, unique = False):

    # add pixelwise
    if not pixelwise and not unique:
        kdtree = spatial.KDTree([item.get_rgb() for item in candidates])
    
    for image in images:
        if unique:
            distance, match_index = find_match_linear2(image.get_rgb(), candidates)
            match = candidates[match_index]
        else:
            distance, match_index = kdtree.query(image.get_rgb())
            match = candidates[match_index]

        image.closest_distance = distance
        image.match = match

        if not match.closest_distance or match.closest_distance and distance <= match.closest_distance:
            match.closest_distance = distance
            match.match = image


def get_piece(image, edge_length, x, y):
    new_image = image.crop((x * edge_length, y * edge_length, (x + 1) * edge_length, (y + 1) * edge_length))  
    return ImageData(image = new_image, x = x, y = y)

def make_mosaigraph2(img, candidates, num_pieces, edge_length = 100, unique = False, randomize_order = False, pixelwise = False):
    # makes a photomosaic, returning a PIL.Image representing it, which can then be displayed, saved, etc, and a dict describing the images used and where

    # TODO: subjective observation, randomize_order seems to make it slower. Is this correct?
    # TODO: replace incoming img with ImageData

    width, height = img.size
        
    edge_length = get_best_edge_length(width, height, num_pieces)
    numX = width // edge_length
    numY = height // edge_length
    num_pieces = numX * numY
    print("Dividing mosaic into {} pieces".format(num_pieces))

    result_image = Image.new(img.mode, (width // edge_length * edge_length, height // edge_length * edge_length))

    current_num = 0

    piece_ids = [get_piece(img, edge_length, x, y) for x in range(0, numX) for y in range(0, numY)]
         
    # if we have a limited pool of images to pick from, pieces of the mosaic added later are likely to look
    # worse than ones added earlier, as we run out of good matches. Offering option to randomize the order in 
    # which we add pieces to the mosaic so that one region doesn't look better/worse than another. Not 
    # confident this is necessarily better - seems to lead to an image that's equally bad all over. 
    if randomize_order:
        random.shuffle(piece_ids)
   
    paths = []
    passnum = 0

    if unique:
        undone_pieces = [piece for piece in piece_ids if not piece.match]
        while undone_pieces:
            passnum += 1
            print "beginning pass: {} images remaining".format(len(undone_pieces))
            get_matches(candidates, undone_pieces, pixelwise, unique)
            matches = set([piece.match for piece in piece_ids if piece.match])
            undone_pieces = [piece for piece in piece_ids if not piece.match]
            candidates = [candidate for candidate in candidates if candidate not in matches]
    else:
        get_matches(piece_ids, candidates, pixelwise, unique)

    def prep_image(image):
        proportional = make_proportional(image.get_image())
        return proportional.resize((edge_length, edge_length))
    
    print piece_ids
    for piece in piece_ids:
        new_piece = piece.match
        try: 
            result_image.paste(prep_image(new_piece), (piece.x * edge_length, piece.y * edge_length))
        except:
            continue
        paths.append({ 'x': piece.x, 'y': piece.y, 'path': new_piece.path })
        current_num += 1  # maybe use enumerate in for instead
        sys.stdout.write("\rCompleted piece " + str(current_num) + " out of " + str(num_pieces))
        sys.stdout.flush()
    print("")

    return result_image, paths


def make_mosaigraph(img, num_pieces, matcher, edge_length = 100, randomize_order = False, unique = False):
    # makes a photomosaic, returning a PIL.Image representing it, which can then be displayed, saved, etc, and a dict describing the images used and where

    # TODO: fix arguments list - some of these descriptions could go elsewhere
    # img: the base image
    # num_pieces: the number of square pieces we want the mosaic to have
    #   Actual number of pieces may differ. We are restricted by the fact that the pieces are square-shaped. Also, the lengths of the edges, as well as the number of
    #   divisions we make along the x and y axes, must be integers.
    # edge_length: the pieces of the mosaic will have edges this long
    # unique: if True, candidate images won't be reused within a given mosaic
    # randomize_order: randomize the order in which we add pieces to the mosaic
    # pixelwise: determines the method for matching candidates to the specific section of the base image that they are intended to stand for
    #   If True, two same-sized images match if the color of pixel (x, y) in image A is likely to be similar to that of pixel (x, y) in image B, for any given (x, y).
    #   If False, two images match if the overall color of one is close to the overall color of the other, based on averaging a sample drawn from all pixels.
    #   Pixelwise matches yield better results but are slow.
    
    # TODO: subjective observation, randomize_order seems to make it slower. Is this correct?

    log = []  # record of which images we've used where in the mosaic so we can report back to our caller
    loaded_images = {}  # loading images is a very expensive operation - keep a cache of loaded images to minimize this

    width, height = img.size
    
    edge_length_orig = get_best_edge_length(width, height, num_pieces)  # original image will be broken into pieces of this length
    numX = width // edge_length_orig
    numY = height // edge_length_orig

    actual_num_pieces = numX * numY
    print("Dividing mosaic into {} pieces".format(actual_num_pieces))

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
        sys.stdout.write("\rCompleted piece " + str(current_num + 1) + " out of " + str(actual_num_pieces))
        sys.stdout.flush()

    print("")
    return result_image, log 


def find_match(original, candidates, comparison_fn):
    closest_image = None
    least_distance = None

    for n, distance in enumerate(comparison_fn(original, candidates)):
        if not least_distance or distance < least_distance:
            closest_index = n
            least_distance = distance

    return closest_index

def find_match_orig(original, candidates, comparison_fn):
      # maybe combine with find_match_linear
      closest_image = None
      least_distance = None

      for n, candidate in enumerate(candidates):
          distance = comparison_fn(original, candidate['path'])
          if not least_distance or distance < least_distance:
                closest_index = n
                least_distance = distance

      return closest_index

def compare_rgb(original, candidates):
    rgb = get_color_avg(get_n_pixels(original, 300))
    for candidate in candidates:
        yield (candidate['r'] - rgb['r'])**2 + (candidate['g'] - rgb['g'])**2 + (candidate['b'] - rgb['b'])**2

def compare_pixelwise2(img, candidates, n = 300):
    width, height = img.size

    global pts_to_sample  # Need these points to be consistent between calls, or data from the image_cache will be bogus
    if not pts_to_sample:
        pts_to_sample = [(random.randint(0, width - 1), random.randint(0, height - 1)) for i in range(n)]

    pixels = [img.getpixel((x, y)) for x, y in pts_to_sample]

    for candidate in candidates:
        path = candidate['path']
        if path in image_cache:  # I think there may be a pythonic way to avoid this idiom; also, make this its own function
            pixels2 = image_cache[path]
        else:
            try:
                img2 = make_proportional(Image.open(path), width / height).resize((width, height)).convert(mode = 'RGB')  # maybe avoid actually resizing/reproportioning the image
            except IOError:
                print('Couldn\'t process file {} as image'.format(path))
                continue
            pixels2 = [img2.getpixel((x, y)) for x, y in pts_to_sample]
            image_cache[path] = pixels2

        diff = 0
        
        for pixel1, pixel2 in zip(pixels, pixels2):
            diff += abs(pixel1[0] - pixel2[0]) + abs(pixel1[1] - pixel2[1]) + abs(pixel1[2] - pixel2[2])
        yield diff


def compare_pixelwise(img, img2 = None, path = None, n = 300):
    # combine with compare_pixelwise2

    width, height = img.size

    if not img2:
        if not path:
            raise  # TODO: Make this better
        img2 = Image.open(path)

    if path and path in image_cache:  # I think there may be a pythonic way to avoid this idiom; also, make this its own function
        img2 = image_cache[path] 
    else:
        img2 = make_proportional(img2, width / height).resize((width, height)).convert(mode = 'RGB')  # maybe avoid actually resizing/reproportioning the image
        image_cache[path] = img2

    pixels = [(random.randint(0, width - 1), random.randint(0, height - 1)) for i in range(n)]
    
    diff = 0
    
    for pixel in pixels:
        pixel1 = img.getpixel((pixel[0], pixel[1]))
        pixel2 = img2.getpixel((pixel[0], pixel[1]))  # probably don't need to sample the original image repeatedly
        diff1 = abs(pixel1[0] - pixel2[0]) + abs(pixel1[1] - pixel2[1]) + abs(pixel1[2] - pixel2[2])
        diff += diff1
  
    return diff

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

def store_image_data(image_files, db, sample_size = 1, pixelwise = False):
    # process a given iterable of image_files, finding and storing the "average" color of the images in 
    # the given database table
    rows = []
    image_data = db['image']

    pixelwise_sampler = PixelwiseSampler(sample_size) if pixelwise else None

    if pixelwise:
        sample_info = db['sample_info']  # TODO: also store "length/width"?
        info = sample_info.find_one(id = 0)
        if info:
            pixelwise_sampler.pts_to_sample = info['pixels']
        else:
            sample_info.insert({ 'id': 0, 'pixels': json.dumps(pixelwise_sampler.pts_to_sample), 'sample_size': sample_size })

    num_files = len(image_files)
    print("{} files to process".format(num_files))

    for file_num, filename in enumerate(image_files):

        path = os.path.abspath(unicode(filename, sys.getfilesystemencoding()))

        original_row = image_data.find_one(path = path)
        if original_row: # only do all the calculations if we haven't already checked this file
            if original_row.get('r') is not None and not pixelwise:
                continue
            if original_row.get('pixels') is not None and pixelwise:
                continue
            update = True

        new_row = process_image(path, not pixelwise, pixelwise, pixelwise_sampler)

        if not new_row:
            print(path + " couldn't be processed")
            continue

        sys.stdout.write("processed image number {}\r".format(file_num + 1))
        sys.stdout.flush()
        
        if 'pixels' in new_row:
            new_row['pixels'] = json.dumps(new_row['pixels'])  # sqlite can't store lists - maybe switch backends? Or just go all out and json or pickle everything
        
        if original_row:
            new_row.update(original_row)
            image_data.update(new_row, ['path'])
        else:
            rows.append(new_row)
    
    print("")
    print("Inserting rows into db...")
    image_data.insert_many(rows)  # might be better to do more often than just once at the end so that interruptions don't ruin everything during a long preprocessing period

def process_image(path, rgb_needed, pixels_needed, pixelwise_sampler = None):
    try:
        image = make_proportional(Image.open(path).convert(mode = "RGB"))
    except (IOError, struct.error):
        # IOError is usually because the file isn't an image file.
        # TODO: look into what's causing struct.error
        # TODO: check for struct.error elsewhere?
        return None
    
    d = dict(path = path)
    if pixels_needed:
       d['pixels'] = pixelwise_sampler.sample_image(image)
    
    if rgb_needed:
        avg = get_color_avg(get_n_pixels(image, 300))
        d.update(avg)

    return d


def main(argv):
    # Process command line args. Depending on the result, either make a new photomosaic and display or save it (mosaic mode), or simply
    # preprocess images and save them to a database file (preprocessing mode)

    arg_parser = argparse.ArgumentParser()
 
    # options/argument for mosaic mode only
    arg_parser.add_argument('-j', '--json', metavar = 'FILE', help = 'produce a json file FILE that shows which images were used where in the mosaic')
    arg_parser.add_argument('-n', '--number', dest = 'n', type = int, default = 500, help = 'the mosaic should consist of about this many pieces')
    arg_parser.add_argument('-o', '--outfile', metavar = 'FILE', help = 'save the mosaic as FILE; if not specified, the mosaic will be opened in your default image viewer, but not saved; format of output file is determined by extension (so .jpg for jpg, .png for png)')
    arg_parser.add_argument('-r', '--randomize', action = 'store_true', help = 'randomize the order in which pieces get added to the mosaic')
    arg_parser.add_argument('-s', '--sourceimages', metavar = 'IMAGE', nargs = '+', help = 'draw from this set of images in constructing mosaic')
    arg_parser.add_argument('-u', '--unique', action = 'store_true', help = 'don\'t use any image as a piece of the mosaic more than once')
    arg_parser.add_argument('-x', '--nooutput', action = 'store_true', help = 'don\'t show mosaic file after it\'s built')
    arg_parser.add_argument('-z', '--piecesize', type = int, default = 100, help = 'each square piece of the resulting image will have edges this many pixels long; increase this value if the individual images are too pixelated and hard to make out, even when zoomed')

    arg_parser.add_argument('filename', nargs = '?', help = 'the filename of the image we\'re making a mosaic of')

    # option to turn on preprocessing mode
    arg_parser.add_argument('-p', '--preprocess', metavar = 'IMAGE', nargs = '+', help = 'switch to preprocessing mode; preprocess specified image file[s], adding to the pool of potential images in our database; options and arguments other than -g and -d will be ignored')

    # options usable in both mosaic mode and preprocessing mode
    arg_parser.add_argument('-d', '--dbfile', help = 'in mosaic mode, construct mosaic using images pointed to by this database file; in preprocessing mode, save the data to this file')
    arg_parser.add_argument('-w', '--pixelwise', action = 'store_true', help = 'compare images pixel by pixel instead of just average color')

    args = arg_parser.parse_args()
    
    dbtable = None
    sample_info = None

    if args.dbfile:
        dbname = 'sqlite:///' + args.dbfile

        db = dataset.connect(dbname)
        dbtable = db['image']
        sample_table = db['sample_info']
        sample_info = sample_table.find_one(id = 0)

    if args.preprocess:
        store_image_data(args.preprocess, db, sample_size = 300, pixelwise = args.pixelwise)

    elif args.filename:

        if args.outfile and os.path.exists(args.outfile):
            print
            if not (raw_input("Overwrite existing file {}? (y/n) ".format(args.outfile)) in ["yes", "y"]):  # TODO: raw_input won't work in python3
                sys.exit(0) 
        
        print("Making mosaic out of {}...".format(args.filename))
        
        # switch to make_mosaigraph2 and collect_candidates2 to attempt the alternative strategy - using my own wrapper object for images, and using alternative "uniqueness" strategy

        sampler = PixelwiseSampler(300)
        if sample_info:
            print("using pixels {}".format(sample_info['pixels']))
            sampler.pts_to_sample = json.loads(sample_info['pixels'])

        candidates = list(collect_candidates(dbtable, args.sourceimages, not args.pixelwise, args.pixelwise, sampler))
        
        print("Using pool of {} candidate images".format(len(candidates)))
        
        input_image = Image.open(args.filename)

        # TODO: should we pass width/height hints (i.e., edge_length_orig x edge_length_orig) to get_matcher?
        matcher = get_matcher(args.pixelwise, args.unique, candidates, 300, sampler = sampler)

        i, dict = make_mosaigraph(input_image, args.n, matcher, edge_length = args.piecesize, randomize_order = args.randomize, unique = args.unique)
       
        print("New mosaic produced with average pixelwise difference {}".format(sampler.compare(input_image, i)))
        #         #print("New mosaic produced with average pixelwise difference {}".format(compare_pixelwise(input_image, i, n = 300) / 1200))  TODO: REMOVE LINE
        if args.outfile:
            print("Saving mosaic as {}".format(args.outfile))
            i.save(args.outfile)
        else:
            if not args.nooutput:
                print "Showing output"
                i.show()
        if args.json:
            print("Saving json output as {}".format(args.json))
            with open(args.json, 'w') as f:
                json.dump(dict, f)

    else:
        print("Nothing to do!\n")
        arg_parser.print_help()

if __name__ == "__main__":
    main(sys.argv)

