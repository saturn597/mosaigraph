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

image_cache = {}
comparison_points = []

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

def collect_candidates(dbtable, paths = [], rgb_needed = False):

    candidates = []

    if not paths:
        return list(dbtable.all())

    for path in paths:
        path = unicode(path, sys.getfilesystemencoding()) # TODO: Does this need a unicode conversion?
        full_path = os.path.abspath(path)  
        new_candidate = dict(path = full_path)

        if rgb_needed:
            result = preprocess_image(full_path)
            print "rgb gotten"
            if result:
                new_candidate = result
            else:
                continue

        candidates.append(new_candidate)
    print candidates
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


def get_piece(image, edgeSize, x, y):
    new_image = image.crop((x * edgeSize, y * edgeSize, (x + 1) * edgeSize, (y + 1) * edgeSize))  
    return ImageData(image = new_image, x = x, y = y)

def make_mosaigraph2(img, candidates, numPieces, newEdgeSize = 100, unique = False, randomize_order = False, pixelwise = False):
    # makes a photomosaic, returning a PIL.Image representing it, which can then be displayed, saved, etc, and a dict describing the images used and where

    # TODO: subjective observation, randomize_order seems to make it slower. Is this correct?
    # TODO: replace incoming img with ImageData

    width, height = img.size
        
    edgeSize = get_best_edge_size(width, height, numPieces)
    numX = width // edgeSize
    numY = height // edgeSize
    numPieces = numX * numY
    print("Dividing mosaic into {} pieces".format(numPieces))

    newImg = Image.new(img.mode, (width // edgeSize * newEdgeSize, height // edgeSize * newEdgeSize))

    currentNum = 0

    the_pieces = [get_piece(img, edgeSize, x, y) for x in range(0, numX) for y in range(0, numY)]
         
    # if we have a limited pool of images to pick from, pieces of the mosaic added later are likely to look
    # worse than ones added earlier, as we run out of good matches. Offering option to randomize the order in 
    # which we add pieces to the mosaic so that one region doesn't look better/worse than another. Not 
    # confident this is necessarily better - seems to lead to an image that's equally bad all over. 
    if randomize_order:
        random.shuffle(the_pieces)
   
    paths = []
    passnum = 0

    if unique:
        undone_pieces = [piece for piece in the_pieces if not piece.match]
        while undone_pieces:
            passnum += 1
            print "pass {}".format(len(undone_pieces))
            get_matches(candidates, undone_pieces, pixelwise, unique)
            matches = set([piece.match for piece in the_pieces if piece.match])
            undone_pieces = [piece for piece in the_pieces if not piece.match]
            candidates = [candidate for candidate in candidates if candidate not in matches]
    else:
        get_matches(the_pieces, candidates, pixelwise, unique)

    def prep_image(image):
        proportional = make_proportional(image.get_image())
        return proportional.resize((newEdgeSize, newEdgeSize))
    
    print the_pieces
    for piece in the_pieces:
        new_piece = piece.match
        try: 
            newImg.paste(prep_image(new_piece), (piece.x * newEdgeSize, piece.y * newEdgeSize))
        except:
            continue
        paths.append({ 'x': piece.x, 'y': piece.y, 'path': new_piece.path })
        currentNum += 1  # maybe use enumerate in for instead
        sys.stdout.write("\rCompleted piece " + str(currentNum) + " out of " + str(numPieces))
        sys.stdout.flush()
    print("")

    return newImg, paths


def make_mosaigraph(img, candidates, numPieces, newEdgeSize = 100, unique = False, randomize_order = False, pixelwise = False):
    # makes a photomosaic, returning a PIL.Image representing it, which can then be displayed, saved, etc, and a dict describing the images used and where

    # TODO: subjective observation, randomize_order seems to make it slower. Is this correct?

    if not unique and not pixelwise:
      # k-d trees are an efficient data structure for nearest neighbor search. However, it's difficult to exclude 
      # items from the search that we've already used. (At least given scipy's implementation). So 
      # for now only using them when we don't mind reusing images.
      kdtree = spatial.KDTree([(item['r'], item['g'], item['b']) for item in candidates])

    width, height = img.size
        
    edgeSize = get_best_edge_size(width, height, numPieces)
    numX = width // edgeSize
    numY = height // edgeSize
    numPieces = numX * numY
    print("Dividing mosaic into {} pieces".format(numPieces))

    newImg = Image.new(img.mode, (width // edgeSize * newEdgeSize, height // edgeSize * newEdgeSize))

    currentNum = 0

    the_pieces = [(x, y) for x in range(0, numX) for y in range(0, numY)]

    # if we have a limited pool of images to pick from, pieces of the mosaic added later are likely to look
    # worse than ones added earlier, as we run out of good matches. Offering option to randomize the order in 
    # which we add pieces to the mosaic so that one region doesn't look better/worse than another. Not 
    # confident this is necessarily better - seems to lead to an image that's equally bad all over. 
    if randomize_order:
        random.shuffle(the_pieces)

    loaded_images = {}  # loading images is our most expensive operation; cache them so we don't need to load 2x
    
    def prep_image(path):
        new_piece = Image.open(path)
        proportional = make_proportional(new_piece)
        return proportional.resize((newEdgeSize, newEdgeSize))

    paths = []
    for x, y in the_pieces:
        oldPiece = img.crop((x * edgeSize, y * edgeSize, (x + 1) * edgeSize, (y + 1) * edgeSize))  

        # currently, integration of the "pixelwise" option is ugly
        if pixelwise:
            
            match_index = find_match_deep(oldPiece, candidates, compare_pixelwise2)
            #match_index = find_match_deep_orig(oldPiece, candidates, compare_pixelwise)

        rgb = get_color_avg(get_n_pixels(oldPiece, 300))

        if unique:
            # if unique, we don't worry about caching, but have to avoid reusing images. Also, we're not
            # using k-d trees for now (explained above).
            if not pixelwise: 
                match_index = find_match_linear(rgb, candidates)
            path = candidates[match_index]['path']
            addition = prep_image(path)
            candidates.pop(match_index)

        else:
            if not pixelwise:
                match_index = kdtree.query((rgb['r'], rgb['g'], rgb['b']))[1]
            path = candidates[match_index]['path']
            if path in loaded_images:
                addition = loaded_images[path]
            else:
                addition = prep_image(path)
                loaded_images[path] = addition  # if we must open the image, add it to our cache
        
        paths.append( {'x': x, 'y': y, 'path': path } )
        newImg.paste(addition, (x * newEdgeSize, y * newEdgeSize))
        currentNum += 1
        sys.stdout.write("\rCompleted piece " + str(currentNum) + " out of " + str(numPieces))
        sys.stdout.flush()
    print("")
    return newImg, paths

def find_match_linear(rgb, candidates):
    # given an iterable of candidate rgb values, find the one that most closely matches "rgb"
    # returns the index of the match within candidates 

    closest_image = None
    least_distance = math.sqrt(255**2 + 255 ** 2 + 255 ** 2)
    for n, image in enumerate(candidates):
        distance = math.sqrt((image['r'] - rgb['r'])**2 + (image['g'] - rgb['g'])**2 + (image['b'] - rgb['b'])**2)  # get rid of this sqrt
        if distance < least_distance:
            closest_index = n
            least_distance = distance
    
    return closest_index

def find_match_linear2(rgb, candidates):
    # given an iterable of candidate rgb values, find the one that most closely matches "rgb"
    # returns the index of the match within candidates 

    closest_image = None
    least_distance = math.sqrt(255**2 + 255 ** 2 + 255 ** 2)
    for n, candidate in enumerate(candidates):
        rgb2 = candidate.get_rgb()
        distance = math.sqrt((rgb[0] - rgb2[0])**2 + (rgb[1] - rgb2[1])**2 + (rgb[2] - rgb2[2])**2)  # get rid of this sqrt
        if distance < least_distance:
            closest_index = n
            least_distance = distance
    
    return least_distance, closest_index


def find_match_deep(original, candidates, comparison_fn):
    # maybe combine with find_match_linear
    closest_image = None
    least_distance = None

    for n, distance in enumerate(comparison_fn(original, candidates)):
        if not least_distance or distance < least_distance:
            closest_index = n
            least_distance = distance

    return closest_index

def find_match_deep_orig(original, candidates, comparison_fn):
      # maybe combine with find_match_linear
      closest_image = None
      least_distance = None

      for n, candidate in enumerate(candidates):
          distance = comparison_fn(original, candidate['path'])
          if not least_distance or distance < least_distance:
                closest_index = n
                least_distance = distance

      return closest_index

def compare_pixelwise2(img, candidates, n = 300):
    width, height = img.size

    global comparison_points  # Need these points to be consistent between calls, or data from the image_cache will be bogus
    if not comparison_points:
        comparison_points = [(random.randint(0, width - 1), random.randint(0, height - 1)) for i in range(n)]

    pixels = [img.getpixel((x, y)) for x, y in comparison_points ]

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
            pixels2 = [img2.getpixel((x, y)) for x, y in comparison_points ]
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

def get_best_edge_size(width, height, numPieces):
    # If we have a rectangle of width and height, and we want to divide it into numPieces squares of equal size
    # how long should each square's sides be to get as close as possible to numPieces?
 
    # get an initial estimate
    # maybe would be simpler to just start with an estimate of 1, then iterate up
    currentEstimate = int(math.sqrt(width * height / numPieces))

    # get the difference between the number of pieces this gives and the number we want
    signedDiff = (width // currentEstimate) * (height // currentEstimate) - numPieces
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
        currentDiff = abs((width // currentEstimate) * (height // currentEstimate) - numPieces)
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

def process_images(image_files, dbtable):
    # process a given iterable of image_files, finding and storing the "average" color of the images in 
    # the given database table
    num = 0
    rows = []

    for filename in image_files:
        path = os.path.abspath(unicode(filename, sys.getfilesystemencoding()))

        if dbtable.find_one(path = path): # only do all the calculations if we haven't already checked this file
            return

        new_row = preprocess_image(path)

        if not new_row:
            print(path + " couldn't be processed")
            continue

        num += 1
        print ("processed image number " + str(num) + " : " + path + " : " + "r: {} g: {} b: {}".format(new_row['r'], new_row['g'], new_row['b']))
        rows.append(new_row)

    print("Inserting rows into db...")
    dbtable.insert_many(rows)  # might be better to do more often than just once at the end so that interruptions don't ruin everything during a long preprocessing period

def preprocess_image(path):

    try:
        image = make_proportional(Image.open(path).convert(mode = "RGB"))
    except (IOError, struct.error):
        # IOError is usually because the file isn't an image file.
        # TODO: look into what's causing struct.error
        return None

    avg = get_color_avg(get_n_pixels(image, 300))

    return dict(path = path, r = avg['r'], g = avg['g'], b = avg['b'])


def main(argv):
    # Process command line args. Depending on the result, either make a new photomosaic and display or save it (mosaic mode), or simply
    # preprocess images and save them to a database file (preprocessing mode)

    arg_parser = argparse.ArgumentParser()
 
    # options/argument for mosaic mode only
    arg_parser.add_argument('-j', '--json', metavar = 'FILE', help = 'produce a json file FILE that shows which images were used where in the mosaic')
    arg_parser.add_argument('-n', '--number', dest = 'n', type = int, default = 500, help = 'the mosaic should consist of about this many pieces')
    arg_parser.add_argument('-o', '--outfile', metavar = 'FILE', help = 'save the mosaic as FILE; if not specified, the mosaic will be opened in your default image viewer, but not saved; format of output file is determined by extension (so .jpg for jpg, .png for png)')
    arg_parser.add_argument('-r', '--randomize', action = 'store_true', help = 'randomize the order in which pieces get added to the mosaic')
    arg_parser.add_argument('-s', '--sourceimages', metavar = 'IMAGE', nargs = '+', help = 'construct mosaic drawing from this set of images')
    arg_parser.add_argument('-u', '--unique', action = 'store_true', help = 'don\'t use any image as a piece of the mosaic more than once')
    arg_parser.add_argument('-w', '--pixelwise', action = 'store_true', help = 'compare images pixel by pixel instead of just average color')
    arg_parser.add_argument('-x', '--nooutput', action = 'store_true', help = 'don\'t show mosaic file after it\'s built')
    arg_parser.add_argument('-z', '--piecesize', type = int, default = 100, help = 'each square piece of the resulting image will have edges this many pixels long; increase this value if the individual images are too pixelated and hard to make out, even when zoomed')

    arg_parser.add_argument('filename', nargs = '?', help = 'the filename of the image we\'re making a mosaic of')

    # option to turn on preprocessing mode
    arg_parser.add_argument('-p', '--preprocess', metavar = 'IMAGE', nargs = '+', help = 'switch to preprocessing mode; preprocess specified image file[s], adding to the pool of potential images in our database; options and arguments other than -g and -d will be ignored')

    # options usable in both mosaic mode and preprocessing mode
    arg_parser.add_argument('-d', '--dbfile', default = 'data.db', help = 'in mosaic mode, construct mosaic using images pointed to by this database file; in preprocessing mode, save the data to this file')

    args = arg_parser.parse_args()
    
    dbname = 'sqlite:///' + args.dbfile
    db = dataset.connect(dbname)
    dbtable = db['image']

    if args.preprocess:
        process_images(args.preprocess, dbtable)
    elif args.filename:

        if args.outfile and os.path.exists(args.outfile):
            print
            if not (raw_input("Overwrite existing file {}? (y/n) ".format(args.outfile)) in ["yes", "y"]):  # TODO: raw_input won't work in python3
                sys.exit(0) 
        
        print("Making mosaic out of {}...".format(args.filename))
        
        candidates = list(collect_candidates(dbtable, paths = args.sourceimages, rgb_needed = not args.pixelwise))
        
        print("Using pool of {} candidate images".format(len(candidates)))
        
        input_image = Image.open(args.filename)
        i, dict = make_mosaigraph(input_image, candidates, args.n, unique = args.unique, newEdgeSize = args.piecesize, randomize_order = args.randomize, pixelwise = args.pixelwise)
        
        print("New mosaic produced with average pixelwise difference {}".format(compare_pixelwise(input_image, i, n = 300) / 1200))

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

