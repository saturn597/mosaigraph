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


def make_mosaigraph(img, numPieces, dbtable, group = None, directory = None, newEdgeSize = 100, unique = False, randomize_order = False):
    # makes a photomosaic, returning a PIL.Image representing it, which can then be displayed, saved, etc, and a dict describing the images used and where

    # TODO: subjective observation, randomize_order seems to make it slower. Is this correct?

    # TODO: what if we get group AND directory?
    if group:
        candidates = list(dbtable.find(group = group))
    elif directory:
        process_images(get_images(directory))  # could conceivably do this as we're looking for matches (so we don't add an extra pass through all the images)
        candidates = list(dbtable.find(directory = directory))
    else:
        candidates = list(dbtable.all())

    if not unique:
      # k-d trees are an efficient data structure for nearest neighbor search. However, it's difficult to exclude 
      # items from the search that we've already used. (At least given scipy's implementation). So 
      # for now only using them when we don't mind reusing images.
      kdtree = spatial.KDTree([(item['r'], item['g'], item['b']) for item in candidates])

    width, height = img.size
        
    edgeSize = get_best_edge_size(width, height, numPieces)
    
    numX = width // edgeSize
    numY = height // edgeSize
    numPieces = numX * numY

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
        rgb = get_color_avg(get_n_pixels(oldPiece, 300))

        if unique:
            # if unique, we don't worry about caching, but have to avoid reusing images, and not
            # using k-d trees for now (explained above).
            match_index = find_match_linear(rgb, candidates)
            path = candidates[match_index]['path']
            addition = prep_image(path)
            candidates.pop(match_index)
        else:
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
        distance = math.sqrt((image['r'] - rgb['r'])**2 + (image['g'] - rgb['g'])**2 + (image['b'] - rgb['b'])**2)
        if distance < least_distance:
            closest_index = n
            least_distance = distance
    
    return closest_index

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

def get_images(d):
    # return a list of image filenames in directory d

    isimg = re.compile('\.(png|jpg)$')
    return [os.path.join(d, f) for f in os.listdir(d) if re.search(isimg, f)]

def process_images(image_files, dbtable, group = None):
    # process a given iterable of image_files, finding and storing the "average" color of the images in 
    # the given database table, and cataloging as belonging to the given "group" (a string describing the
    # category into which the image falls)
    num = 0

    for filename in image_files:
        path = unicode(filename, sys.getfilesystemencoding())

        print path
        if not dbtable.find_one(path = path):  # only do all the calculations if we haven't already checked this file
            try: 
                image = make_proportional(Image.open(path).convert(mode = "RGB"))
            except (IOError, struct.error):
                # IOError is usually because the file isn't an image file.
                # TODO: look into what's causing struct.error
                print(filename + " couldn't be processed")
                continue
            avg = get_color_avg(get_n_pixels(image, 300))
            num += 1
            print ("processed image number " + str(num) + " : " + path + " : " + str(avg))

            # build a new row to add to the db
            new_row = dict(path = path, r = avg['r'], g = avg['g'], b = avg['b'], directory = os.path.dirname(path))
            if group != None:  # using None as a value in an insert raises warnings, avoiding it
                new_row['group'] = str(group)

            dbtable.insert(new_row)


def main(argv):
    # Process command line args. Depending on the result, either make a new photomosaic and display or save it (mosaic mode), or simply
    # preprocess images and save them to a database file (preprocessing mode)
    print 'processing args'

    arg_parser = argparse.ArgumentParser()
 
    # options/argument for mosaic mode only
    arg_parser.add_argument('-j', '--json', metavar = 'FILE', help = 'produce a json file FILE that shows which images were used where in the mosaic')
    arg_parser.add_argument('-n', '--number', dest = 'n', type = int, default = 500, help = 'the mosaic should consist of about this many pieces')
    arg_parser.add_argument('-o', '--outfile', metavar = 'FILE', help = 'save the mosaic as FILE')
    arg_parser.add_argument('-r', '--randomize', action = 'store_true', help = 'randomize the order in which pieces get added to the mosaic')
    arg_parser.add_argument('-s', '--sourcedirectory', dest = 'source_directory', metavar = 'DIRECTORY', help = 'use images from specified directory as "pieces" of the mosaic')
    arg_parser.add_argument('-u', '--unique', action = 'store_true', help = 'don\'t use any image as a piece of the mosaic more than once')
    arg_parser.add_argument('-x', '--nooutput', action = 'store_true', help = 'don\'t show mosaic file after it\'s built')
    arg_parser.add_argument('-z', '--piecesize', type = int, default = 100, help = 'pieces of the mosaic will have edges this many pixels long')

    arg_parser.add_argument('filename', nargs = '?', help = 'the filename of the image we\'re making a mosaic of; the mosaic will look like this image')

    # option to turn on preprocessing mode
    arg_parser.add_argument('-p', '--preprocess', metavar = 'IMAGE', nargs = '+', help = 'switch to preprocessing mode; preprocess specified image file[s], adding to the pool of potential images in our database; options and arguments other than -g and -d will be ignored')

    # options usable in both mosaic mode and preprocessing mode
    arg_parser.add_argument('-d', '--dbfile', default = 'data.db', help = 'in mosaic mode, use images pointed to by this database file; in preprocessing mode, save the data to this file')
    arg_parser.add_argument('-g', '--group', help = 'in mosaic mode, use only images from a previously defined group as "pieces" of the mosaic; in preprocessing mode, adds images we preprocess to the specified group')

    args = arg_parser.parse_args()
    
    print 'connecting to db'
    dbname = 'sqlite:///' + args.dbfile
    db = dataset.connect(dbname)
    dbtable = db['image']

    if args.preprocess:
        process_images(args.preprocess, dbtable, args.group)
    elif args.filename:

        if args.outfile and os.path.exists(args.outfile):
            print("Overwrite existing file {}? y/n".format(args.outfile))
            if not (raw_input() in ["yes", "y"]):
                sys.exit(0) 

        print 'making mosaigraph'
        i, dict = make_mosaigraph(Image.open(args.filename), args.n, dbtable, group = args.group, directory = args.source_directory, unique = args.unique, newEdgeSize = args.piecesize, randomize_order = args.randomize)
        if args.outfile:
            i.save(args.outfile)
        else:
            if not args.nooutput:
              print "showing output"
              i.show()
        if args.json:
            with open(args.json, 'w') as f:
                json.dump(dict, f)
    else:
        print("Nothing to do!\n")
        arg_parser.print_help()

if __name__ == "__main__":
    main(sys.argv)

