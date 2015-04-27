
from __future__ import division

from PIL import Image

import dataset
import getopt
import math
import random
import os
import re
import sys

imagePath = './images'
dbname = 'sqlite:///data.db'
db = dataset.connect(dbname)
dbtable = db['image']

# TODO: Fix usage string
usage_string = """
Usage: 
-c: specify a directory to catalog
-g: specify a "group" of images
"""

def getColorAvg(pixels):
    r, g, b = 0.0, 0.0, 0.0
    numPixels = 0

    for pix in pixels:
        r += pix[0]
        g += pix[1]
        b += pix[2]
        numPixels += 1
    
    r = r / numPixels
    g = g / numPixels
    b = b / numPixels

    return (int(r), int(g), int(b))

def getAllPixels(img):
    width, height = img.size
    for x in range(0, width):
        for y in range(0, height):
            yield img.getpixel((x, y))

def getSomePixels(img, n):
    # return iterator through n randomly selected pixels in an image
    width, height = img.size
    for i in range(0, n):
        x = random.randint(0, width - 1)  # is subtracting 1 correct?
        y = random.randint(0, height - 1) 
        yield img.getpixel((x, y))
         
def getImages(d):
    # return a list of image filenames in directory d
    isimg = re.compile('\.(png|jpg)$')
    return [f for f in os.listdir(d) if re.search(isimg, f)]

def divideRect(width, height, pieceWidth, pieceHeight):
    # take rectangle of width and height, and divide it into rectangles of pieceWidth x pieceHeight
    numX = width // pieceWidth
    numY = height // pieceHeight
    divisionsX = [ x * pieceWidth for x in range(0, numX) ]
    divisionsY = [ y * pieceHeight for y in range(0, numY) ]
    return [(x, y) for x in divisionsX for y in divisionsY] 

def divideImage(img, pieceWidth, pieceHeight):
    width, height = img.size
    pieces = divideRect(width, height, pieceWidth, pieceHeight)
    pieceData = []
    for piece in pieces:
        cropped = img.crop((piece[0], piece[1], piece[0] + pieceWidth, piece[1] + pieceHeight))  
        # Do I need to subtract one from pieceWidth and pieceHeight? Might be needed to ensure that the pieces don't overlap
        pieceData.append({ 'x': piece[0], 'y': piece[1], 'image': cropped })
    return pieceData

def divideImgXPieces(img, numPieces):
    width, height = img.size
    edgeSize = int(math.sqrt(numPieces / (width * height)))
    numX = width // edgeSize
    numY = height // edgeSize

    for x in range(0, numX):
        for y in range(0, numY):
            xcoord = x * edgeSize
            ycoord = y * edgeSize
            image = img.crop(xcoord, ycoord)
            yield { 'x': xcoord, 'y': ycoord, 'image': image }

def findBestEdgeSize(width, height, numPieces):
    
    # get an initial estimate
    # maybe would be simpler to start with an estimate of 1, then iterate up
    currentEstimate = int(math.sqrt(width * height / float(numPieces)))

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

def makeMosaigraph(img, numPieces, group = None, directory = None, newEdgeSize = 300, unique = False):
    
    # what if we get group AND directory?
    if group:
        candidates = list(dbtable.find(group = group))
    elif directory:
        processImages(directory)  # could conceivably do this as we're looking for matches (so we don't add an extra pass through all the images)
        candidates = list(dbtable.find(directory = directory))
    else:
        candidates = list(dbtable.all())
        
    width, height = img.size
        
    edgeSize = findBestEdgeSize(width, height, numPieces)
    
    numX = width // edgeSize
    numY = height // edgeSize
    numPieces = numX * numY

    newImg = Image.new(img.mode, (width // edgeSize * newEdgeSize, height // edgeSize * newEdgeSize))

    currentNum = 0
    for x in range(0, numX):
        for y in range(0, numY):
            oldPiece = img.crop((x * edgeSize, y * edgeSize, (x + 1) * edgeSize, (y + 1) * edgeSize))  
            rgb = getColorAvg(getSomePixels(oldPiece, 300))
            closest_index = findMatch(rgb, candidates)
            path = candidates[closest_index]['path']
            if unique:
                candidates.pop(closest_index)
            addOn = makeProportional(Image.open(path)).resize((newEdgeSize, newEdgeSize))
            newImg.paste(addOn, (x * newEdgeSize, y * newEdgeSize))
            currentNum += 1
            print("Completed piece " + str(currentNum) + " out of " + str(numPieces))
            print(path)
    
    """ 
    pieces = divideImage(img, edgeSize, edgeSize)
    length = len(pieces)
    newImg = Image.new(img.mode, (width / edgeSize * newEdgeSize, height / edgeSize * newEdgeSize))
    currentNum = 0
    
    for piece in pieces:
        rgb = getColorAvg(getSomePixels(piece['image'], 300))
        #addOn = Image.new('RGB', (pieceWidth, pieceHeight), (rgb)) #this line would just paste the exact color in
        closest_index = findMatch(rgb, candidates)
        path = candidates[closest_index]['path']
        if unique:
            candidates.pop(closest_index)
        addOn = makeProportional(Image.open(path)).resize((newEdgeSize, newEdgeSize))
        newImg.paste(addOn, (piece['x'] / edgeSize * newEdgeSize, piece['y'] / edgeSize * newEdgeSize))
        currentNum += 1
        print "Currently on piece " + str(currentNum) + " out of " + str(length)
        print path
    """
    return newImg

def findMatch(rgb, candidates):

    closest_image = None
    least_distance = math.sqrt(255**2 + 255 ** 2 + 255 ** 2)
    for n, image in enumerate(candidates):
        distance = math.sqrt((image['r'] - rgb[0])**2 + (image['g'] - rgb[1])**2 + (image['b'] - rgb[2])**2)
        if distance < least_distance:
            closest_index = n
            least_distance = distance
    
    return closest_index

#toShow.paste(cropped, (0, 0))  # for some reason passing None as the box results in an error ("images do not match") - docs say these are the same

def makeProportional(img, ratio = 1):
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


def processImages(directory, group = None):
    for filename in getImages(directory):
        processImage(os.path.join(directory, filename), group)

def processImage(path, group):
    # maybe combine with processImages
    if not dbtable.find_one(path = path):  # only do all the calculations if we haven't already checked this file
        image = makeProportional(Image.open(path))
        avg = getColorAvg(getSomePixels(image, 300))
        print (path + " : " + str(avg))
        dbtable.insert(dict(path = fullPath, r = avg[0], g = avg[1], b = avg[2], group = group, directory = os.path.dirname(path)))

def main(argv):
    directoryToProcess = None
    group = None
    imagePath = None
    n = 100
    outfile = None
    sourceDirectory = None
    unique = False

    try:
        opts, args = getopt.getopt(argv, "g:n:o:p:s:u", ["group=", "number=", "output=", "preprocess=", "sourcedirectory=", "unique"])
    except getopt.GetoptError:
        print(usage_string)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-g", "--group"):
            group = arg
        elif opt in ("-n", "--number"):
            n = int(arg)
        elif opt in ("-o", "--output"):
            outfile = arg
        elif opt in ("-p", "--preprocess"):
            directoryToProcess = os.path.abspath(arg)
        elif opt in ("-s", "--sourcedirectory"):
            sourceDirectory = os.path.abspath(arg)
        elif opt in ("-u", "--unique"):
            unique = True
    

    if directoryToProcess:
        processImages(directoryToProcess, group)
    elif len(args) == 1:
        imagePath = args[0]
        i = makeMosaigraph(Image.open(imagePath), n, group = group, directory = sourceDirectory, unique = unique)
        if outfile:
            i.save(outfile)
        else:
            i.show()
    else:
        print(usage_string)
        sys.exit(2)

if __name__ == "__main__":
    main(sys.argv[1:])



"""
img = Image.open('./images/test3.jpg')
makeMosaigraph(img, 10, 10).show()
makeMosaigraph(img, 50, 50).show()
makeMosaigraph(img, 100, 100).show()
makeMosaigraph(img, 300, 300).show()
"""
