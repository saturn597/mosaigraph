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

usage_string = """
Usage: 
-c: specify a directory to catalog
-g: specify a "group" of images
"""

def getColorAvg(img):
    # returns the average color of all pixels in an rgb image
    width, height = img.size

    r, g, b = 0.0, 0.0, 0.0
    for x in xrange(0, width):
        for y in xrange(0, height):
            pix = im.getpixel((x, y))
            r += pix[0]
            g += pix[1]
            b += pix[2]
           
    numPixels = width * height
    
    r = r / numPixels
    g = g / numPixels
    b = b / numPixels

    return (int(r), int(g), int(b))

def getColorAvg2(pixels):
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
    for x in xrange(0, width):
        for y in xrange(0, height):
            yield img.getpixel((x, y))

def getSomePixels(img, n):
    width, height = img.size
    for i in xrange(0, n):
        x = random.randint(0, width - 1)  # is subtracting 1 correct?
        y = random.randint(0, height - 1) 
        yield img.getpixel((x, y))
         
    

def getImages(d):
    # return a list of image filenames in directory d
    isimg = re.compile('\.(png|jpg)$')
    return [f for f in os.listdir(d) if re.search(isimg, f)]

def divideRect(width, height, pieceWidth, pieceHeight):
    # take rectangle of width and height, and divide it into rectangles of pieceWidth x pieceHeight
    numX = width / pieceWidth
    numY = height / pieceHeight
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
    numX = width / edgeSize
    numY = height / edgeSize

    for x in range(0, numX):
        for y in range(0, numY):
            xcoord = x * edgeSize
            ycoord = y * edgeSize
            image = img.crop(xcoord, ycoord)
            yield { 'x': xcoord, 'y': ycoord, 'image': image }

def makeMosaigraph2(img, numPieces, group, newPieceSize = 100):
    width, height = img.size
    edgeSize = int(1 / math.sqrt(float(numPieces) / (width * height)))
    scale = newPieceSize / edgeSize 

    pieces = divideImage(img, edgeSize, edgeSize)
    length = len(pieces)

    newImg = Image.new(img.mode, (width * scale, height * scale))
    currentNum = 0

    newPieceSize = edgeSize * scale

    for piece in pieces:
        rgb = getColorAvg2(getSomePixels(piece['image'], 1))
        #addOn = Image.new('RGB', (pieceWidth, pieceHeight), (rgb)) #this line would just paste the exact color in
        path = findMatch(rgb, group)
        print path
        addOn = makeProportional(Image.open(path)).resize((newPieceSize, newPieceSize))
        newImg.paste(addOn, (piece['x'] * scale, piece['y'] * scale))
        currentNum += 1
        print "Currently on piece " + str(currentNum) + " out of " + str(length)

    return newImg
     

def makeMosaigraph(img, horizDivs, vertDivs, group, minPieceSize = 100):
    width, height = img.size

    pieceWidth = width / horizDivs
    pieceHeight = height / vertDivs

    if pieceWidth > pieceHeight:
        pieceWidth = pieceHeight
    else:
        pieceHeight = pieceWidth

    scale = 1

    if pieceWidth < minPieceSize:
        scale = minPieceSize / pieceWidth
    if pieceHeight * scale < minPieceSize:
        scale = minPieceSize / pieceHeight

    newImg = Image.new(img.mode, (width * scale, height * scale))

    pieces = divideImage(img, pieceWidth, pieceHeight)
    length = len(pieces)
    currentNum = 0
    for piece in pieces:
        rgb = getColorAvg2(getSomePixels(piece['image'], 1))
        #addOn = Image.new('RGB', (pieceWidth, pieceHeight), (rgb)) #this line would just paste the exact color in
        path = findMatch(rgb, group)
        print path
        addOn = makeProportional(Image.open(path)).resize((pieceWidth * scale, pieceHeight * scale))
        newImg.paste(addOn, (piece['x'] * scale, piece['y'] * scale))
        currentNum += 1
        print "Currently on piece " + str(currentNum) + " out of " + str(length)
    return newImg
 
def findMatch(rgb, group):
    path_to_closest = ""
    least_distance = math.sqrt(255**2 + 255 ** 2 + 255 ** 2)
    for image in dbtable.find(group = group):
        distance = math.sqrt((image['r'] - rgb[0])**2 + (image['g'] - rgb[1])**2 + (image['b'] - rgb[2])**2)
        if distance < least_distance:
            path_to_closest = image['path']
            least_distance = distance
    
    return path_to_closest

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


def catalogImages(imagePath, group):
    for filename in getImages(imagePath):
        fullPath = os.path.join(imagePath, filename)
        if not dbtable.find_one(path = fullPath):  # only do all the calculations if we haven't already checked this file
            image = makeProportional(Image.open(fullPath))
            avg = getColorAvg2(getSomePixels(image, 100))
            print fullPath + " : " + str(avg)
            dbtable.insert(dict(path = fullPath, r = avg[0], g = avg[1], b = avg[2], group = group))


def main(argv):
    catalog_directory = None
    group = None
    image_path = None
    n = 100

    try:
        opts, args = getopt.getopt(argv, "c:g:i:n:", ["catalog=", "group=", "image=", "number="])
    except getopt.GetoptError:
        print usage_string
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-c", "--catalog"):
            catalog_directory = arg
        elif opt in ("-g", "--group"):
            group = arg
        elif opt in ("-i", "--image"):
            image_path = arg
        elif opt in ("-n", "--number"):
            n = arg
            
    if catalog_directory:
        catalogImages(catalog_directory, group)
    else:
        i = makeMosaigraph2(Image.open(image_path), n, group)
        i.show()

if __name__ == "__main__":
    main(sys.argv[1:])



"""
img = Image.open('./images/test3.jpg')
makeMosaigraph(img, 10, 10).show()
makeMosaigraph(img, 50, 50).show()
makeMosaigraph(img, 100, 100).show()
makeMosaigraph(img, 300, 300).show()
"""
