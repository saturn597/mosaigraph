from PIL import Image

import dataset
import random
import re
import os

imagePath = './images'
dbname = 'sqlite:///data.db'
dbtable = 'image'

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
    # take rectangle of width and height, and divide it into squares of pieceWidth x pieceHeight
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

def makeMosaigraph(img, horizDivs, vertDivs):
    newImg = Image.new(img.mode, img.size)
    width, height = img.size
    pieceWidth = width / horizDivs
    pieceHeight = height / vertDivs
    pieces = divideImage(img, pieceWidth, pieceHeight)
    for piece in pieces:
        r, g, b = getColorAvg2(getSomePixels(piece['image'], 1))
        addOn = Image.new('RGB', (pieceWidth, pieceHeight), (r, g, b))
        newImg.paste(addOn, (piece['x'], piece['y']))
    return newImg
        
    
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


def catalogImages(imagePath):
    db = dataset.connect(dbname)
    table = db[dbtable]
    for filename in getImages(imagePath):
        fullPath = os.path.join(imagePath, filename)
        image = makeProportional(Image.open(fullPath))
        if not table.find_one(path = fullPath):  # only do all the calculations if we haven't already checked this file
            avg = getColorAvg(image)
            table.insert(dict(path = fullPath, r = avg[0], g = avg[1], b = avg[2]))
            print filename + ' ' + str(getColorAvg(image))



img = Image.open('./images/test3.jpg')
makeMosaigraph(img, 10, 10).show()
makeMosaigraph(img, 50, 50).show()
makeMosaigraph(img, 100, 100).show()
makeMosaigraph(img, 300, 300).show()
