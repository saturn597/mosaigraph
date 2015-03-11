from PIL import Image

import re
import os

def getColorAvg(im):
    # returns the average color of all pixels in an image
    width, height = im.size

    r, g, b = 0, 0, 0
    for x in range(0, width - 1):
        for y in range(0, height - 1):
            r += im.getpixel((x, y))[0]
            g += im.getpixel((x, y))[1]
            b += im.getpixel((x, y))[2]
           
    numPixels = width * height
    
    r = float(r) / numPixels
    g = float(g) / numPixels
    b = float(b) / numPixels

    return (r, g, b)

def getImages(d):
    # return a list of image files in directory d
    isimg = re.compile('\.(png|jpg)$')
    return [f for f in os.listdir(d) if re.search(isimg, f)]

print getImages('.')

im = Image.open('test3.jpg')

