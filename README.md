About
=====================
Mosaigraph is a command line utility for making photomosaics. It is written in Python and should be compatible with Python 2 or 3. 

Mosaigraph depends on [the Pillow library](https://python-pillow.github.io/). You can install Pillow with: 

`pip install pillow`

You will probably need to be a superuser, use sudo, or use virtualenv for the installation of Pillow to work.

Otherwise, Mosaigraph only needs a Python interpreter to run. Download the source code contained in mosaigraph.py and use it as described in the following sections.

[Here's](https://www.dropbox.com/s/qn8392c8rlo6x3s/mona1.jpg?dl=0) an example of what you can make with Mosaigraph.

Basic usage
=====================
A simple way to run Mosaigraph looks like this:

`python mosaigraph.py base_image -i candidate_image_list`

"base\_image" is a path to an image file. The argument to "-i", "candidate\_image\_list", is a space-separated list of paths either to image files or to directories containing image files. 

The above command will produce a mosaic that resembles "base_image" from a distance. A closer look will reveal that it is actually made up of about 500 smaller images taken from "candidate_image_list". 

If you have your candidate images in a directory called "catpictures" and your command looks like this:

`python mosaigraph.py monalisa.jpg -i catpictures`

...you might get a result that looks like [this](https://www.dropbox.com/s/jix588oep7f7gq9/mona2.jpg?dl=0).

(The candidate images in that image came from [Pixabay](https://pixabay.com/)).

Basic options
=====================
By passing different arguments and options on the command line, you can change the way the mosaic is constructed and looks. Here are a few.

* By default, mosaics will be composed of approximately 500 square pieces, but you can specify a different number using the "-n" argument. 

* A given image in "candidate_image_list" may be reused more than once in the mosaic, unless you pass the "-u" option to make each piece unique. 

* Mosaigraph will try to open the mosaic in an image viewer after it is made. If you specify an output file using the "-o" argument, it will also save the image to the specified file. The format depends on the filename extension you use - so use ".jpg" to save as a JPEG, ".png" for PNG, etc.

* Mosaigraph can use either of two different methods for choosing which of the candidate images to use in the mosaic. The default method is fast, but doesn't always yield the best results. To use a slower method that often provides better results, use the "-w" option. 

Fast versus slow candidate selection
=====================
Specifically, the default "fast" method looks only at an "average" pixel in each candidate image, and compares it to an "average" pixel in the relevant area of the base image. So if the base image is reddish in a particular area, Mosaigraph will replace it with a candidate image that is reddish overall. This can work, but it means that base image details smaller than the individual pieces of the mosaic are averaged out and lost. 

The "slow" method, used whenever "-w" is specified, compares a number of pixels in each candidate to the *corresponding* pixels in the base image. The idea is that, if a section of the base image is red up above, but blue down below, Mosaigraph will attempt to find a candidate image that is also red up above and blue down below. This allows more detail to come through in the mosaic compared with the fast, averaging method. However, since it ignores overall "average" colors in favor of a detailed, pixelwise comparison, this method is more likely to produce mosaics that seem to have the wrong overall tint to them.

Examples
=====================
Here are some more example mosaics along with commands that could have made them.

=====================
[Image](https://www.dropbox.com/s/j2c5aveia113esx/mona3.jpg?dl=0)

`python mosaigraph.py monaface.jpg -i wikiartportraits -n 5000 -u -o mona3.jpg`

Mosaic of Mona Lisa's face, composed of about 5000 images. Those 5000 were selected from a directory called "wikiartportraits," which contained about 18000 images. Each image was used no more than once, due to the "-u" option. The default "fast" image selection method was used, since the "-w" option is not present. The mosaic was saved to a file called "mona3.jpg".

The pool of candidate images was from [Wikiart](http://www.wikiart.org/).

=====================
[Image](https://www.dropbox.com/s/qn8392c8rlo6x3s/mona1.jpg?dl=0)

`python mosaigraph.py monaface.jpg -i wikiartportraits -n 5000 -w -u -o mona1.jpg`

Same as the flast example, but with the "-w" option - so this mosaic was made using the "slow" method of image comparison. This mosaic looks much better the one produced by the "fast" method.

This is the same mosaic that was listed in the "About" section above.

=====================
[Image](https://www.dropbox.com/s/vz750bx7netbbe6/armstrong1.jpg?dl=0)

`python mosaigraph.py armstrongface.jpg -i hubbleimages -n 3000 -o armstrong1.jpg`

Photo of Neil Armstrong, composed of images taken by the Hubble Space Telescope. The images were drawn from a directory called "hubbleimages" (which contained about 1500 files). Duplicate images were allowed (so no "-u" option) - thus we were able to make a mosaic containing 3000 pieces when we only had 1500 files. The "fast" method of image comparison was used (since the "-w" option was omitted).

Pool of candidate images from [Hubblesite](http://hubblesite.org/).

=====================
[Image](https://www.dropbox.com/s/aguqpixctsay2hw/armstrong2.jpg?dl=0)

`python mosaigraph.py armstrongface.jpg -i hubblehimages -n 3000 -o armstrong2.jpg -w`

Same as above, but using the "slow" image selection method ("-w" option). The face looks less noisy, though in this case the color looks off compared with the "fast" mosaic.

=====================

Preprocessing files
=====================

The first step of constructing a mosaic involves opening and inspecting each candidate image. Specifically, Mosaigraph obtains a sample of pixels from each image and finds out what color each sampled pixel is, all so that it can later determine how similar that sample is to pixels in the base image.

Because it involves opening and fully decoding every single candidate image, this "preprocessing" stage can be time consuming. Thus, Mosaigraph offers the option of saving the preprocessed data (i.e., what color each sampled pixel was) to a file.

The next time you want to reuse that set of candidate images, Mosaigraph can simply reference the file rather than going through the entire preprocessing stage again.

You create a preprocessing file using the "-P" argument. The "-P" should be followed by the filename to which to save. The command will look like this:

`python mosaigraph.py -P preprocessing_file_name -i candidate_image_list`

All the images referenced by "candidate_image_list" will be preprocessed and the results saved to "preprocessing_file_name." If you also specify a base image, Mosaigraph will construct a mosaic after preprocessing, as well as saving the preprocessing data to the file.

To save more images to the same file, just repeat with the additional images, using -P and specifying the file again.

To actually use the preprocessed data, just add the "-p" argument (lower case) when constructing a mosaic and pass it the preprocessing file you created with "-P". (You can also use upper case "-P" to use the preprocessed data, but with the side effect that any additional images specified with "-i" will have their preprocessing data saved to the file).

A command to construct a mosaic from a preprocessing file will look like this:

`python mosaigraph.py base_image -p preprocessing_file_name`

Note that, when constructing a mosaic using a preprocessing file, you do *not* need to reference the candidate images using the "-i" option again - all images saved to the preprocessing file will automatically be used as candidates in constructing the mosaic. 

If you don't want all of the images in a given preprocessing file to automatically be used as candidates, add the "-c" option. This will tell Mosaigraph to only use as candidates those files that were explicitly referenced in the "-i" argument.

Your mosaics will look basically the same whether you do or do not use a preprocessing file. And constructing mosaics can still be time consuming. The point of using the preprocessing file is simply to avoid repeating the preprocessing step as much as possible. 

**NOTE:** The preprocessing file identifies images by their absolute paths. This means that if you rename images or move them to a different directory after producing the preprocessing file, Mosaigraph will be unable to find them again.

Preprocessing file example
=====================
Here's an example:

[Image](https://www.dropbox.com/s/axndethqpxdr70w/cedar.jpg?dl=0) | [Image](https://www.dropbox.com/s/131kgtm4iqnfz1k/guido.jpg?dl=0) | [Image](https://www.dropbox.com/s/kh1lka55chdr9qg/maui.jpg?dl=0)

`python mosaigraph.py cedarface.jpg -w -n 2000 -o cedar.jpg -i catpictures -P catpreprocessing`

`python mosaigraph.py guidoface.jpg -w -n 2000 -o guido.jpg -p catpreprocessing`

`python mosaigraph.py mauiface.jpg -w -n 2000 -o maui.jpg -p catpreprocessing`

In the first command, we not only create a mosaic out of a cat's face and save it as "cedar.jpg", but we also create a file called "catpreprocessing" that will save the information Mosaigraph learned about the candidate images we used (which were stored in the catpictures directory

In the subsequent commands, we refer to the "catpreprocessing" file (using "-p") instead of referencing the "catpictures" directory again. This lets Mosaigraph avoid duplicating effort that it already made in the first command (namely, opening and inspecting every image in catpictures). Thus, the second and third mosaics can be made more quickly.

The pool of images in these mosaics came from [Pixabay](https://pixabay.com/).

Ways to improve your mosaics
=====================
If you would like your mosaics to look better, there are a few options:

* Use a higher piece count (the "-n" command line argument).
* Use more and more varied candidate images. [Scrapy](http://scrapy.org/) can be a helpful tool for acquiring large numbers of images. Also see Wikimedia's [free media resource lists](https://commons.wikimedia.org/wiki/Commons:Free_media_resources).
* Crop base images strategically. Including unnecessary background "wastes" a lot of pieces on the background, rather than using them for the more interesting parts of the image.
* Try using the "slow" comparison method (the "-w" option).

Option and argument reference
=====================
**-c: Only use candidates explicitly specified in the "-i" argument.**

By default, if a preprocessing file is specified using "-p", all images saved in the file are used as candidate images. The "-c" option tells Mosaigraph not to do this. Only images explicitly specified in the "-i" argument will be used as candidates. 

This lets you specify a preprocessing file (and thus avoid repeatedly preprocessing images stored in it) without having to use *all* of the images it contains.

**-e [edge length in pixels]: Change the size of the pieces.**

Mosaigraph scales each image it uses as a piece of the mosaic in order to make each piece the same size (and clips them to make them square). 

By default, each piece will be 100x100 pixels. 

You can modify the size Mosaigraph scales images to using the -e argument. Setting "-e 200" will result in pieces that are 200x200 pixels, for example.

**-i [image list]: Specify images to use in constructing the mosaic.**

This is a space separated list of images and/or directories in which to find images. The referenced images will be used as the "pieces" of the mosaic being built.

**-l [log file path]: Produce log of which images are used where.**

When building a mosaic, use -l followed by a filename to output json-formatted information about the mosaic's construction. The json output is an array of javascript objects, each of which contains information about one of the pieces of the mosaic. That information includes the "x coordinate" and "y coordinate" of the piece (where the top left piece is (0, 0) and the one to its right is (1, 0), etc.), and the path of the candidate image used for that piece.

This allows you to easily find out which images were used in the mosaic and where.

**-n [number]: Adjust the number of pieces in the mosaic.**

By default, mosaics constructed by Mosaigraph will be composed of about 500 pieces. If you want your mosaic to be composed of more or fewer pieces, you can adjust this with the "-n" argument.

The precise number of pieces in the mosaic will usually not be exactly what was specified, but it should be fairly close. 

It will take longer to build a mosaic with more pieces. However, using more pieces usually means a sharper looking result.

**-o [output file path]: Specify an output file.**

If you want Mosaigraph to write the mosaic it makes to a file, use "-o filename". For example, to write the mosaic image to a file called "monacats.jpg" you might use:

`python mosaigraph.py base_image -i image_list -o monacats.jpg`

The filename extension you use will determine the format of the output. For example, use ".jpg" to produce JPEG output, or ".png" to produce PNG output, etc. You should be able to use any file type supported by the Pillow library and by your system. 

**-P [path to preprocessing file]: Save preprocessing data to this file.**

Mosaigraph will save the results of any "preprocessing" to this file, if it is specified. In particular, any images listed under the "-i" argument that are not already in the preprocessing file will be preprocessed and the resulting data saved to the file.

**-p [path to preprocessing file]: Use preprocessing data from this file.**

When constructing a mosaic, all images saved to the specified preprocessing file will be used as candidate images. Instead of preprocessing those images (potentially a time consuming step), Mosaigraph will simply pull the data from the file.

**-r: Randomize the order in which pieces are added to the mosaic.**

Most of the time, the order in which pieces are added makes no difference. However, when using the "-u" option, the order does matter. See the discussion of this issue in the description of the "-u" option.

**-s [number]: Adjust the sample size.**

When deciding which of the candidate images to use, Mosaigraph samples pixels from each section of the base image, and then compares those samples with samples it takes from each candidate image. The candidate whose sampled pixels most resemble the sample from a given section of the base image is the one used in the corresponding section of the mosaic.

By default, the sample size used (both for sampling the candidate images and for each section of the base image) is 300. Usually this default works well. You can modify this sample size using the -s argument. 

Very low sample sizes will be faster, but will not yield good results.

**-u: Don't reuse candidate images.**

By default, Mosaigraph will reuse a given candidate image multiple times in a given mosaic, if it fits well in multiple places. To make each piece "unique," so that candidate images are not reused, you can use the -u option. 

Of course, if the -u option is selected, you have to provide at least enough candidate images to fill all the pieces. 

Further, Mosaigraph's current approach to ensuring "uniqueness" is to simply remove images from the candidate pool as they are added to the mosaic. This means that areas of the mosaic constructed later may look worse than those constructed earlier, since we start to run out of good candidate images as we move along in the process. Since by default images are added to the mosaic from left to right, the right side can look worse than the left as a result.

To mitigate this problem, use a large pool of candidate images relative to the number of pieces in your mosaic, so you don't end up "running low" as you get close to the end.

You might also consider using the "-r" option to randomize the order in which pieces are added. This way, pieces selected later will still look worse than ones added earlier, but they won't be grouped to one region of the mosaic (say, the right, if we started from the left). However, if you are not using enough candidate images in the first place, this can lead to a mosaic that looks equally bad all over.

An improved Mosaigraph would attempt to find an *overall* best fit when uniqueness is required - not only trying to match each piece to the best candidate, but also matching each candidate used to the best piece. That approach may be implemented at a later date.

**-w: Select images using the "pixelwise" method.**

Deciding which image to use in which piece of the mosaic involves dividing up the base image into sections, then finding a candidate that is visually similar to each section. Mosaigraph can use either of two basic methods to determine visual similarity. The first is fast but, because it only checks the "average" pixel in each image, it doesn't always result in the sharpest looking mosaic. The second is slow because it compares candidates to the base image pixel-by-pixel, but it often has higher quality results. The first, faster method is used by default. To use the second, slower method, use the "-w" option.

Mosaic construction using the "-w" option will take longer, but the mosaics will often be of a higher quality and look more like the original image.

**-x: Don't display mosaic after completion.**
 
If you use the "-x" option, Mosaigraph will not display the mosaic after constructing it. If you don't specify an output file, the constructed mosaic will be lost and never seen. 
