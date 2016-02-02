About
=====================
Mosaigraph is a command line utility for making photomosaics. 

A "photomosaic" consists of a large number of small images that, when placed next to each other, look like some other image. For example, here's a photomosaic produced using Mosaigraph:

[mona5.jpg](http://saturn597.github.io/mosaics/mona5.jpg)

That image contains about 5000 portraits downloaded from [Wikiart](http://www.wikiart.org/). When those portraits are stitched together into a mosaic, they look like Mona Lisa's face.

Running Mosaigraph
=====================
Mosaigraph is written in Python. It should be compatible with Python 2 or 3. 

You'll need to install [the Pillow library](https://python-pillow.github.io/). To do this using Pip: 

`pip install pillow`

Successfully installing Pillow will require you to be superuser, to use sudo, or to use virtualenv.

Once Pillow is installed, simply download the file "mosaigraph.py" and navigate to the directory that contains it. Then you can run the program as described in the following sections.

Preprocessing
=====================
To start, you'll need a collection of images to serve as individual "pieces" of your mosaic. 

Once you have such a collection, Mosaigraph will need to spend some time inspecting it before the images can be used. This "preprocessing" step is potentially very time consuming.

Fortunately, Mosaigraph can save its preprocessing work to a file. In the future, you can reference the file rather than going through the entire preprocessing stage again. So for a given collection of images, the preprocessing only needs to be done once.

You create a preprocessing file using the `-P` argument (upper case). The `-P` is then immediately followed by a path indicating the filename you want Mosaigraph to save to. You specify the list of files you want to preprocess using the `-i` argument. 

So the command will look like this:

`python mosaigraph.py -P preprocessing_file_path -i image_list`

This command will cause Mosaigraph to preprocess all images referenced by `image_list` and it will save the results to `preprocessing_file_path`.

The paths in `candidate_image_list` can lead either to image files or to directories containing image files. If Mosaigraph encounters a directory in `candidate_image_list`, it will attempt to use all files within that directory as candidate images. Files that cannot be processed as images will be ignored. You can also use Unix wildcards like `*` in the `image_list`, which is convenient when referencing large numbers of images.



Basic usage
=====================
The only things you need to make a mosaic are 1) a "base image" and 2) images to use as pieces of the mosaic. 

Thus, you can run Mosaigraph using a command that looks like this:

`python mosaigraph.py base_image -i candidate_image_list`

The `base_image` argument is a path to a single image file. This is the image the mosaic will look like when viewed as a whole.

The argument to `-i`, `candidate_image_list`, is a space-separated list of paths. The "candidate images" referenced by this list will serve as the individual "pieces" of the mosaic. 

The paths in `candidate_image_list` can lead either to image files or to directories containing image files. If Mosaigraph encounters a directory in `candidate_image_list`, it will attempt to use all files within that directory as candidate images. Files that cannot be processed as images will be ignored. You can also use Unix wildcards like `*` in the `candidate_image_list`, which is convenient when referencing large numbers of images.

Not all candidate images will necessarily end up in the final mosaic. Mosaigraph uses only the ones it thinks "fit" best.

More advanced usage
=====================
A slightly more complex invocation of Mosaigraph looks like this:

`python mosaigraph.py base_image -i candidate_image_list -o output_file -n number_of_pieces -u -w`

You specify an output file using `-o output_file`. Mosaigraph will save the mosaic to that file, using a format based on the file's extension. End `output_file` with ".jpg" to save as a JPEG, ".png" for PNG, etc.

Once a mosaic is complete, Mosaigraph attempts to open it in your system's default image viewer. So even if you don't specify an output file, you should still be able to see the result. But to ensure that your work gets saved, you should specify an output file.

You can specify the number of pieces your mosaic should have using `-n number_of_pieces`. The resulting mosaic won't have exactly this many pieces, but Mosaigraph will try to get close. If the `-n` argument isn't used, Mosaigraph defaults to 500 pieces.

Finally, there are a couple of important options you can set when running Mosaigraph. 

One is `-u`, which guarantees that each piece of the mosaic is "unique." Without the `-u` option, any candidate image can be reused any number of times within the mosaic. When the `-u` option is present, candidate images will not be reused. Each will appear no more than once.

The other important option is `-w`. This causes Mosaigraph to switch to a "slow" method of constructing mosaics. It can take a lot longer to construct a mosaic when you use `-w`, but the results will often look better.

Examples
=====================
Say we want to make a mosaic that looks like [this image](http://saturn597.github.io/mosaics/monaface.jpg) that we cropped from the Mona Lisa and stored as "monaface.jpg".

Say we also have about 18,500 portrait paintings that we obtained from [Wikiart](http://www.wikiart.org/). They're stored in a directory called "wikiartportraits." These portraits can serve as "pieces" of our mosaic.

Both "wikiartportraits" and "monaface.jpg" are in the current directory, the same directory that contains mosaigraph.py.

**Example 1**

We can try starting with this command:

`python mosaigraph.py monaface.jpg -i wikiartportraits -o mona1.jpg`

This creates a file called `mona1.jpg` that might look like this:

[mona1.jpg](http://saturn597.github.io/mosaics/mona1.jpg)

Okay! That's a start!

But we didn't capture much detail from the original image. It's hard to tell it's supposed to be the Mona Lisa, unless you zoom out and squint.

Creating this mosaic didn't take long. It took only about 1 minute on a 15-inch, mid-2010 MacBook Pro, if you disregard the initial processing of candidate images.

Processing the 18,500 candidate images in `wikiartportraits` could take hours. However, using methods discussed below, it is only necessary to process a given set of candidate images once. 

**Example 2**

But let's try making a mosaic with about 2000 pieces instead of the default 500:

`python mosaigraph.py monaface.jpg -i wikiartportraits -o mona2.jpg -n 2000`

We get this:

[mona2.jpg](http://saturn597.github.io/mosaics/mona2.jpg)

That looks more like it!

The 2000 piece mosaic took roughly twice as long to make as the 500 piece one (again, disregarding candidate preprocessing). But it looks much better.

**Example 3**

In the mosaics produced thus far, some candidate images were reused many times. For example, Mosaigraph repeatedly used Egon Schiele's ["Portrait of Frau Dr. Horak"](http://www.wikiart.org/en/egon-schiele/portrait-of-madame-dr-horak-1910) to stand in for Mona Lisa's skin.

Adding the `-u` option prevents Mosaigraph from using any candidate image more than once:

`python mosaigraph.py monaface.jpg -i wikiartportraits -o mona3.jpg -n 2000 -u`

And this is the result:

[mona3.jpg](http://saturn597.github.io/mosaics/mona3.jpg)

Producing a mosaic using `-u` may take longer, since Mosaigraph will have to load a larger number of distinct images. The above image, `mona3.jpg`, took about 7 minutes to create on the test machine (starting from preprocessed candidate images). 

Also, for the `-u` option to work well, you'll need a pool of candidate images that is much larger than the target number of pieces. Otherwise, Mosaigraph will run out of good candidates and the mosaic will not look good.

**Example 4**

We can also construct a mosaic using the "slow" method by adding the `-w` option. We run this command:

`python mosaigraph.py monaface.jpg -i wikiartportraits -o mona4.jpg -n 2000 -w`

Resulting in this image:

[mona4.jpg](http://saturn597.github.io/mosaics/mona4.jpg)

This takes quite a long time. It took 5 hours and 40 minutes on the test machine, not counting candidate preprocessing! But the mosaic looks more like the Mona Lisa than any of the other examples so far.


More Examples
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

Same as the last example, but with the "-w" option - so this mosaic was made using the "slow" method of image comparison. This mosaic looks much better the one produced by the "fast" method.

This is the same mosaic that was listed in the "About" section above.

=====================
[Image](https://www.dropbox.com/s/vz750bx7netbbe6/armstrong1.jpg?dl=0)

`python mosaigraph.py armstrongface.jpg -i hubbleimages -n 3000 -o armstrong1.jpg`

Photo of Neil Armstrong, composed of images taken by the Hubble Space Telescope. The images were drawn from a directory called "hubbleimages" (which contained about 1500 files). Duplicate images were allowed (so no "-u" option) - thus we were able to make a mosaic containing 3000 pieces when we only had 1500 files. The "fast" method of image comparison was used (since the "-w" option was omitted).

Pool of candidate images from [Hubblesite](http://hubblesite.org/).

=====================
[Image](https://www.dropbox.com/s/aguqpixctsay2hw/armstrong2.jpg?dl=0)

`python mosaigraph.py armstrongface.jpg -i hubbleimages -n 3000 -o armstrong2.jpg -w`

Same as above, but using the "slow" image selection method ("-w" option). The face looks less noisy, though in this case the color looks off compared with the "fast" mosaic.

=====================

How candidates are selected
=====================
The default "fast" method looks only at an "average" pixel in each candidate image, and compares it to an "average" pixel in the relevant area of the base image. So if the base image is reddish in a particular area, Mosaigraph will replace it with a candidate image that is reddish overall. This can work, but it means that base image details smaller than the individual pieces of the mosaic are averaged out and lost. 

The "slow" method, used whenever "-w" is specified, compares a number of pixels in each candidate to the *corresponding* pixels in the base image. The idea is that, if a section of the base image is red up above, but blue down below, Mosaigraph will attempt to find a candidate image that is also red up above and blue down below. This allows more detail to come through in the mosaic compared with the fast, averaging method. However, since it ignores overall "average" colors in favor of a detailed, pixelwise comparison, this method is more likely to produce mosaics that seem to have the wrong overall tint to them.

Preprocessing
=====================
To construct a mosaic, Mosaigraph starts by opening and inspecting every single candidate image. This "preprocessing" stage can be very time consuming. 

Fortunately, Mosaigraph offers the option of saving the preprocessed data to a file. The next time you want to reuse the same set of candidate images, Mosaigraph can simply reference the file rather than going through the entire preprocessing stage again.

**Creating a preprocessing file**

You create a preprocessing file by using the `-P` argument (upper case). The `-P` is then immediately followed by a path indicating the file you want to save to. You specify the list of files you want to preprocess using the same `-i` option you used to construct mosaics earlier. 

So the command will look like this:

`python mosaigraph.py -P preprocessing_file_path -i candidate_image_list`

All the images referenced by `candidate_image_list` will be preprocessed and the results saved to `preprocessing_file_path`.

**Using a preprocessing file**

To actually use the preprocessed data, just add the `-p` argument (lower case) when constructing a mosaic. Right after the `-p`, insert the path to the preprocessing file you created earlier. 

So to build a mosaic from an existing preprocessing file, you'll use a command that looks like this:

`python mosaigraph.py base_image -p preprocessing_file_path`

This command will construct a new mosaic using all of the images you preprocessed earlier and saved to `preprocessing_file_path`. Instead of taking the potentially time consuming step of preprocessing those images again, Mosaigraph will use the data it already saved. 

This way, if you make multiple mosaics using the same set of candidate images, you only have to do the preprocessing stage once.

**Important note**

The preprocessing file identifies images by their absolute paths. This means that if you rename images or move them to a different directory after producing the preprocessing file, Mosaigraph will be unable to find them again.

Additional notes on preprocessing
=====================
When constructing a mosaic using a preprocessing file, you don't need to name the candidate images again. All images saved to the preprocessing file will automatically be used as candidate images. You can change this using the `-c` option. If you add a `-c`, Mosaigraph will only use candidate images that you explicitly specify using an `-i` argument. 

When using `-P` to save preprocessing data to a file, you can create a mosaic with the same command. Just specify a `base image` like any other time you are constructing a mosaic. So the command will be: 

`python mosaigraph.py base_image -P preprocessing_file_path -i candidate_image_list`

If you repeatedly run mosaigraph.py with `-P`, using the same `preprocessing_file_path` but different candidate images, the preprocessing data for any new candidate images will be added to the file. Candidate images that you already preprocessed won't be processed again.

Your mosaics will look the same whether or not you use a preprocessing file. And constructing mosaics can still be time consuming. The point of using the preprocessing file is simply to avoid repeating the preprocessing step as much as possible.

Preprocessing file example
=====================
Here's an example:

[Image](https://www.dropbox.com/s/axndethqpxdr70w/cedar.jpg?dl=0) | [Image](https://www.dropbox.com/s/131kgtm4iqnfz1k/guido.jpg?dl=0) | [Image](https://www.dropbox.com/s/kh1lka55chdr9qg/maui.jpg?dl=0)

`python mosaigraph.py cedarface.jpg -w -n 2000 -o cedar.jpg -i catpictures -P catpreprocessing`

`python mosaigraph.py guidoface.jpg -w -n 2000 -o guido.jpg -p catpreprocessing`

`python mosaigraph.py mauiface.jpg -w -n 2000 -o maui.jpg -p catpreprocessing`

In the first command, we not only create a mosaic out of a cat's face and save it as "cedar.jpg", but we also create a file called "catpreprocessing" that will save the information Mosaigraph learned about the candidate images we used.

In the subsequent commands, we refer to the "catpreprocessing" file (using "-p") instead of referencing the "catpictures" directory again. This lets Mosaigraph avoid duplicating effort that it already made in the first command (namely, opening and inspecting every image in catpictures). Thus, the second and third mosaics can be made more quickly.

The pool of images in these mosaics came from [Pixabay](https://pixabay.com/).

Ways to improve your mosaics
=====================
If you would like your mosaics to look better, there are a few options:

* Use a higher piece count (the "-n" command line argument).
* Use more and more varied candidate images. [Scrapy](http://scrapy.org/) provides one way to quickly acquire lots of images. Also see Wikimedia's [free media resource lists](https://commons.wikimedia.org/wiki/Commons:Free_media_resources).
* Crop base images strategically. Including unnecessary background "wastes" a lot of pieces on the background, rather than using them for the more interesting parts of the image.
* Try using the "slow" comparison method (the "-w" option).

Option and argument reference
=====================
Mosaigraph has a number of additional arguments and options. See below for a more complete reference.

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

You might also consider using the "-r" option to randomize the order in which pieces are added. This way, pieces selected later will still look worse than ones added earlier, but they won't be grouped to one region of the mosaic (say, the right, if we started from the left). However, if you are not using enough candidate images in the first place, this just leads to a mosaic that looks equally bad all over.

An improved Mosaigraph would attempt to find an *overall* best fit when uniqueness is required - not only trying to match each piece to the best candidate, but also matching each candidate used to the best piece. That approach may be implemented at a later date.

**-w: Select images using the "pixelwise" method.**

Deciding which image to use in which piece of the mosaic involves dividing up the base image into sections, then finding a candidate that is visually similar to each section. Mosaigraph can use either of two basic methods to determine visual similarity. The first is fast but, because it only checks the "average" pixel in each image, it doesn't always result in the sharpest looking mosaic. The second is slow because it compares candidates to the base image pixel-by-pixel, but it often has higher quality results. The first, faster method is used by default. To use the second, slower method, use the "-w" option.

Mosaic construction using the "-w" option will take longer, but the mosaics will often be of a higher quality and look more like the original image.

**-x: Don't display mosaic after completion.**
 
If you use the "-x" option, Mosaigraph will not display the mosaic after constructing it. If you don't specify an output file, the constructed mosaic will be lost and never seen. 
