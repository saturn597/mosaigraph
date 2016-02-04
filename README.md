About
=====================
Mosaigraph is a command line utility for making photomosaics. 

A "photomosaic" consists of a large number of small images that, when viewed together, look like some other, larger image. For example, here's a photomosaic produced using Mosaigraph:

[mona6.jpg](http://saturn597.github.io/mosaics/mona6.jpg)

That file contains about 5000 portraits obtained from [Wikiart](http://www.wikiart.org/). When those portraits are stitched together into a mosaic, they look like Mona Lisa's face.

Here's another mosaic:

[armstrong1.jpg](http://saturn597.github.io/mosaics/armstrong1.jpg)

It's Neil Armstrong composed of photographs taken by the Hubble Space telescope.

Hubble images from [HubbleSite](http://hubblesite.org/). Neil Armstrong photo from [Wikimedia](https://commons.wikimedia.org/wiki/File:Neil_Armstrong_pose.jpg).

And here are some pictures of cats, made up of pictures of cats:

[cedar.jpg](http://saturn597.github.io/mosaics/cedar.jpg) |
[guido.jpg](http://saturn597.github.io/mosaics/guido.jpg) |
[maui.jpg](http://saturn597.github.io/mosaics/maui.jpg)

Pictures of cats from [Pixabay](Pixabay.com) and from personal collection.

Running Mosaigraph
=====================
Mosaigraph is written in Python. It should be compatible with Python 2 or 3. 

You'll need to install [the Pillow library](https://python-pillow.github.io/) to use Mosaigraph. To do this using Pip: 

`pip install pillow`

Installing Pillow will require you to be superuser, to use sudo, or to use virtualenv.

Once Pillow is installed, simply download the file "mosaigraph.py" and navigate to the directory that contains it. One approach:

`git clone https://github.com/saturn597/mosaigraph.git`

`cd mosaigraph`

Now you can run the program as described in the following sections.

Quick start
=====================
Say you have a "base image", the image you want your mosaic to look like when viewed as a whole. It is stored at `base_image_path`.

Say you also have a collection of "candidate images," i.e., images Mosaigraph can use as the pieces of the mosaic. They are stored in the directory `candidate_image_directory`.

So how do you use Mosaigraph to make a mosaic?

First, preprocess the candidate images and save the results to a file at `preprocessing_file_path`:

`python mosaigraph.py -P preprocessing_file_path -i candidate_image_directory`

This creates the file `preprocessing_file_path` if it doesn't exist and saves information about your candidate images to that file.

Now, using the same `preprocessing_file_path` you created above, the command to make a mosaic will be: 

`python mosaigraph.py base_image_path -p preprocessing_file_path [-o output_file] [-n number_of_pieces] [-u] [-w]`

Once the mosaic is complete, Mosaigraph will try to show the result in your system's image viewer. If an output file is specified, Mosaigraph will also save the mosaic to that file. The format of the file depends on the extension you use. End the filename in ".jpg" for a jpg, etc. To ensure that your work gets saved, you should specify an output file.

Mosaics consist of about 500 pieces by default. You can specify a different number of pieces using the `-n` flag as above. Mosaigraph will try to get close to the specified number of pieces, though it may not be exact.

Use `-u` to ensure that each piece is "unique," i.e., that a given candidate image won't be used more than once.

Using `-w` triggers "slow mode" - Mosaigraph will proceed much more slowly but may produce better mosaics. It can take a lot longer to produce mosaics this way.

More on preprocessing
=====================
This "preprocessing" step discussed above is potentially very time consuming if you have a lot of images. That's why Mosaigraph lets you save the results of preprocessing to a file. 

When starting out, consider making a preprocessing file with only a small number of images (<100). Using only a few images won't result in the best mosaics, but everything will happen faster. This way it'll be easier to play around and try things out.

In the command above to create a preprocessing file, the `-i` argument doesn't only take a single directory. It actually takes a space-separated list of directories and image files. You can use Unix wildcards like `*`, which is convenient when referencing large numbers of images.If Mosaigraph encounters a directory in the list it will attempt to preprocess all files in that directory. Files that cannot be processed as images will be ignored. Mosaigraph only preprocesses all files within those directories that are explicitly specified - if a directory you list contains a second directory, the second directory will be ignored.

If you decide you want to add more images to your preprocessing file, just run the same command you used to create the file, but specify a new set of images. The preprocessing data for any new images will be added to the file. Images that you already preprocessed won't be processed again.

Your mosaics will look the same whether or not you use a preprocessing file. And constructing mosaics can still be time consuming. The point of using the preprocessing file is simply to avoid repeating the preprocessing step as much as possible.

The preprocessing file identifies images by their absolute paths. This means that if you rename images or move them to a different directory after producing the preprocessing file, Mosaigraph will be unable to find them again.

Examples
=====================
Say we want to make a mosaic that looks like [this image](http://saturn597.github.io/mosaics/monaface.jpg) that we cropped from the Mona Lisa and stored as "monaface.jpg" in our current directory.

Say we also have about 18,500 portrait paintings that we obtained from [Wikiart](http://www.wikiart.org/). They're stored in the directory "~/mosaics/portraits" These portraits can serve as "pieces" of our mosaic.

So let's start by preprocessing those files and saving the results so we can use them to make a few mosaics. Let's save the results to a file called "portraitpreprocessing" in our current directory. To do this, you'd run this command:

`python mosaigraph.py -P portraitpreprocessing -i ~/mosaics/portraits`

If you really have 18,500 images, you'll have to let this run over night and then some.

But once it's done we can make some mosaics!

**Example 1**

We can try this command:

`python mosaigraph.py monaface.jpg -p portraitpreprocessing -o mona1.jpg`

This creates a file called `mona1.jpg` that might look like this:

[mona1.jpg](http://saturn597.github.io/mosaics/mona1.jpg)

Okay! That's a start!

Once the preprocessing was done, creating this mosaic didn't take long. The above command took only about 1 minute on a 15-inch, mid-2010 MacBook Pro.

But we didn't capture much detail from the original image. It's hard to tell it's supposed to be the Mona Lisa, unless you zoom out and squint.

**Example 2**

But let's try making a mosaic with about 2000 pieces instead of the default 500:

`python mosaigraph.py monaface.jpg -p portraitpreprocessing -o mona2.jpg -n 2000`

We get this:

[mona2.jpg](http://saturn597.github.io/mosaics/mona2.jpg)

That's an improvement!

The 2000 piece mosaic took roughly twice as long to make as the 500 piece one (again, not counting the initial creation of the preprocessing file). But it looks much better.

**Example 3**

In the mosaics produced thus far, some candidate images were reused many times. For example, Mosaigraph repeatedly used Egon Schiele's ["Portrait of Frau Dr. Horak"](http://www.wikiart.org/en/egon-schiele/portrait-of-madame-dr-horak-1910) to stand in for Mona Lisa's skin.

Adding the `-u` option prevents Mosaigraph from using any candidate image more than once:

`python mosaigraph.py monaface.jpg -p portraitpreprocessing -o mona3.jpg -n 2000 -u`

And this is the result:

[mona3.jpg](http://saturn597.github.io/mosaics/mona3.jpg)

Producing a mosaic using `-u` may take longer, since Mosaigraph will have to load a larger number of distinct images. The above image, `mona3.jpg`, took about 7 minutes to create on the test machine (again, starting from the preprocessed candidate images in `portraitpreprocessing`). 

For the `-u` option to work well, you'll need a pool of candidate images that is much larger than the target number of pieces. Otherwise, Mosaigraph will run out of good candidates and the mosaic will have large areas with little resemblance to the base image.

**Example 4**

We can also construct a mosaic using the "slow" method by adding the `-w` option. We run this command:

`python mosaigraph.py monaface.jpg -p portraitpreprocessing -o mona4.jpg -n 2000 -w`

Resulting in this image:

[mona4.jpg](http://saturn597.github.io/mosaics/mona4.jpg)

This takes quite a long time. It took 5 hours and 40 minutes on the test machine, still not counting candidate preprocessing! But the mosaic looks more like the Mona Lisa than any of the other examples so far.

**Example 5**

For completeness, let's also try using the `-u` option along with the `-w` option. So the mosaic will be constructed using the "slow" method and each "piece" of it will be unique, without reuse of images. 

The command looks like this:

`python mosaigraph.py monaface.jpg -p portraitpreprocessing -o mona5.jpg -n 2000 -w -u`

And the result looks like this:

[mona5.jpg](http://saturn597.github.io/mosaics/mona5.jpg)

The command took about 5 hours, 12 minutes to run.

**Example 6**

The Mona Lisa mosaic in the "About" section above was created similarly, but with 5000 individual pieces instead of 2000.

`python mosaigraph.py monaface.jpg -p portraitpreprocessing -o mona6.jpg -n 5000 -w -u`

[mona6.jpg](http://saturn597.github.io/mosaics/mona6.jpg)

**Example 7**

The Neil Armstrong image in the "About" section was creating using a command like this:

`python mosaigraph.py armstrongface.jpg -p hubbleimages -n 3000 -o armstrong1.jpg`

[armstrong1.jpg](http://saturn597.github.io/mosaics/armstrong1.jpg)

`hubbleimages` was a preprocessing file containing only about 1500 files. we couldn't use the `-u` flag because there weren't enough files.
 
Using the "slow" approach on Neil Armstrong looks like this:

[armstrong2.jpg](http://saturn597.github.io/mosaics/armstrong2.jpg)

Ways to improve your mosaics
=====================
If you would like your mosaics to look better, there are a few options:

* Use a higher piece count (the "-n" command line argument).
* Use more and more varied candidate images. [Scrapy](http://scrapy.org/) is one way to quickly acquire lots of images. Also see Wikimedia's [free media resource lists](https://commons.wikimedia.org/wiki/Commons:Free_media_resources).
* Crop base images strategically. Including unnecessary background "wastes" a lot of pieces on the background, rather than using them for the more interesting parts of the image.
* Try using the "slow" comparison method (the "-w" option).

Option and argument reference
=====================
Mosaigraph has a few other arguments and options. See below for a more complete reference.

**-c: Only use candidates explicitly specified in the "-i" argument.**

By default, if a preprocessing file is specified using "-p", all images saved in the file are used as candidate images. The "-c" option tells Mosaigraph not to do this. Only images explicitly specified in an "-i" argument will be used as candidates. 

This lets you specify a preprocessing file (and thus avoid repeatedly preprocessing images stored in it) without having to use *all* of the images it contains.

**-e [edge length in pixels]: Change the size of the pieces.**

Mosaigraph scales each image it uses as a piece of the mosaic in order to make each piece the same size (and clips them to make them square). 

By default, each piece will be 100x100 pixels. 

You can modify the size Mosaigraph scales images to using the -e argument. Setting "-e 200" will result in pieces that are 200x200 pixels, for example.

**-i [image list]: Specify images to use in constructing the mosaic.**

This is a space separated list of paths to images and/or directories that contain images.

If you are building a mosaic, the images will be used as the "pieces" of the mosaic. If you are just saving preprocessing data, data on the referenced images will be saved to your preprocessing file.

The paths in "image list" can lead either to image files or to directories containing image files. If Mosaigraph encounters a directory in "image list", it will attempt to preprocess all files in that directory. Files that cannot be processed as images will be ignored. You can also use Unix wildcards like `*` in the "image list", which is convenient when referencing large numbers of images.

**-l [log file path]: Produce log of which images are used where.**

When building a mosaic, use -l followed by a filename to output json-formatted information about the mosaic's construction. The json output is an array of javascript objects, each of which contains information about one of the pieces of the mosaic. That information includes the "x coordinate" and "y coordinate" of the piece, where the top left piece is (0, 0) and the one to its right is (1, 0), etc. It also includes the path of the candidate image used for that piece.

**-n [number]: Adjust the number of pieces in the mosaic.**

By default, mosaics constructed by Mosaigraph will be composed of about 500 pieces. If you want your mosaic to be composed of more or fewer pieces, you can adjust this with the "-n" argument.

The precise number of pieces in the final mosaic will usually not be exactly what you specify. This is because Mosaigraph tries to capture as much of the base image as it can, which constrains the proportions of the mosaic, and because the number of rows and the number of columns have to be integers, and the height and width of each piece in pixels have to be integers.

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

Most of the time, the order in which pieces are added makes no difference. However, when using the "-u" option, the order does matter. See the discussion of this issue in the reference for the "-u" option.

**-s [number]: Adjust the sample size.**

When deciding which of the candidate images to use, Mosaigraph samples pixels from each section of the base image, and then compares those samples with samples it takes from each candidate image. The candidate whose sampled pixels most resemble the sample from a given section of the base image is the one used in the corresponding section of the mosaic.

By default, the sample size used (both for sampling the candidate images and for each section of the base image) is 300. Usually this default works well. If you want, you can modify the sample size using the -s argument. 

Very low sample sizes will be faster, but will not yield good results.

**-u: Don't reuse candidate images.**

By default, Mosaigraph is allowed to reuse a given candidate image multiple times in a given mosaic. To make each piece "unique," so that candidate images are not reused, you can use the -u option. 

Of course, if the -u option is selected, you have to provide at least enough candidate images to fill all the pieces. 

Further, Mosaigraph's current approach to ensuring "uniqueness" is to simply remove images from the candidate pool as they are added to the mosaic. This means that areas of the mosaic constructed later may look worse than those constructed earlier, since we start to run out of good candidate images as we move along in the process. Since by default images are added to the mosaic from left to right, the right side can look worse than the left.

To mitigate this problem, use a large pool of candidate images relative to the number of pieces in your mosaic, so you don't end up "running low" as you get close to the end.

You can also consider using the "-r" option to randomize the order in which pieces are added. This way, pieces selected later will still look worse than ones added earlier, but they won't be grouped to one region of the mosaic (say, the right, if we started from the left). However, if you are not using enough candidate images in the first place, this just leads to a mosaic that looks equally bad all over.

An improved Mosaigraph would attempt to find an *overall* best fit when uniqueness is required - not only trying to match each piece to the best candidate, but also matching each candidate used to the best piece. That approach may be implemented at a later date.

**-w: Select images using the slow, "pixelwise" method.**

Deciding which image to use in which piece of the mosaic involves dividing up the base image into sections, then finding a candidate that is visually similar to each section. Mosaigraph can use either of two basic methods to determine visual similarity. The first is fast but, because it only checks the "average" pixel in each image, it doesn't always result in the sharpest looking mosaic. The second is slow because it compares candidates to the base image pixel-by-pixel, but it often has higher quality results. The first, faster method is used by default. To use the second, slower method, use the "-w" option.

Mosaic construction using the "-w" option will take longer, but the mosaics will often be of a higher quality and look more like the original image.

**-x: Don't display mosaic after completion.**
 
If you use the "-x" option, Mosaigraph will not display the mosaic after constructing it. If you don't specify an output file, the constructed mosaic will be lost and never seen. 
