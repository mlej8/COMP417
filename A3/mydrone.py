import time
import random
import drawSample
import math
import _tkinter
import sys
import loader
import sys
import pickle
import os.path
import tkinter as tk
import sys
from PIL import Image
from PIL import ImageTk
missing = []
import geoclass
fix = ""
import TileServer
try:
    from PIL import ImageFilter
    from PIL import Image
    from PIL import ImageTk
    from PIL import ImageDraw
except ImportError:
     missing.append( " PIL (more recently known as pillow) " )
     fix = fix + \
        """
          sudo easy_install pip
          sudo pip install pillow
        """
debug = 0  # debug 1 or 2 means using a very simplified setup
verbose=0  # print info (can be set on command line)
doSaveState   = 1     # save a pickle of the data after Gabot filtering to speed up later data analysis
versionNumber = 1.3


documentation = \
"""
  This program is a stb for your COMP 417 robotics assignment.
"""




##########################################################################################
#########  Do non-stardard imports and print helpful diagnostics if necessary ############
#########  Look  for "real code" below to see where the real code goes        ############
##########################################################################################


missing = []
fix = ""

try: 
    import scipy
    from scipy import signal
except ImportError: 
    missing.append( " scipy" )
    fix = fix +  \
        """
        On Ubuntu linux you can try: sudo apt-get install python-numpy python-scipy python-matplotlib 
        On OS X you can try:  
              sudo easy_install pip
              sudo pip install  scipy
        """
try: 
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError: 
    missing.append( " matplotlib" )
    fix = fix +  \
        """
        On Ubuntu linux you can try: sudo apt-get install python-matplotlib
        On OS X you can try:  
              sudo easy_install pip
              sudo pip install matplotlib
        """

try: 
    import numpy as np
except ImportError: 
     missing.append( " numpy " )
     fix = fix + \
        """
          sudo easy_install pip
          sudo pip install numpy
        """
try: 
    from sklearn.decomposition import PCA
    from sklearn.svm import SVC
except ImportError: 
     missing.append( " scikit-learn " )
     fix = fix + \
        """
          sudo easy_install pip
          sudo pip install scikit-learn
        """
try: 
    from PIL import Image
except ImportError: 
     missing.append( " PIL (more recently known as pillow) " )
     fix = fix + \
        """
          sudo easy_install pip
          sudo pip install pillow
        """

if missing:
     print("*"*60)
     print("Cannot run due to missing required libraries of modules.")
     print("Missing modules: ")
     for i in missing: print("    ",i)
     print("*"*60)
     print(fix)
     sys.exit(1)

version = "Greg's drone v%.1f  $HGdate: Thu, 23 Nov 2017 21:59:22 -0500 $ $Revision: 2b96164b4f54 Local rev 0 $" % versionNumber

print(version)
print(" ".join(sys.argv))
##########################################################################################
#########     Parse command-line arguments   #############################################
##########################################################################################
while len(sys.argv)>1:
    if len(sys.argv)>1 and sys.argv[1]=="-v":
        verbose = verbose+1
        del sys.argv[1]
    elif len(sys.argv)>1 and sys.argv[1]=="-load":
        doSaveState = 0
        if len(sys.argv)>2 and not sys.argv[2].startswith("-"):
            loadStateFile = sys.argv[2]
            del sys.argv[2]
        else:
            loadStateFile = 'classifier.state'
        del sys.argv[1]
    elif len(sys.argv)>1 and sys.argv[1]=="-r":
        del sys.argv[1]
        resize_factor = float(sys.argv[1])
        del sys.argv[1]
    elif len(sys.argv)>1 and sys.argv[1] in ["-h", "-help", "--help"]: # help
        print(documentation)
        sys.argv[1] = "-forceusagemesasge"
    elif len(sys.argv)>2 and sys.argv[1]=="-t": # load a directory of test images
        testDirectory = sys.argv[2]
        del sys.argv[1]
        del sys.argv[1]
    elif len(sys.argv)>2 and sys.argv[1]=="-f":
        data_folder_path = sys.argv[2]
        del sys.argv[1]
        del sys.argv[1]
    else:
        print("Unknown argument:",sys.argv[1])
        print("Usage: python ",sys.argv[0]," [-h (help)][-v]    [-f TRAININGDATADIR] [-t TESTDIR] [-load [STATEFILE]]")
        sys.exit(1)


##########################################################################################
#########  "real code" is here, at last!                                      ############
##########################################################################################
def autodraw():
    """ Automatic draw. """
    draw_objects()
    tkwindow.canvas.after(100, autodraw)

def draw_objects():
    """ Draw target balls or stuff on the screen. """
    global tx, ty, maxdx, maxdy, unmoved
    global oldp
    global objectId
    global ts # tileServer
    global actual_pX, actual_pY
    global fill
    global scalex, scaley  # scale factor between out picture and the tileServer

    #tkwindow.canvas.move( objectId, int(tx-MYRADIUS)-oldp[0],int(ty-MYRADIUS)-oldp[1] )
    if unmoved:
        # initialize on first time we get here
        unmoved=0
        tx,ty = 0,0
    else:
        # draw the line showing the path
        tkwindow.polyline([oldp,[oldp[0]+tx,oldp[1]+ty]], style=5, tags=["path"]  )
        tkwindow.canvas.move( objectId, tx,ty )

    # update the drone position
    oldp = [oldp[0]+tx,oldp[1]+ty]

    # map drone location back to lat, lon
    # This transforms pixels to WSG84 mapping, to lat,lon
    # TODO, You need to implement imagePixelsToLL in Tile Server if you need this function
    # right now just default function
    lat,lon = ts.imagePixelsToLL( actual_pX, actual_pY, zoomLevel,  oldp[0]/(256/scalex), oldp[1]/(256/scaley) )

    # get the image tile for our position, using the lat long we just recovered
    im, foox, fooy, fname = ts.tiles_as_image_from_corr(lat, lon, zoomLevel, 1, 1, 0, 0)

    # Use the classifier here on the image "im"

    # This is the drone, let's move it around
    tkwindow.canvas.itemconfig(objectId, tag='userball', fill=fill)
    tkwindow.canvas.drawn = objectId

    #  Take the tile and shrink it to go in the right place
    im = im.resize((int(im.size[0]/scalex),int(im.size[1]/scaley)))
    im.save("/tmp/locationtile.gif")
    photo = tk.PhotoImage(file="/tmp/locationtile.gif" )
    tkwindow.image = tkwindow.canvas.create_image( 256/scalex*int(oldp[0]/(256/scalex)), 256/scalex*int(oldp[1]/(256/scalex)), anchor=tk.NW, image=photo, tags=["tile"] )
    image_storage.append( photo ) # need to save to avoid garbage collection

    # This arrenges the stuff being shown
    tkwindow.canvas.lift( objectId )
    tkwindow.canvas.tag_lower( "tile" )
    tkwindow.canvas.tag_lower( "background" )
    tkwindow.canvas.pack()


    # Code to move the drone can go here
    # Move a small amount by changing tx,ty
    # TODO
    # RIGHT NOW It moves diagonally ...
    tx = 1
    ty = 1



fill = "white"
ts = TileServer.TileServer()

lat, lon = 45.44203, -73.602995    # verdun
tilesX = 20
tilesY = 20
tilesOffsetX = 0
tilesOffsetY = 0
zoomLevel = 18
image_storage = [ ] # list of image objects to avoid memory being disposed of

bigpic, actual_pX, actual_pY, fname = ts.tiles_as_image_from_corr(lat, lon, zoomLevel, tilesX, tilesY, tilesOffsetX, tilesOffsetY)
# bigpic is really big
#bigpic.show()

im = bigpic.resize((1024,1024))
im.save("mytemp.gif") # save the image as a GIF and re-load it does to fragile nature of Tk.PhotoImage


import tkinter as tk
# How to draw a rectangle.
# You should delete or comment out the next 3 lines.
draw = ImageDraw.Draw(bigpic)
xt,yt = 0,0
draw.rectangle(((xt*256-1, yt*256-1), (xt*256+256+1, yt*256+256+1)), fill="red")

# put in image

# Size of our on-screen drawing is arbitrarily small
myImageSize = 1024
# it should divide evenly into the big image
myImageSize = int(256*tilesX/int(256*tilesX/myImageSize))
scalex,scalex = 1,1
scalex = bigpic.size[0]/myImageSize  # scale factor between our picture and the tileServer
scaley = bigpic.size[1]/myImageSize  # scale factor between our picture and the tileServer
im = bigpic.resize((myImageSize,myImageSize))
im = im.filter(ImageFilter.BLUR)
im = im.filter(ImageFilter.BLUR)
im.save("mytemp.gif") # save the image as a GIF and re-load it does to fragile nature of Tk.PhotoImage


tkwindow  = drawSample.SelectRect(xmin=0,ymin=0,xmax=1024 ,ymax=1024, nrects=0, keepcontrol=0 )#, rescale=800/1800.)
root = tkwindow.root
root.title("Drone simulation")

# Full background image
photo = tk.PhotoImage(file="mytemp.gif")
tkwindow.imageid = tkwindow.canvas.create_image( 0, 0, anchor=tk.NW, image=photo, tags=["background"] )
image_storage.append( photo )
tkwindow.canvas.pack()

tkwindow.canvas.pack(side = "bottom", fill = "both",expand="yes")


MYRADIUS = 7
MARK="mark"

# Place our simulated drone on the map
sx,sy=600,640 # over the river
sx,sy = 220,280 # over the canal in Verdun, mixed environment
oldp = [sx,sy]
objectId = tkwindow.canvas.create_oval(int(sx-MYRADIUS),int(sy-MYRADIUS), int(sx+MYRADIUS),int(sy+MYRADIUS),tag=MARK)
unmoved =  1

# initialize the classifier
# We can use it later using these global variables.
#
# launch the drawing thread
autodraw()

#Start the GUI
root.mainloop()

