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
import pyscreenshot as ImageGrab
import imageio
import datetime
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
    brownian = True
    draw_objects(brownian)
    global delta_t
    tkwindow.canvas.after(delta_t, autodraw)

def draw_objects(brownian=True):
    """ Draw target balls or stuff on the screen. """
    global tx, ty, maxdx, maxdy, unmoved
    global oldp, og_lat, tilesX, tilesY, og_lon, maps_storage, objectId, ts, actual_pX, stack, actual_pY, fill, scalex, scaley, delta_t, myImageSize, total_distance, tiles_occ, terrain_distribution, num_tiles_visited, pca, clf, classnames

    #tkwindow.canvas.move( objectId, int(tx-MYRADIUS)-oldp[0],int(ty-MYRADIUS)-oldp[1] )
    if unmoved:
        # initialize on first time we get here
        unmoved=0
        tx,ty = 0,0
    else:
        # draw the line showing the path
        tkwindow.polyline([oldp,[oldp[0]+tx,oldp[1]+ty]], style=5, tags=["path"]  )
        tkwindow.canvas.move( objectId, tx,ty )

    prev_tileX, prev_tileY = tile_coords(oldp)
    
    # update the drone position
    oldp = [oldp[0]+tx,oldp[1]+ty]

    curr_tileX, curr_tileY = tile_coords(oldp)
    
    # map drone location back to lat, lon
    # This transforms pixels to WSG84 mapping, to lat,lon
    lat,lon = ts.imagePixelsToLL( actual_pX, actual_pY, zoomLevel,  oldp[0]/(256/scalex), oldp[1]/(256/scaley) )

    # get the image tile for our position, using the lat long we just recovered
    im, foox, fooy, fname = ts.tiles_as_image_from_corr(lat, lon, zoomLevel, 1, 1, 0, 0)
    im.save("tile.png")
    # TODO use the classifier here on the image "im"
    pred, class_name = geoclass.classifyFile(pca, clf, fname, classnames)
    
    # TODO avoid tiles that are urban (if we entered one, go back and make sure we start on non-urban tiles)
    if class_name == "urban":
        # if the current tile is urban, then go back
        tx = -tx
        ty = -ty
        if not brownian and stack:
            # remove this tile from the stack
            stack.pop()
        return
    
    # This is the drone, let's move it around
    tkwindow.canvas.itemconfig(objectId, tag='userball', fill=fill)
    tkwindow.canvas.drawn = objectId

    #  Take the tile and shrink it to go in the right place
    im = im.resize((int(im.size[0]/scalex),int(im.size[1]/scaley)))
    im.save("tmp.gif")
    photo = tk.PhotoImage(file="tmp.gif" )
    tkwindow.image = tkwindow.canvas.create_image( 256/scalex*int(oldp[0]/(256/scalex)), 256/scalex*int(oldp[1]/(256/scalex)), anchor=tk.NW, image=photo, tags=["tile"] )
    image_storage.append( photo ) # need to save to avoid garbage collection

    # This arrenges the stuff being shown
    tkwindow.canvas.lift( objectId )
    tkwindow.canvas.tag_lower( "tile" )
    tkwindow.canvas.tag_lower( "background" )
    tkwindow.canvas.pack()

    # TODO validate that we are appending a valid map here for the GIF
    # keep track of state of the map to generate a gif of the trajectory
    x0 = tkwindow.canvas.winfo_rootx()
    y0 = tkwindow.canvas.winfo_rooty()
    x1 = x0 + tkwindow.canvas.winfo_width()
    y1 = y0 + tkwindow.canvas.winfo_height()
    window_scc = ImageGrab.grab(bbox=(x0,y0,x1,y1))

    # make sure the canvas appeared before storing screenshots
    if x1 - x0 > 1 and y1 - y0 > 1:
        window_scc.save("temp.png")
        maps_storage.append(window_scc)
    
    # coverage algoirthm #1 random Brownian motion which is the baseline
    if brownian:
        # https://scipy-cookbook.readthedocs.io/items/BrownianMotion.html
        # https://ipython-books.github.io/133-simulating-a-brownian-motion/
        
        # randomly generate the duration of each sampled brownian motion
        n_steps = random.randint(1e2,1e6)

        # simulate x and y movements following Brownian motion
        movements = np.random.normal(size=(n_steps - 1, len(oldp))) / np.sqrt(n_steps)

        # scale movements proportional to grid size (e.g. a movement of (1,1), corresponds to (256,256) because every grid cell is (256x256))
        tx, ty = np.cumsum(movements, axis=0)[-1]

        # constant factor to scale up drone's movements
        movement_scale = 10
        
        # times a scalar to make movement bigger (otherwise, drone will stay in the same grid cell forever as brownian motion results in small movements)
        tx *= movement_scale
        ty *= movement_scale
    else:
        # coverage algorithm #2 : DFS
        
        # if this is the second time we visit this tile (means we came back), remove this tile
        if tiles_occ[curr_tileX][curr_tileY]:
            stack.pop()
        else:
            # for each adjacent tile check if its in bounds, if it is add it to stack
            if 0 <= curr_tileX + 1  < tilesX and 0 <= curr_tileY < tilesY:
                stack.append((curr_tileX + 1, curr_tileY))
            if 0 <= curr_tileX - 1  < tilesX and 0 <= curr_tileY < tilesY:
                stack.append((curr_tileX - 1, curr_tileY))
            if 0 <= curr_tileX  < tilesX and 0 <= curr_tileY + 1 < tilesY:
                stack.append((curr_tileX, curr_tileY + 1))
            if 0 <= curr_tileX  < tilesX and 0 <= curr_tileY - 1 < tilesY:
                stack.append((curr_tileX, curr_tileY - 1))

        # take top of stack
        next_tileX, next_tileY = stack[-1]

        # compute distance to get to the middle of target tile
        tx = ((next_tileX + 0.5) * (myImageSize/tilesX)) - oldp[0]
        ty = ((next_tileY + 0.5) * (myImageSize/tilesY)) - oldp[1]
    
    # make sure drone doesn't go out of frame
    if oldp[0] + tx > myImageSize:
        tx = 0
    
    if oldp[1] + ty > myImageSize:
        ty = 0
    
    if oldp[0] + tx < 0:
        tx = 0
    
    if oldp[1] + ty < 0:
        ty = 0

    # account for distance travelled
    total_distance += math.sqrt(tx ** 2 + ty ** 2)
    
    # account for distribution of terrain types visited (only add if this is an unvisited tile)
    terrain_distribution[class_name] += 0 if tiles_occ[curr_tileX][curr_tileY] else 1

    # mark this tile as visited
    tiles_occ[curr_tileX][curr_tileY] = 1

    # count total number of tiles visited
    num_tiles_visited += 0 if prev_tileX == curr_tileX and curr_tileY == prev_tileY else 1


    if total_distance > 20000:
        # write down all the stats and exit

        # generate gifs
        name_suffix = "brownian" if brownian else "custom"
        print("Generating gif")
        imageio.mimsave(f"movie_{name_suffix}.gif", maps_storage)
        os.makedirs("logs", exist_ok=True)
        with open(f"logs/{name_suffix}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}","w") as f:
            f.write(f"total_distance: {total_distance}\ntile ratio: {tiles_occ.sum()/num_tiles_visited}\nnumber of total tiles visited: {num_tiles_visited}\nnum unique tiles: {tiles_occ.sum()}\nterrain distribution: {terrain_distribution}\n")

        exit(0)

def tile_coords(pos):
    """ Given a position in (pixelX, pixelY), return tile coordinates (tileX, tileY) """
    global myImageSize
    return int(pos[0] // (myImageSize/tilesX)), int(pos[1] // (myImageSize/tilesY))

fill = "white"
ts = TileServer.TileServer()

lat, lon = 45.44203, -73.602995    # verdun
tilesX = 20
tilesY = 20
tilesOffsetX = 0
tilesOffsetY = 0
zoomLevel = 18
image_storage = [ ] # list of image objects to avoid memory being disposed of
maps_storage = []

bigpic, actual_pX, actual_pY, fname = ts.tiles_as_image_from_corr(lat, lon, zoomLevel, tilesX, tilesY, tilesOffsetX, tilesOffsetY)
# bigpic is really big
#bigpic.show()

im = bigpic.resize((1024,1024))
im.save("mytemp.gif") # save the image as a GIF and re-load it does to fragile nature of Tk.PhotoImage


import tkinter as tk

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
sx,sy=1000,640 # over the river
# sx,sy = 220,220 # over the canal in Verdun, mixed environment
oldp = [sx,sy]
objectId = tkwindow.canvas.create_oval(int(sx-MYRADIUS),int(sy-MYRADIUS), int(sx+MYRADIUS),int(sy+MYRADIUS),tag=MARK)
unmoved =  1

# global variables
delta_t = 100

# map storing the occupancy of each tile
tiles_occ = np.zeros((tilesX, tilesY))
total_distance = 0
terrain_distribution = {"urban": 0, "water": 0, "arable": 0} # contains unique number of tiles
num_tiles_visited = 1
stack = [] # stack for DFS

# initialize the classifier
pca, clf, classnames = geoclass.loadState("classifier.state", 1.0)

# launch the drawing thread
autodraw()

#Start the GUI
root.mainloop()

