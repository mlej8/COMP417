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

import TileServer

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

ts = TileServer.TileServer()

lat, lon = 45.44203, -73.602995    # verdun
tilesX = 8
tilesY = 8
tilesOffsetX = 0
tilesOffsetY = 0
zoomLevel = 18

bigpic, actual_pX, actual_pY, fname = ts.tiles_as_image_from_corr(lat, lon, zoomLevel, tilesX, tilesY, tilesOffsetX, tilesOffsetY)
# bigpic is really big
#bigpic.show()

im = bigpic.resize((1024,1024))
im.save("mytemp.gif") # save the image as a GIF and re-load it does to fragile nature of Tk.PhotoImage


import tkinter as tk
root = tk.Tk()
root.title("display an image")
photo = tk.PhotoImage(file="mytemp.gif")
#The Label widget is a standard Tkinter widget used to display a text or image on the screen.
panel = tk.Label(root, image = photo)
panel.img = photo

#The Pack geometry manager packs widgets in rows or columns.
panel.pack(side = "bottom", fill = "both", expand = "yes")

#Start the GUI
root.mainloop()


"""
while 1:
    try: tkwindow.events()
    except:
        tkwindow.terminated()
        break

"""


