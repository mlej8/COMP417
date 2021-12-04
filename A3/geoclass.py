"""
===================================================
 recognition example 
===================================================

"""




import sys
import os
from time import time
import logging
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
from PIL import Image
from imageio import imread
# module for loading image data
import loader

###############################################################################
# Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
# dataset): unsupervised feature extraction / dimensionality reduction
n_components = 150
n_components = 20
# Use color images
color=1
# enable extra debugging?
debug = 0
#
# Sadly, the image size is embedded here
IMAGE_SIDE=256

###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4, normalize=0, bigtitle=None, color=color):
    """Helper function to plot a gallery of images.
    A default viewer is used. 
    Pillow, by default, searches only for the commands "xv" and "display", the latter one being provided by imagemagick. 
    Therefore, if you install imagemagick, calling "display" will indeed open up the image. 
    To use eog for something els,e see: https://stackoverflow.com/questions/16279441/image-show-wont-display-the-picture
    """
    if debug: print ("Plotting:",bigtitle)
    #plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    f = plt.figure(figsize=(2.1 * n_col, 1.1 * n_row))
    if bigtitle: f.canvas.set_window_title(bigtitle)
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        try: 
            if color:
                if debug: 
                    h,w=IMAGE_SIDE,IMAGE_SIDE
                    print ("Show image:",images[i].shape, "as", images[i].reshape((h, w, 3)).shape )
                    print ("MIN:",np.amin(images[i]),"MAX:",np.amax(images[i]))
                    #im = images[i].reshape((h, w, 3))
                    #im = Image.fromarray(im, 'RGB')
                    #im.show()
                if normalize:
                    low  = np.amin(images[i])
                    high = np.amax(images[i])
                    images[i] = images[i] - low
                    images[i] = images[i]*255.0/(high-low)
                    print ("Rescaled MIN:",np.amin(images[i]),"MAX:",np.amax(images[i]))
                h,w=IMAGE_SIDE,IMAGE_SIDE
                plt.imshow(images[i].reshape((h, w, 3)))# , cmap=plt.cm.seismic)
                #plt.imshow(images[i].reshape((h, w)))# , cmap=plt.cm.seismic)
            else:
                if normalize:
                    plt.imshow(images[i].reshape((h, w)), cmap=plt.get_cmap('gray'))
                else:
                    plt.imshow(images[i].reshape((h, w)), cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
            plt.title(titles[i], size=12)
        except: 
            import traceback
            traceback.print_exc()
            pass
        plt.xticks(())
        plt.yticks(())
    if debug: plt.show()


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


def generate_and_train_classifier(classesDirectory="classes",verbose=1):
    """ Create a PCA-based classifier using a support vector machine (SVM).
        Return the classifier.
        Starts with a directory (i.e. folder) name (classes). In that directory there should be a folder for each class, 
        and each contains some images.
    """
    global X_train, X_test, y_train, y_test
    global target_names, n_classes
    global debug

    if verbose: print(__doc__)
    if verbose>1: debug=1

    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    ###############################################################################
    # Download the data, if not already on disk and load it as numpy arrays

    # See http://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_lfw_people.html
    #lfw_people = fetch_lfw_people(download_if_missing=False, data_home="testset", min_faces_per_person=1, resize=0.4)

    lfw_people = loader.fetch_imagesets(classesDirectory,
                     min_images_per_category=2, resize=1.0, color=color, restrict=None )  
    if debug:
        print ("On return from loader.fetch_imagesets:")
        print ( "lfw_people.keys():",list(lfw_people.keys()) )
        print( "Image[0] shape:", lfw_people.images[0].shape )
        print( "Image[0] data shape:", lfw_people.data[0].shape )
    n = len(lfw_people.images)
    lfw_people.data =  []
    for i in lfw_people.images:
      lfw_people.data.append( np.ravel( i ) )
    lfw_people.data = np.array( lfw_people.data )

    lfw_people.images = np.stack( lfw_people.images, axis=0 )
    #if color:
    #    lfw_people.images = lfw_people.images.reshape((n, IMAGE_SIDE, IMAGE_SIDE))
    #else:
    #    lfw_people.images = lfw_people.images.reshape((n, IMAGE_SIDE, IMAGE_SIDE))

    if debug:
        # SHow first image just to assure everything is OK
        print( lfw_people.target_names[0] )
        plot_gallery( [lfw_people.data[0]], [lfw_people.target_names[0]] , IMAGE_SIDE, IMAGE_SIDE, 
            n_row=1, n_col=1, normalize=0, bigtitle="First image DATA", color=color)


    # introspect the images arrays to find the shapes (for plotting)
    if color:
        n_samples, h, w, colordepth = lfw_people.images.shape
    else:
        n_samples, h, w = lfw_people.images.shape
    if debug: print ("lfw_people.images.shape n_samples, h, w:",lfw_people.images.shape)

    # for machine learning we use the data directly  -- one long 1D vector (as relative pixel
    # positions info is ignored by this model)
    X = lfw_people.data
    n_features = X.shape[1]

    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print ("Target names:",target_names)
    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)


    ###############################################################################
    # Split into a training set and a test set using a stratified k fold

    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.5, random_state=42)
    if verbose: print("Extracting the top %d basis functions from %d images" % (n_components, X_train.shape[0]))
    if debug: print("Shape of training set is:",X_train.shape)

    t0 = time()
    #pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True).fit(X_train)
    pca = PCA(n_components=n_components, svd_solver='auto', whiten=True).fit(X_train)
    if verbose: print("done in %0.3fs" % (time() - t0))

    explained_variance = np.cumsum(pca.explained_variance_ratio_)
    n = sum(explained_variance <= 0.80)
    if verbose: print ("With",n,"basis functions we capture 80% of the variance.")
    if verbose: print ("Means:",pca.mean_)

    # do pca and convert vectors into image-shaped hunks of data
    if color:
        eigenfaces = pca.components_.reshape((n_components, h, w, 3))
    else:
        eigenfaces = pca.components_.reshape((n_components, h, w))

    t0 = time()
    X_train_pca = pca.transform(X_train)
    if n_classes>1:
        X_test_pca = pca.transform(X_test)
    if verbose: print("done in %0.3fs" % (time() - t0))

    # Show the data
    recon = X_train[0:36]
    txt = [ target_names[y_train[i]].rsplit(' ', 1)[-1] for i in range(len(X_train[0:36]))]
    if verbose: plot_gallery(recon, txt, h, w, n_row=6, n_col=6, bigtitle="Raw Training data", color=color)

    # Show the reconstructions of some of the data
    recon = pca.inverse_transform(X_train_pca[0:36])
    if verbose: print ("Reconstructions:",w,h,recon[0].shape, recon[0].reshape(w,h,3).shape )
    txt = [ target_names[y_test[i]].rsplit(' ', 1)[-1] for i in range(len(X_train_pca[0:36]))]
    Z = recon[0].reshape(w,h,3)
    if verbose:
        plot_gallery(Z, ["Z"], h, w, n_row=1, n_col=1,  bigtitle="Reconstructions of training data", normalize=0, color=color )
        plt.show()
        plot_gallery(recon, txt, h, w, n_row=6, n_col=6,  bigtitle="Reconstructions of training data", normalize=0, color=color )
        plt.show()

    ###############################################################################
    # Train a SVM classification model

    if n_classes>1:
        if debug: print("Fitting the classifier to the training set")
        t0 = time()
        param_grid = {'C': [1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        #clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
        clf = GridSearchCV(SVC(kernel='linear' ), param_grid)
        clf = clf.fit(X_train_pca, y_train)
        if verbose:
            print("done in %0.3fs" % (time() - t0))
            print("Best estimator found by grid search:")
            print(clf.best_estimator_)


    ###############################################################################
    # Quantitative evaluation of the model quality on the test set

    if n_classes>1:
        if verbose: print("Predicting class names on the test set")
        t0 = time()
        y_pred = clf.predict(X_test_pca)
        if verbose: print("done in %0.3fs" % (time() - t0))
        print(classification_report(y_test, y_pred, target_names=target_names))
        if verbose: print(confusion_matrix(y_test, y_pred, labels=list(range(n_classes))))
        prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
        if verbose: plot_gallery(X_test, prediction_titles, h, w, bigtitle="Predictions", color=color)

        #y_pred = clf.predict(X_train_pca)
        #prediction_titles = [title(y_pred, y_train, target_names, i) for i in range(y_pred.shape[0])]
        #plot_gallery(X_train, prediction_titles, h, w, n_row=8, n_col=1+len(X_train_pca)/8 )

        # plot the gallery of the most significative eigenfaces

        eigenface_titles = ["basis %d" % i for i in range(eigenfaces.shape[0])]
        if verbose: plot_gallery(eigenfaces, eigenface_titles, h, w, normalize=0, bigtitle="Basis functions", color=color)
    if verbose: plt.show()

    # import deeper
    # print( "SIZES:",deeper.deep_getsizeof(pca,set()), deeper.deep_getsizeof(clf,set()), deeper.deep_getsizeof(target_names,set()))
    print(pca,set(), clf,set(), target_names,set())
    return pca, clf,target_names

def classifyOne(pca, clf, patch, target_names=[""]*10):
    """ Give a pca dimensionality-reduction eigenspace and a SVM classifier for it, use that to
    classify a single image patch. """
    X_test_pca = pca.transform([patch])
    y_pred = clf.predict(X_test_pca)
    txt = target_names[y_pred[0]]
    print (txt)
    return y_pred[0],txt

def classifyFile(pca, clf, filePath, target_names=[""]*10):
    """ Give a pca dimensionality-reduction eigenspace and a SVM classifier for it, use that to
    classify a single image patch. """
    im = imread(filePath)
    im = np.asarray(im, dtype=np.float32)
    im = np.ravel( im )
    return classifyOne(pca, clf, im,  classnames)


import pickle

def saveState(saveStateFile, pca, clf, classnames, versionNumber ):
    """ Save a bunch of global variables in a pickle, for faster restarting """
    print ("Saving state")
    output = open(saveStateFile, 'wb')
    checkFirst = "first"
    checkLast = "last"

    if debug: 
        print ("  pickle","checkfirst")
    pickle.dump(eval("checkFirst"), output)
    # variables to save
    savelist = [ "pca", "clf", "classnames", "checkLast" ]
    if debug: 
        print ("  pickle","savelist")

    pickle.dump(eval("versionNumber"), output)
    pickle.dump( ["checkFirst","versionNumber","savelist",] + savelist, output) # save list of what else we are saving

    for i in savelist:
        if debug: print ("  pickle",i)
        pickle.dump(eval(i), output)
        print("Object",i,"dumped size",len(pickle.dumps(eval(i))))
    output.close()
    if debug: print ("Saved state.")

def loadState(loadStateFile, versionNumber):
    print ("Loading saved state from",loadStateFile)
    savedstate = open(loadStateFile, 'rb')
    first = pickle.load(savedstate)
    savedVersion = pickle.load(savedstate) # we can do version-dependent loading
    print ("savedVersion is",savedVersion)
    if (versionNumber!=savedVersion): print ("State file version mismatch. Prepare for disaster.")

    savelist = pickle.load(savedstate)
    print ("loading",savelist)

    pca = pickle.load(savedstate)
    clf = pickle.load(savedstate)
    classnames = pickle.load(savedstate)

    last = pickle.load(savedstate)
    if (first != "first") or (last != "last"):
        print ("*** ERROR: load from pickle was bad ***", first,last)
        sys.exit(1)
    return pca, clf,classnames



if __name__ == '__main__':
    versionNumber=1.0
    if not os.access("classifier.state",os.R_OK):
        # no saved state, mst re-train
        pca, clf,classnames = generate_and_train_classifier("classes",verbose=0)
        print ("ORIGINAL CLASSIFY ONE RESULT in X_test[0]:",classifyOne(pca, clf, X_test[0], classnames))
        saveState("classifier.state", pca, clf, classnames, versionNumber )

    pca, clf, classnames = loadState("classifier.state", versionNumber)
    print (classifyFile(pca, clf, "tiles/18_77475_93848_a.jpg", classnames))
