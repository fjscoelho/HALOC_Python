####################################################################################################################################################################################
# Author	: Francisco Bonin Font
# History     : 27-June-2019 - Creation
# NOTES:
#	1. This code runs fine with python 3.7.3. inside an Anaconda environment (https://www.anaconda.com/distribution/?gclid=EAIaIQobChMImM39x9Ow5gIVhYxRCh3CXgvnEAAYASAAEgI91vD_BwE). 
# 	2. Not tested on other versions of python.
# 	3. You will need to install OpenCv for python: pip install opencv-python 
# 	4. Take into account that , if you are using Jupyter Notebook, it is necessary to run, first: import sys
# 	5.if you have installed ROS (Robot Operating System) Kinetic or any other ROS distribution, first you will need to deactivate the Python lybraries installed with ROS:  
# 			sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
#####################################################################################################################################################################################


import matplotlib.pyplot as plt
import cv2 # import open CV for image processing: SIFT features
import numpy as np # mathematic operations
import os # utility for access to directories.
from dataset import DataSet # import the Dataset management class 
from descriptorgenerator import HALOCGenerator
from scipy.spatial.distance import cdist
from imagematcher import ImageMatcher
import time

# change this flag to True if you ant to plot the loop candidates
debugPlot = False

#############################################################
qPath='/home/fabio/NetHALOC/HALOC/HALOC_Python/DATASETS/QUERY/' # write here the global path of your queries
dBPath='/home/fabio/NetHALOC/HALOC/HALOC_Python/DATASETS/DATABASE/' # write here the global path of the corresponding database of images

print('********** LOADING DATASET ***********')
dataSet1=DataSet('DATASETS/DATASET1.TXT')

num_max_features = 50 # define the maximum number of features
imgSize = (240,320)    # define size images
queryIndex= 5

Haloc= HALOCGenerator(num_max_features); #create an Haloc object (define the orthogonal vectors to projections)

# just visualize some results
print("Orthogonal projection vectors")
print("||v1|| = "+str(np.linalg.norm(Haloc.vector1)))
print("||v2|| = "+str(np.linalg.norm(Haloc.vector2)))
print("||v3|| = "+str(np.linalg.norm(Haloc.vector3)))
print("\nv1.v2 = "+str(np.dot(Haloc.vector1, Haloc.vector2)))
print("v2.v3 = "+str(np.dot(Haloc.vector2, Haloc.vector3)))
print("v1.v3 = "+str(np.dot(Haloc.vector1, Haloc.vector3)))

#select one query
query_image, qFileName = dataSet1.get_qimage(queryIndex) # the global  path of the query image
hash_query = Haloc.get_descriptors(qFileName) # get the query hash


qHashs = np.zeros((dataSet1.numQImages,len(hash_query)))   # initialize array of hash for all dataset query images
dbHashs = np.zeros((dataSet1.numDBImages,len(hash_query)))   # initialize array of hash for all dataset db images

print('COMPUTING ALL DATASET IMAGE HASHES')
t_initial = time.time()
for i in range(dataSet1.numQImages):
    q_image, qFileName = dataSet1.get_qimage(i)      # get a query image
    hash_qimage = Haloc.get_descriptors(qFileName) # get hash of candidate
    qHashs[i] = hash_qimage

for i in range(dataSet1.numDBImages):
    db_image, dbFileName = dataSet1.get_dbimage(i)     # get a DB image
    hash_candidate = Haloc.get_descriptors(dbFileName) # get hash of DB image
    dbHashs[i] = hash_candidate
t_hashes = time.time() - t_initial
print('Time to compute '+ str(dataSet1.numQImages + dataSet1.numDBImages)+ ' image hashes = '+ str(t_hashes) + ' s')

print('COMPUTING DISTANCES BETWEEN DESCRIPTORS')
t_initial = time.time()
distanceType = 'cityblock' # cityblock = Manhattan (L1 - norm)

# return a matrix (Returns a matrix of shape (m, p) where each element [i, j] is the distance between
#  the ith query hash and the jth db hash.) 
theDistances = cdist(qHashs,dbHashs,distanceType)
t_distances = time.time() - t_initial
print('Time to compute '+ str(dataSet1.numQImages*dataSet1.numDBImages)+ ' hash distances = '+ str(t_distances) + ' s')

tp=tn=fp=fn=0
nItems=5
contaux = 0
findloop  = 0 
print('THE CLOSEST '+str(nItems)+' DATABASE IMAGES WILL BE SEARCHED')

for qIndex in range(dataSet1.numQImages):
    dbLoopCandidates = np.argsort(theDistances[qIndex,:])[:nItems] # get nItems with shortest distance to query
    qImage, qFileName  = dataSet1.get_qimage(qIndex) # retrieve the query image
    dbActualLoops = dataSet1.get_qloop(qIndex) # search for all dB images that contain a loop with the qIndex query

    # count the actual loops found in candidates
    for i in range(len(dbLoopCandidates)):
        if dbLoopCandidates[i] in dbActualLoops:
            findloop +=1

    if debugPlot:
        # Plot the query and the 5 images. For each of these 5 images, state if it is
        # an actual loop or not.
        # plt.close('all')
        plt.figure(2*qIndex+1)
        plt.subplot(2,3,1)
        # Plot the query
        plt.imshow(qImage)
        plt.title('QUERY: (i = '+str(qIndex)+') '+qFileName)
        # Plot each of the selected database images
        for i in range(5):
            plt.subplot(2,3,i+2)
            dbImage, dbFileName = dataSet1.get_dbimage(dbLoopCandidates[i])
            plt.imshow(dbImage)
            # If the image is an actual loop (a true positive), show the message
            # "ACTUAL LOOP".
            if dbLoopCandidates[i] in dbActualLoops:
                plt.title('ACTUAL LOOP: (i = '+str(dbLoopCandidates[i])+') '+dbFileName)
        plt.show()

        # Plot the query and its actual loop images
        plt.figure(2*qIndex+2)
        plt.subplot(2,3,1)
        plt.title('QUERY: (i = '+str(qIndex)+') '+qFileName)
        plt.imshow(qImage)
        for i in range(len(dbActualLoops)):
            plt.subplot(2,3,i+2)
            dbImage, dbFileName = dataSet1.get_dbimage(dbActualLoops[i])
            plt.imshow(dbImage)
            plt.title('ACTUAL LOOP: (i = '+str(dbActualLoops[i])+') '+dbFileName)
        plt.show()

print(str(findloop)+' loops de um total de '+str(dataSet1.numLoops)+' = '+str(findloop/dataSet1.numLoops))