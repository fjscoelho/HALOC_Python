
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
QRdebug = True

#############################################################

print('********** LOADING DATASET ***********')
dataSet1=DataSet('DATASETS/DATASET_LABCON.TXT')

queryIndex= 0
k = 3                  # number of projection directions

#select one query
query_image, qFileName = dataSet1.get_qimage(queryIndex) # the global  path of the query image
shape = query_image.shape
size = shape[0]*shape[1]
num_max_features = 0.05*size # define the maximum number of features = 5% of length in pixels
num_max_features = 100
print('num_max_features = '+ str(num_max_features))

Haloc= HALOCGenerator(num_max_features, k, QRdebug); #create an Haloc object (define the orthogonal vectors to projections)


hash_query = Haloc.get_descriptors(qFileName, True) # get the query hash

n = dataSet1.numQImages
l = len(hash_query)

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