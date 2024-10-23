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
dataFields = qFileName.split('/')
qshortName = dataFields[-1]
plt.imshow(query_image)
plt.title('QUERY')
plt.show()

hash_query = Haloc.get_descriptors(qFileName,True) # get the query hash

# allFiles=[x for x in os.listdir(dBPath) if x.upper().endswith('.JPG')] ## list of images of the db directory. Assume that all are JPG, change if they are png or others

# Get the database image index that actually close loop with the used query.
actualLoops=dataSet1.get_qloop(queryIndex)
distance_matrix=np.zeros((0,2),dtype='S20, f4')

plt.figure()
plt.title('ACTUAL LOOPS')
print("\n Distances:")
for i in range(len(actualLoops)):
    loop_image, lFileName = dataSet1.get_dbimage(actualLoops[i])
    hash_loop = Haloc.get_descriptors(lFileName) # get the query hash
    dist = np.linalg.norm((hash_loop-hash_query), ord=1) # compute l1 norm between hashes
    dataFields = lFileName.split('/')
    lshortName = dataFields[-1]
    distance_matrix = np.append(distance_matrix, np.array([(lshortName, dist)], dtype='S20, f4'))
    plt.subplot(1,2,i+1)
    plt.imshow(loop_image)
    print('l1-norm between '+ lFileName +' and '+ qshortName + '= ' +str(dist))
plt.show()

nLoopIndex = 12

plt.figure()
plt.title('NOT LOOP')
nloop_image, nlFileName = dataSet1.get_dbimage(nLoopIndex)
hash_nloop = Haloc.get_descriptors(nlFileName, True) # get the query hash
dist = np.linalg.norm((hash_nloop-hash_query), ord=1) # compute l1 norm between hashes
print('l1-norm between '+ nlFileName +' and '+ qshortName + '= ' +str(dist))
plt.imshow(nloop_image)
plt.show()

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
distanceType = 'euclidean'

# return a matrix (Returns a matrix of shape (m, p) where each element [i, j] is the distance between
#  the ith query hash and the jth db hash.) 
theDistances = cdist(qHashs,dbHashs,distanceType)
t_distances = time.time() - t_initial
print('Time to compute '+ str(dataSet1.numQImages*dataSet1.numDBImages)+ ' hash distances = '+ str(t_distances) + ' s')

tp=tn=fp=fn=0
nItems=5
contaux = 0
print('THE CLOSEST '+str(nItems)+' DATABASE IMAGES WILL BE SEARCHED')

theMatcher = ImageMatcher()
t_start = time.time()
for qIndex in range(dataSet1.numQImages):
    dbLoopCandidates = np.argsort(theDistances[qIndex,:])[:nItems] # get nItems with shortest distance to query
    qImage, qFileName  = dataSet1.get_qimage(qIndex) # retrieve the query image
    dbActualLoops = dataSet1.get_qloop(qIndex) # search for all dB images that contain a loop with the qIndex query
    
    for dbIndex in range(dataSet1.numDBImages): # For all database images
        isLoop=dbIndex in dbActualLoops # See if the dBIndex is identical to the set of images that contain loops
        foundLoop=False
        if dbIndex in dbLoopCandidates: # If the DbIndex image is in the five 5 loop candidates
            dbImage, dbFileName=dataSet1.get_dbimage(dbIndex)
            theMatcher.define_images(qImage,dbImage)
            foundLoop=theMatcher.estimate() # Verify if there is a loop closing RANSAC between query and dB
            if isLoop:
                contaux+=1
            # foundLoop = True
            del dbImage
        if foundLoop and isLoop:
            tp+=1   # estava dentro dos candidatos a loop e o RANSAC também confirmou
        elif foundLoop and (not isLoop):
            fp+=1   # o loop real não estava entre os candidatos e o RANSAC confirmou uma imagem erroneamente
        elif (not foundLoop) and isLoop:
            fn+=1   # é um loop real, mas não entrou nos candidatos e/ou o RANSAC não confirmou
        elif (not foundLoop) and (not isLoop):
            tn+=1
    print('    + COMPLETED '+str(qIndex+1)+' OF '+str(dataSet1.numQImages)+' QUERIES')
    del qImage

tloops=time.time()-t_start
print('TESTS FINISHED')
theAccuracy=(tp+tn)/(tp+tn+fp+fn)
theTPR=tp/(tp+fn)
theFPR=fp/(fp+tn)
print('Loops perdidos pelo RANSAC = ' +str(contaux - tp)) # Quantos vezes uma imagem que forma loop estava entre os cadidatos
# e o RANSAC não confirmou
print('[TP, FP, FN, TN '+str(tp)+' '+str(fp)+' '+str(fn)+' '+str(tn)+']')
print('[FULLSTATS COMPUTED '+str(theAccuracy)+' '+str(theTPR)+' '+str(theFPR)+' '+str(tloops)+']')

# distance_matrix = np.append(distance_matrix, np.array([(allFiles[i], dist)], dtype='S20, f4')) # append candidate names and distances into a matrix


# for i in range(len(allFiles)): # fron 0 to len(allFiles)-1 --> search for all images in the database
#     candidate_image=os.path.join(dBPath,allFiles[i]) # get candidate
#     hash_candidate= get_descriptors(candidate_image,num_max_features) # get hash of candidate
#     dist = np.linalg.norm((hash_candidate-hash_query), ord=1) # compute l1 norm between hashes
#     distance_matrix = np.append(distance_matrix, np.array([(allFiles[i], dist)], dtype='S20, f4')) # append candidate names and distances into a matrix
    

# np.sort(distance_matrix.view('S20,f4'), order=['f1'], axis=0) # The sort matrix of distances by distances. This is the list of images in the database with the distance between the query hash and the 
# database image hash.