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



def imshow(theImage):
    plt.figure()
    img = cv2.imread(theImage,cv2.IMREAD_COLOR)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    

#############################################################3
qPath='/home/fabio/NetHALOC/HALOC/HALOC_Python/DATASETS/QUERY/' # write here the global path of your queries
dBPath='/home/fabio/NetHALOC/HALOC/DATASETS/HALOC_Python/DATABASE/' # write here the global path of the corresponding database of images

num_max_features = 100 # define the maximum number of features
Haloc= HALOCGenerator(num_max_features); #create an Haloc object (define the orthogonal vectors to projections)

#select one query
query_image=os.path.join(qPath,'4.jpg') # the global  path of the query image
imshow(query_image)
hash_query=Haloc.get_descriptors(query_image) # get the query hash

allFiles=[x for x in os.listdir(dBPath) if x.upper().endswith('.JPG')] ## list of images of the db directory. Assume that all are JPG, change if they are png or others


distance_matrix=np.zeros((0,2),dtype='S20, f4')


# for i in range(len(allFiles)): # fron 0 to len(allFiles)-1 --> search for all images in the database
#     candidate_image=os.path.join(dBPath,allFiles[i]) # get candidate
#     hash_candidate= get_descriptors(candidate_image,num_max_features) # get hash of candidate
#     dist = np.linalg.norm((hash_candidate-hash_query), ord=1) # compute l1 norm between hashes
#     distance_matrix = np.append(distance_matrix, np.array([(allFiles[i], dist)], dtype='S20, f4')) # append candidate names and distances into a matrix
    

# np.sort(distance_matrix.view('S20,f4'), order=['f1'], axis=0) # The sort matrix of distances by distances. This is the list of images in the database with the distance between the query hash and the 
# database image hash.