import numpy as np
import cv2 # import open CV for image processing: SIFT features
import numpy as np # mathematic operations
import matplotlib.pyplot as plt
from skimage.io import imread

class HALOCGenerator:
    def __init__(self, num_max_features=100, k=3, QRdebug = False):
        # Compute the orthogonal projection vectors (Once computed are the same for all the process)
        vectors=self.__calculatevectors__(num_max_features,k, QRdebug)
        # self.imgSize = imgSize
        self.num_max_features = num_max_features
        self.vectors = vectors
        self.k = k

    def get_descriptors(self, theImage, debug = False):
        curImage = cv2.imread(theImage, cv2.IMREAD_COLOR) # read the image "theImage" from the hard disc in gray scale and loads it as a OpenCV CvMat structure. 
        resized_image = cv2.resize(curImage, (320, 240), interpolation=cv2.INTER_AREA)
        gsImage = cv2.cvtColor(resized_image,cv2.COLOR_BGR2GRAY) # convert image to gray scale

         # creates a object type SIFT 
        theSIFT=cv2.SIFT_create((self.num_max_features-3)) # sometimes the number of Keypoints is larger than num_max_features
        keyPoints,theDescriptors=theSIFT.detectAndCompute(gsImage,None) # keypoint detection and descriptors descriptors, sense mascara
        
        # sanity checks:
        nbr_of_keypoints=len(keyPoints)
        if nbr_of_keypoints==0:
            print("ERROR: descriptor Matrix is Empty")
            return 
        if nbr_of_keypoints>self.num_max_features:
            print("ERROR:  The number of descriptors is larger than the size of the projection vector. This should not happen.")
            return
        
        if debug:
            img = cv2.drawKeypoints(gsImage,keyPoints,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # uncomment in case you want to see the keypoints
            plt.figure()
            plt.subplot(1,2,1)
            plt.imshow(curImage) # shows image
            plt.title("cv2 color image")
            # plt.show()
            plt.subplot(1,2,2)
            plt.imshow(img)
            plt.title("nº keypoints = "+ str(nbr_of_keypoints))
            plt.show()
        
        num_of_descriptors=theDescriptors.shape[0] #--> 100
        num_of_components=theDescriptors.shape[1] # --> 128
        hash=[] # initilize hash

        if debug:
            print("Nº de descritores = "+ str(num_of_descriptors))
            print("Nº de componentes = "+ str(num_of_components))
        # initialize auxiliar variables
        dot = 0
        dot_normalized=0
        suma = 0
        
        # for j in range(num_of_components):
        #     # print("Antes da normalização:")
        #         # print(theDescriptors[:,i])
        #         # print("Norma = "+str(np.linalg.norm(theDescriptors[:,i])))
                
        #         theDescriptors[:,j] = theDescriptors[:,j]/np.linalg.norm(theDescriptors[:,j])

        #         # print("Antes da normalização:")
        #         # print(theDescriptors[:,i])
        #         # print("Norma = "+str(np.linalg.norm(theDescriptors[:,i])))

        for l in range(self.k): # for each ul projection vector
            for i in range(num_of_components):
                suma=0
                for j in range(num_of_descriptors):
                    dot = theDescriptors[j,i]*self.vectors[j,l] # for a fixed component, (a fixed column) vary the descriptor (row): dot product between the matrix column and the vector
                    dot_normalized = (dot + 1.0) / 2.0
                    suma = suma + dot_normalized
                    # suma = suma + dot
            
                hash=np.append(hash, (suma/num_of_descriptors))   

        if debug:
            print("Tamanho do hash= "+ str(len(hash)))
        return hash


    def __calculatevectors__(self, num_max_features, k, QRdebug):
        # get the k orthogonal unitary vectors using QR factorization

        A = np.random.uniform(0,1,(num_max_features,k)) # It's important to keep the random numbers between 0 and 1
        Q, R = np.linalg.qr(A)
        
        if QRdebug:
            # Print the results
            print("Matrix Q:")
            print(Q)

            print("\nMatrix R:")
            # print(R)

            # Verify the unit norm
            for i in range(k):
                print("norm q"+str(i+1)+" = "+str(np.linalg.norm(Q[:,i])))

            # Verify the dot products between columns
            for i in range(k):
                for j in range(i+1,k):
                    print("dot product: q"+str(i+1)+"q"+str(j+1)+" = "+str(np.dot(Q[:,i],Q[:,j])))

        return Q