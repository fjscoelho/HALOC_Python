import numpy as np
import cv2 # import open CV for image processing: SIFT features
import numpy as np # mathematic operations

class HALOCGenerator:
    def __init__(self, imgSize=(240,320), num_max_features=100):
        # Compute the orthogonal projection vectors (Once computed are the same for all the process)
        vector1,vector2,vector3=self.__calculatevectors__(num_max_features)
        self.imgSize = imgSize
        self.num_max_features = num_max_features
        self.vector1=vector1
        self.vector2=vector2
        self.vector3=vector3

    def get_descriptors(self, theImage):
        gsImage=cv2.imread(theImage,cv2.IMREAD_GRAYSCALE) # read the image "theImage" from the hard disc in gray scale and loads it as a OpenCV CvMat structure. 
    
        # gsImage=cv2.cvtColor(curImage,cv2.COLOR_BGR2GRAY) # convert image to gray scale
        # plt.figure()
        # plt.imshow(gsImage) # shows image
        # plt.show()
        theSIFT=cv2.SIFT_create((self.num_max_features-3)) # creates a object type SIFT 
        keyPoints,theDescriptors=theSIFT.detectAndCompute(gsImage,None) # keypoint detection and descriptors descriptors, sense mascara
        img = cv2.drawKeypoints(gsImage,keyPoints,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) # uncomment in case you want to see the keypoints
        # plt.figure()
        # plt.imshow(img)
        # plt.show()
        nbr_of_keypoints=len(keyPoints)
        # sanity checks: 
        if nbr_of_keypoints==0:
            print("ERROR: descriptor Matrix is Empty")
            return 
        if nbr_of_keypoints>len(self.vector1):
            print("ERROR:  The number of descriptors is larger than the size of the projection vector. This should not happen.")
            return

        num_of_descriptors=theDescriptors.shape[0] #--> 100
        num_of_components=theDescriptors.shape[1] # --> 128
        hash=[] # initilize hash

        # initialize auxiliar variables
        dot = 0
        dot_normalized=0
        suma = 0

        for i in range(num_of_components):
            suma=0
            for j in range(num_of_descriptors):
                dot = theDescriptors[j,i]*self.vector1[j] # for a fixed component, (a fixed column) vary the descriptor (row): dot product between the matrix column and the vector
                dot_normalized = (dot + 1.0) / 2.0
                suma = suma + dot_normalized
        
        hash=np.append(hash, (suma/num_of_descriptors))   

        for i in range(num_of_components):
            suma=0
            for j in range(num_of_descriptors):
                dot = theDescriptors[j,i]*self.vector2[j] 
                dot_normalized = (dot + 1.0) / 2.0
                suma = suma + dot_normalized
        
        hash=np.append(hash, (suma/num_of_descriptors))   


        for i in range(num_of_components):
            suma=0
            for j in range(num_of_descriptors):
                dot = theDescriptors[j,i]*self.vector3[j] 
                dot_normalized = (dot + 1.0) / 2.0
                suma = suma + dot_normalized
            
            hash=np.append(hash, (suma/num_of_descriptors))   
        return hash


    def __calculatevectors__(self, num_max_features):
        # get the 3 orthogonal unitary vectors

        vector1=np.random.uniform(0,1,num_max_features) # creates a vector of random numbers between 0 and 1
        vector1 /= np.linalg.norm(vector1) # normalize vector 1
        # now create two vectors orthogonal to vector1 and with module = 1
        vector2 = np.random.uniform(0,1,(num_max_features-1)) # second random vector, one less  component 
        const1=0
        long=num_max_features-1
        for i in range(long): # dot product between vector2 and vector 1 for the num_max_features-1 components
            const1=const1+(vector1[i]*vector2[i])

        xn=-const1/vector1[num_max_features-1] # the last component of vector2 and the one that makes vector1·vector2=0
        vector2=np.append(vector2, xn) # add the last component to vector2. Now, vector 1 and vector 2 are orthogonals

        vector2 /= np.linalg.norm(vector2) # normalize vector2 again
        vector3 = np.random.uniform(0,1,(num_max_features-2)) # create another vector , random and unitary


        # vector 3 is orthogonal to vector1 and vector2, forcing all components to be random except the two last, which will result from solving a system of two equations with two variables and 
        # where the scalar product of vector 3 with vector1 and vector2 must be 0 


        const1=0
        const2=0
        long=num_max_features-2
        for i in range(long): # dot product between vector3 and the num_max_features-2 components of vector1 and vector2
            const1=const1+(vector1[i]*vector3[i])
            const2=const2+(vector2[i]*vector3[i])

        # force the last two elements of vector3 to be orthogonal to vector1 and vector2. Solve a linear system of 
        # equations Ax=B, where A --> the last two components of vector1 and vector2, in the form of
        # two rows of A, row 1 = vector1, row 2 = vector2. B is the constant components, taken from the 
        # dot product between the first num_max_features-2 components of vector1 and the num_max_features-2 components of vector2, 
        # with all the random components of vector3. And X are the last two components of vector3, in such a way that
        # vector1 · vector3=0 and vector2 · vector3=0. 
        A = np.array([[vector1[num_max_features-2],vector1[num_max_features-1]], [vector2[num_max_features-2],vector2[num_max_features-1]]])
        B = np.array([-const1,-const2])
        X = np.linalg.solve(A, B) # solve the linear system. X[0] is the penultimate element of vector3 ,X[1] is the last element of vector3
        #np.allclose(np.dot(A, X), B) # true if Ax=B

        vector3=np.append(vector3, X[0]) ## append the last two elements to vector3
        vector3=np.append(vector3, X[1])
        vector3 /= np.linalg.norm(vector3) # normalize vector3

        #  print ("lengh of vector3: "+str(len(vector3)))
        # just visualize some results
        print("Orthogona projection vectors")
        print("||v1|| = "+str(np.linalg.norm(vector1)))
        print("||v1|| = "+str(np.linalg.norm(vector2)))
        print("||v1|| = "+str(np.linalg.norm(vector3)))
        print("\nv1.v2 = "+str(np.dot(vector1, vector2)))
        print("v2.v3 = "+str(np.dot(vector2, vector3)))
        print("v1.v3 = "+str(np.dot(vector1, vector3)))

        return vector1, vector2, vector3