import numpy as np

k = 5   # number of ortogonal directions

A = np.random.uniform(0,1,(100,k))

Q, R = np.linalg.qr(A)

# Print the results
print("Matrix Q:")
print(Q)

print("\nMatrix R:")
print(R)

# Verify the unit norm
for i in range(k):
    print("norm q"+str(i+1)+" = "+str(np.linalg.norm(Q[:,i])))

# Verify the dot products between columns
for i in range(k):
    for j in range(i+1,k):
        print("dot product: q"+str(i+1)+"q"+str(j+1)+" \
              = "+str(np.dot(Q[:,i],Q[:,j])))


