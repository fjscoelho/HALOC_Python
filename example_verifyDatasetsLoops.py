
from dataset import DataSet
import matplotlib.pyplot as plt
import sys


# Dataset LABCON
dataSetPath = 'DATASETS/DATASET_LABCON.TXT'

dataSetName =  dataSetPath.split('/')
dataSetName = dataSetName[-1]
dataSetName = dataSetName.split('.')
dataSetName = dataSetName[0]

# Load dataset
print('[[ LOADING DATASETS ]]')
dataSet=DataSet(dataSetPath)
print('[[DATASETS LOADED ]]\n\n')

# Let's print the dataSet info
print('[[ PRINTINT '+ dataSetName +' INFO ]]')
dataSet.print()
print('[[ '+ dataSetName +' PRINTED ]]\n\n')

print('[[ PLOTTING ALL LOOPS]]')

# Plot all loops
loopCount = 0
for curLoop in dataSet.theLoops:
    dbFileName, qFileName = dataSet.get_loop(loopCount)
    loopCount +=1 
    plt.figure(loopCount)
    plt.figtext(0.5, 0.7, 'Loop '+ str(loopCount), ha = "center")
    plt.subplot(1,2,1)
    # Query image
    plt.imshow(qFileName)

    plt.subplot(1,2,2)
    # DataBase image
    plt.imshow(dbFileName)
    plt.figtext(0.5, 0.2, 'Q: '+ dataSet.qImageFns[curLoop[1]] + '      DB: '+dataSet.dbImageFns[curLoop[0]], ha="center")
    # images[loopCount-1] = plt.figure(loopCount)
    plt.show()

# print('[[ PLOT DONE ]]')