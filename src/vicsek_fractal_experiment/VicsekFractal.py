import tensorflow as tf
print(tf.__version__)
import matplotlib.pyplot as plt
import numpy as np

#TF implementation of N-flake (Vicsek fractal)
#Process Coords via tensors
#plot white squares based on coords
fig, ax = plt.subplots()
from matplotlib.patches import Rectangle

#Initial blue square
ax.add_patch(Rectangle((0, 0), 1, 1,
             facecolor = 'blue',
             ))

#Determine number of tierations
maxDepth = 3

def vicsek(rectStartPoint,rectSize, depth):
    if (depth < maxDepth):
        #print(rectStartPoint)
        newSize = 1/3 * rectSize
        newCoordsX = rectStartPoint[0].numpy()
        newCoordsY = rectStartPoint[1].numpy()
        firstOffset = rectSize * 1/3
        secondOffset = rectSize * 2/3
        initPoints = tf.constant([newCoordsX, newCoordsY], dtype=float)
        
        #Is not intrinsically parallelisable.
        #Can use parallelism in initalising vectors (vectorisation) and
        #adding their offsets to each corner in parallel
        whiteOffSet = tf.constant([[0, 0], 
                                [secondOffset, 0],
                                [0,secondOffset],
                                [secondOffset,secondOffset]])
        #Add offsets in parallel based on previous iteration/recursive call
        newWhiteSquareCoords = tf.math.add(whiteOffSet, initPoints)
        
        #Initialise tensor for blue square coordinates
        blueSquareOffset = tf.constant([
                                   [0, firstOffset],
                                   [firstOffset, 0],
                                   [secondOffset, firstOffset],
                                   [firstOffset, secondOffset],
                                    [firstOffset, firstOffset]
                                   ])
        #Add offsets in parallel based on previous iteration/recursive call
        newBlueSquareCoords = tf.math.add(blueSquareOffset, initPoints)
        
        #Whiteout corners of blue square
        ax.add_patch(Rectangle(newWhiteSquareCoords[0], newSize, newSize, facecolor='white'))
        ax.add_patch(Rectangle(newWhiteSquareCoords[1], newSize, newSize, facecolor='white'))
        ax.add_patch(Rectangle(newWhiteSquareCoords[2], newSize, newSize, facecolor='white'))
        ax.add_patch(Rectangle(newWhiteSquareCoords[3], newSize, newSize, facecolor='white'))
        
        #Recurse for each blue square
        vicsek(newBlueSquareCoords[0], newSize,  depth + 1)
        vicsek(newBlueSquareCoords[1], newSize, depth + 1)
        vicsek(newBlueSquareCoords[2], newSize,  depth + 1)
        vicsek(newBlueSquareCoords[3], newSize,  depth + 1)
        vicsek(newBlueSquareCoords[4], newSize,  depth + 1)

        
initStartingPoints = tf.constant([0, 0])
print(initStartingPoints)
initSize = 1
vicsek(initStartingPoints, 1, 0)
plt.show()
