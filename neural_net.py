
# Author : Prashant Srivastava
# Dated  : April 3rd, 2018
# Based on Neural Net in C++ Tutorial by David Miller (https://www.youtube.com/watch?v=KkwX7FkLfug)

import math
import random

class Connection(object):
    def __init__(self, weight ):
        self.weight = weight
        self.deltaWeight = 0.0
    
class Neuron(object):
    eta    = 0.15 #overall net training rate
    alpha  = 0.5  #multiplier of last weight change
    
    def transferFunction( self, x ):
        return math.tanh( x )
    
    def transferFunctionDerivative(self,x):
        return 1.0 - x*x ;
    
    def randomWeight(self):
        return random.random()
    
    def __init__(self, numOutputs, myIndex ):
        self.outputVal = 0
        self.outputWeights = [ ]
        self.gradient = 0.0
        
        for c in range(0, numOutputs ):
            self.outputWeights.append( Connection( self.randomWeight() ) )

        self.myIndex = myIndex

        
    def setOutputval( self, val ):
        self.outputVal = val

        
    def getOutputVal( self ):
        return self.outputVal
    
    def feedForward( self, prevLayer ):
        sumN = 0.0
        for n,layer in enumerate(prevLayer):
            sumN += layer.getOutputVal() * layer.outputWeights[self.myIndex].weight

        self.outputVal = self.transferFunction(sumN);
    
    def calcOutputGradients( self, targetVals ):
        delta = targetVals - self.outputVal
        self.gradient = delta * self.transferFunctionDerivative(self.outputVal)

    def calcHiddenGradients( self, nextlayer ):
        dow = self.sumDOW(nextlayer)
        self.gradient = dow * self.transferFunctionDerivative(self.outputVal);
	
    def updateInputWeights(self, prevlayer ):
        for n in range(len(prevlayer)):
            oldDeltaWeight = prevlayer[n].outputWeights[self.myIndex].deltaWeight
            newDeltaWeight = prevlayer[n].eta * prevlayer[n].getOutputVal() * self.gradient # Individual input, magnified by the gradient and train rate
            # Also add momentum = a fraction of the previous delta weight
            + Neuron.alpha * oldDeltaWeight

            prevlayer[n].outputWeights[self.myIndex].deltaWeight = newDeltaWeight
            prevlayer[n].outputWeights[self.myIndex].weight += newDeltaWeight
            

    def sumDOW(self, nextlayer):
        sumN = 0.0
        for n in range(len(nextlayer)-1):
            sumN += self.outputWeights[n].weight * nextlayer[n].gradient
        return sumN

    
        

class Net(object):
    recentAverageSmoothingFactor = 100.0
    
    def __init__(self, topology):
        numberLayers = len(topology)
        self.layers = []
        self.recentAverageError = 0.0
        self.error = 0.0
        
        for layerNum in range(len(topology)):
            self.layers.append( [] )
            if layerNum == len(topology) - 1 :
                numOutputs  = 0
            else :
                numOutputs = topology[layerNum + 1]

            for neuronNum in range( 0, topology[layerNum] + 1 ):
                self.layers[-1].append( Neuron( numOutputs, neuronNum ) )
                print( 'Neuron Created !' )

            self.layers[-1][-1].setOutputval( 1.0 )
            

    def getRecentAverageError(self) :
        return self.recentAverageError
    
    def feedForward(self, inputVals):
        assert(len(inputVals) == len(self.layers[0]) - 1)

        for i in range(len(inputVals)):
            self.layers[0][i].setOutputval(inputVals[i])

        for layerNum in range (1, len(self.layers) ):
            prevLayer = self.layers[layerNum-1]
            for n in range( len(self.layers[layerNum]) -1 ):
                self.layers[layerNum][n].feedForward(prevLayer)
    
        
        
    def backProp(self, targetVals):
        outputLayer = self.layers[-1]
        self.error = 0.0

        for n in range(len( self.layers[-1]) -1):
            delta = targetVals[n] - outputLayer[n].getOutputVal()
            self.error += delta *delta;

        self.error /= len(outputLayer) - 1
        self.error = math.sqrt(self.error);

        self.recentAverageError =\
        (self.recentAverageError * Net.recentAverageSmoothingFactor + self.error) / (Net.recentAverageSmoothingFactor + 1.0)

        for n in range(len( outputLayer) -1):
            self.layers[-1][n].calcOutputGradients(targetVals[n])


        for layerNum in range( len(self.layers)-2,  0, -1 ):
            #hiddenLayer = self.layers[layerNum]
            nextLayer = self.layers[layerNum + 1]
            
            for n in range(len( self.layers[layerNum])):
                self.layers[layerNum][n].calcHiddenGradients(nextLayer)

        for layerNum in range( len(self.layers)-1, 0, -1 ):
            #layer = self.layers[layerNum]
            prevLayer = self.layers[layerNum - 1]
            for n in range(len( self.layers[layerNum] )-1):
                self.layers[layerNum][n].updateInputWeights(prevLayer)
                
        
    def getResults(self):
        results = []
        outputLayer = self.layers[-1]
        for n in range(len(outputLayer) -1 ):
            results.append( outputLayer[n].getOutputVal() )

        return results
    


# Not using the training class as the trainer did, instead I'm using a on fly inputs calculate to train the network

topology = [ 2 ,4, 1 ]    # 2 Inputs Neuron(Layer -1 ), 4 Hidden Neurons(Layer -2 ), 1 Output Neuron (Layer -3)
myNet = Net(topology)

for n in range(2000):
    a = random.randint(0,1)
    b = random.randint(0,1)
    c = (a and not b) or ( not a and b)
    myNet.feedForward([ a, b ] )

    res = myNet.getResults()
    print ( 'STEP %d : %d xor %d = %.2f' % ( n+1, a, b, res[0] )  )
    
    myNet.backProp( [c] )

    print ( 'Net recent average error %.2f' % ( myNet.getRecentAverageError() ) )

    
