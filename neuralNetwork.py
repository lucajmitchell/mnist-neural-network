import numpy as np
from .neuralNetworkUtils import oneHotEncode



# ===== SETUP =====

def initialiseParameters(layers):

    params = {}

    for i in range(len(layers)-1):
        nodes_thisLayer = layers[i]
        nodes_nextLayer = layers[i+1]

        params[f"L{i}W"] = np.random.randn(nodes_nextLayer, nodes_thisLayer) * np.sqrt(2 / nodes_thisLayer)
        params[f"L{i}B"] = np.random.rand(nodes_nextLayer, 1) - 0.5

    return params



# ===== FORWARD PROPOGATION =====

def leakyReLU(n, x):
    return np.maximum(n * x, x)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def tanh(x):
    return np.tanh(x)


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def forwardPropagate(params, inputs, activationFunction, numOfLayers):

    updatedParams = params

    for i in range(numOfLayers - 1):
        weights = params[f"L{i}W"]
        biases = params[f"L{i}B"]

        unactivated = np.dot(weights, inputs) + biases
        
        if i == numOfLayers - 2:
            layerActivationFunction = "softmax"
        else:
            layerActivationFunction = activationFunction

        match layerActivationFunction:
            case "ReLU":
                activated = leakyReLU(0, unactivated)
            case "leakyReLU":
                activated = leakyReLU(0.01, unactivated)
            case "sigmoid":
                activated = sigmoid(unactivated)
            case "tanh":
                activated = tanh(unactivated)
            case "softmax":
                activated = softmax(unactivated)

        updatedParams[f"L{i}Z"] = unactivated
        updatedParams[f"L{i}A"] = activated

        inputs = activated

    return updatedParams



# ===== BACK PROPOGATION =====

def backPropogate(params, inputs, oneHotLabels, numOfLayers):
    updatedParams = params
    numOfImages = inputs.shape[1]

    # Last layer (output layer)
    updatedParams[f"dL{numOfLayers-2}Z"] = params[f"L{numOfLayers-2}A"] - oneHotLabels  # Error at output layer
    updatedParams[f"dL{numOfLayers-2}W"] = np.dot(updatedParams[f"dL{numOfLayers-2}Z"], params[f"L{numOfLayers-3}A"].T) / numOfImages
    updatedParams[f"dL{numOfLayers-2}B"] = np.sum(updatedParams[f"dL{numOfLayers-2}Z"], axis=1, keepdims=True) / numOfImages

    for i in range(numOfLayers - 3, 0, -1):
        updatedParams[f"dL{i}A"] = np.dot(params[f"L{i+1}W"].T, updatedParams[f"dL{i+1}Z"])
        updatedParams[f"dL{i}Z"] = updatedParams[f"dL{i}A"] * (params[f"L{i}Z"] > 0)
        updatedParams[f"dL{i}W"] = np.dot(updatedParams[f"dL{i}Z"], params[f"L{i-1}A"].T) / numOfImages
        updatedParams[f"dL{i}B"] = np.sum(updatedParams[f"dL{i}Z"], axis=1, keepdims=True) / numOfImages

    # Input layer (first layer after the inputs)
    updatedParams["dL0A"] = np.dot(params["L1W"].T, updatedParams["dL1Z"])
    updatedParams["dL0Z"] = updatedParams["dL0A"] * (params["L0Z"] > 0)
    updatedParams["dL0W"] = np.dot(updatedParams["dL0Z"], inputs.T) / numOfImages
    updatedParams["dL0B"] = np.sum(updatedParams["dL0Z"], axis=1, keepdims=True) / numOfImages

    return updatedParams



def updateParams(params, alpha, numOfLayers):
    for i in range(numOfLayers - 1):
        params[f"L{i}W"] -= alpha * params[f"dL{i}W"]
        params[f"L{i}B"] -= alpha * params[f"dL{i}B"]
    return params



# ===== GRADIENT DESCENT =====

def gradientDescent(trainingLabels, trainingImages, i, alpha, activationFunction, layers):

    numOfLayers = len(layers)

    params = initialiseParameters(layers)
    oneHotLabels = oneHotEncode(trainingLabels)

    for iter in range(i):

        params = forwardPropagate(params, trainingImages, activationFunction, numOfLayers)
        params = backPropogate(params, trainingImages, oneHotLabels, numOfLayers)
        params = updateParams(params, alpha, numOfLayers)

        print(f"Iteration {iter}")

    return params