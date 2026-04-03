#pragma once

// Include specific concrete layer implementations.
// This header acts as a central point to include all defined neural network layer types.
#include "Layers/TrainableLayers/Dense/DenseLayer.hpp"
#include "Layers/TrainableLayers/Convolutional/ConvolutionalLayer.hpp"
#include "Layers/ActivationLayers/Sigmoid/SigmoidLayer.hpp"
#include "Layers/ActivationLayers/Tanh/TanhLayer.hpp"
#include "Layers/ActivationLayers/Softmax/SoftmaxLayer.hpp"
#include "Layers/ActivationLayers/PreActivationLayers/LeakyReLU/LeakyReLULayer.hpp"
#include "Layers/ActivationLayers/PreActivationLayers/ReLU/ReLULayer.hpp"