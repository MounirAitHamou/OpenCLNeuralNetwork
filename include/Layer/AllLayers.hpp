#pragma once

// Include specific concrete layer implementations.
// This header acts as a central point to include all defined neural network layer types.
#include "Layer/Dense/DenseLayer.hpp" // Includes the definition for the DenseLayer class.

// Add more #include statements for other concrete layer types as they are implemented, e.g.:
// #include "Layer/Convolutional/ConvolutionalLayer.hpp"
// #include "Layer/Pooling/PoolingLayer.hpp"

// This file itself doesn't define any classes or functions, but serves to
// aggregate all layer definitions, making it easier to include them
// in other parts of the neural network framework (e.g., in a Network class).