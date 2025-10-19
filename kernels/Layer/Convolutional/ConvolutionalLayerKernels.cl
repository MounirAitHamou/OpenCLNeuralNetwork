#include "HelperFunctions.clh"

__kernel void convolutionalIm2col(
    __global float* p_im2colBuffer,
    const int p_inputHeight,
    const int p_inputWidth,
    const int p_inputChannels,
    const int p_filterHeight,
    const int p_filterWidth,
    const int p_strideY,
    const int p_strideX,
    const int p_paddingY,
    const int p_paddingX,
    const int p_outputHeight,
    const int p_outputWidth,
    __global const float* p_input)
{
    const int im2colRow = get_global_id(0);
    const int outputPosition = get_global_id(1);
    const int batchIndex = get_global_id(2);

    const int outputRow = outputPosition / p_outputWidth;
    const int outputCol = outputPosition % p_outputWidth;

    const int fx = im2colRow % p_filterWidth;
    const int fy = (im2colRow / p_filterWidth) % p_filterHeight;
    const int c  = im2colRow / (p_filterWidth * p_filterHeight);

    const int inputRow = outputRow * p_strideY - p_paddingY + fy;
    const int inputCol = outputCol * p_strideX - p_paddingX + fx;

    const int im2colRows = p_inputChannels * p_filterHeight * p_filterWidth;

    const int im2colCols = p_outputHeight * p_outputWidth;

    const int totalIm2colCols = im2colCols * get_global_size(2);

    const int columnIndex = batchIndex * im2colCols + outputPosition;

    const int outputIndex = im2colRow * totalIm2colCols + columnIndex;

    const int inputIndex = batchIndex * (p_inputChannels * p_inputHeight * p_inputWidth)    
                         + c * (p_inputHeight * p_inputWidth)
                         + inputRow * p_inputWidth
                         + inputCol;

    if (inputRow >= 0 && inputRow < p_inputHeight &&
        inputCol >= 0 && inputCol < p_inputWidth) {
        p_im2colBuffer[outputIndex] = p_input[inputIndex];
    } else {
        p_im2colBuffer[outputIndex] = 0.0f;
    }
}

__kernel void convolutionalBiasActivation(
    __global float* p_preActivations,
    __global const float* p_biases,
    __global float* p_outputs,
    const unsigned int p_totalOutputElements,
    const unsigned int p_outputChannels,
    const unsigned int p_activationType)
{
    const unsigned int globalId = get_global_id(0);
    const unsigned int batchIndex = get_global_id(1);

    const unsigned int batchOffset = batchIndex * p_totalOutputElements;
    const unsigned int outputIndex = batchOffset + globalId;

    const unsigned int spatialElementsPerImage = p_totalOutputElements / p_outputChannels;
    const unsigned int outputChannelIndex = globalId / spatialElementsPerImage;
    const float biasValue = p_biases[outputChannelIndex];

    p_preActivations[outputIndex] += biasValue;

    p_outputs[outputIndex] = applyActivation(p_preActivations[outputIndex], p_activationType);
}

__kernel void convolutionalBackpropActivation(
    __global float* p_deltas,
    __global const float* p_preActivations,
    const unsigned int p_activationType)
{
    const unsigned int gid = get_global_id(0);
    p_deltas[gid] *= applyActivationDerivative(p_preActivations[gid], p_activationType);
}


__kernel void convolutionalCol2im(
    __global const float* p_col2imBuffer,
    const int p_inputHeight,
    const int p_inputWidth,
    const int p_inputChannels,
    const int p_filterHeight,
    const int p_filterWidth,
    const int p_strideY,
    const int p_strideX,
    const int p_paddingY,
    const int p_paddingX,
    const int p_outputHeight,
    const int p_outputWidth,
    __global float* p_deltasBuffer)
{
    const int flatIndexPerImage = get_global_id(0);
    const int batch = get_global_id(1);

    const int spatialSize = p_inputHeight * p_inputWidth;
    
    const int inputC = flatIndexPerImage / spatialSize;
    const int H_W_Index = flatIndexPerImage % spatialSize;
    const int inputY = H_W_Index / p_inputWidth;
    const int inputX = H_W_Index % p_inputWidth;
    
    const int elementsPerSample = p_inputChannels * spatialSize;

    const int outputDeltaIndex = (batch * elementsPerSample) + flatIndexPerImage;

    float sum = 0.0f;

    for (int fY = 0; fY < p_filterHeight; ++fY) {
        for (int fX = 0; fX < p_filterWidth; ++fX) {
            
            const int inputY_padded = inputY + p_paddingY;
            const int inputX_padded = inputX + p_paddingX;

            if ((inputY_padded - fY) >= 0 && (inputY_padded - fY) % p_strideY == 0 &&
                (inputX_padded - fX) >= 0 && (inputX_padded - fX) % p_strideX == 0) {
                
                const int outputMapY = (inputY_padded - fY) / p_strideY;
                const int outputMapX = (inputX_padded - fX) / p_strideX;

                if (outputMapY >= 0 && outputMapY < p_outputHeight && 
                    outputMapX >= 0 && outputMapX < p_outputWidth) {
                    
                    const int im2colRow = (inputC * p_filterHeight * p_filterWidth) +
                                          (fY * p_filterWidth) + fX;
                    
                    const int outputElementsPerImage = p_outputHeight * p_outputWidth;
                    const int im2colCol = (batch * outputElementsPerImage) +
                                          (outputMapY * p_outputWidth) + outputMapX;

                    const int totalIm2colCols = get_global_size(1) * outputElementsPerImage;
                    
                    const int im2colIndex = im2colRow * totalIm2colCols + im2colCol;
                    
                    sum += p_col2imBuffer[im2colIndex];
                }
            }
        }
    }
    
    p_deltasBuffer[outputDeltaIndex] += sum;
}

__kernel void convolutionalAverageWeightsGradientsKernel(
    __global float* p_weightsGradients,
    const unsigned int p_batchSize)
{
    const unsigned int gid = get_global_id(0);
    p_weightsGradients[gid] /= (float)p_batchSize;
}

__kernel void convolutionalAverageBiasesGradientsKernel(
    __global float* p_biasesGradients,
    const unsigned int p_batchSize)
{
    const unsigned int gid = get_global_id(0);
    p_biasesGradients[gid] /= (float)p_batchSize;
}
