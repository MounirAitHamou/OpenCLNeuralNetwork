#include "Layers/TrainableLayers/Convolutional/ConvolutionalLayer.hpp"
namespace Layers::Trainable {
    ConvolutionalLayer::ConvolutionalLayer(const size_t p_layerId, 
                                        std::shared_ptr<Utils::SharedResources> p_sharedResources,
                                        const Utils::Dimensions& p_inputDimensions,
                                        const Utils::FilterDimensions& p_filterDimensions,
                                        const Utils::StrideDimensions& p_strideDimensions,
                                        const Utils::PaddingType p_paddingType,
                                        const size_t p_batchSize,
                                        std::mt19937& p_rng)
        : 
        TrainableLayer(p_layerId, p_sharedResources, validateInputDimensions(p_inputDimensions, p_filterDimensions, p_strideDimensions), calculateOutputDimensions(validateInputDimensions(p_inputDimensions, p_filterDimensions, p_strideDimensions), p_filterDimensions, p_strideDimensions, p_paddingType), p_batchSize),
        m_filterDimensions(p_filterDimensions),
        m_strideDimensions(p_strideDimensions),
        m_paddingValues(calculatePaddingValues(m_inputDimensions, p_filterDimensions, p_strideDimensions, p_paddingType)),
        m_paddingType(p_paddingType)
    {
        try {
            initializeWeightsAndBiases(p_rng);
            allocateConvolutionalLayerBuffers(p_batchSize);
            setupKernels();
        }
        catch (const std::exception& e) {
            std::cerr << "Error constructing ConvolutionalLayer: " << e.what() << std::endl;
            throw; 
        }
    }

    ConvolutionalLayer::ConvolutionalLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                                        const H5::Group& p_layerGroup, 
                                        const size_t p_batchSize)
        : TrainableLayer(p_sharedResources, p_layerGroup, p_batchSize) {
        m_filterDimensions = Utils::FilterDimensions(Utils::readVectorFromHDF5<size_t>(p_layerGroup, "filterDimensions"));
        m_strideDimensions = Utils::StrideDimensions(Utils::readVectorFromHDF5<size_t>(p_layerGroup, "strideDimensions"));
        m_paddingValues = Utils::PaddingValues(Utils::readVectorFromHDF5<size_t>(p_layerGroup, "paddingValues"));
        m_paddingType = Utils::paddingTypeFromUint(Utils::readValueFromHDF5<unsigned int>(p_layerGroup, "paddingType"));
        m_weights = Utils::loadBuffer(p_sharedResources->getContext(), p_layerGroup, "weights", getWeightsSize());
        m_biases  = Utils::loadBuffer(p_sharedResources->getContext(), p_layerGroup, "biases", getBiasesSize());
        allocateConvolutionalLayerBuffers(p_batchSize);
        setupKernels();
    }

    cl::Event ConvolutionalLayer::runForward(const cl::CommandQueue& p_forwardBackpropQueue,
                                            const cl::Buffer& p_inputs,
                                            const size_t p_batchSize) {
        cl_command_queue raw_queue = p_forwardBackpropQueue.get();

        auto status = clblast::Convgemm<float>(
            clblast::KernelMode::kCrossCorrelation,
            getInputChannels(), getInputHeight(), getInputWidth(),
            m_filterDimensions.getHeight(), m_filterDimensions.getWidth(),
            m_paddingValues.getTop(), m_paddingValues.getLeft(),
            m_strideDimensions.getHeight(), m_strideDimensions.getWidth(),
            1, 1,
            getOutputChannels(),
            p_batchSize,
            p_inputs(), 0,
            getWeights()(), 0,
            getOutputs()(), 0,
            &raw_queue, nullptr
        );

        if (status != clblast::StatusCode::kSuccess) {
            throw std::runtime_error("CLBlast Convgemm failed with status: " + std::to_string(static_cast<int>(status)));
        }

        cl::Event returnEvent;
        cl::NDRange globalSize(getOutputChannels(), getOutputHeight() * getOutputWidth(), p_batchSize);
        
        cl_int err = p_forwardBackpropQueue.enqueueNDRangeKernel(
            m_biasKernel, 
            cl::NullRange, 
            globalSize, 
            cl::NullRange, 
            nullptr, 
            &returnEvent
        );

        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue bias addition kernel.");
        }

        return returnEvent;
    }

    cl::Event ConvolutionalLayer::backpropDeltas(
        const cl::CommandQueue& p_forwardBackpropQueue,
        const cl::Buffer& p_previousLayerDeltas,
        size_t p_batchSize
    ) {
        size_t globalWidth = (getInputWidth() + 1) / 2; 

        cl::NDRange globalSize(
            globalWidth, 
            (size_t)getInputHeight(), 
            (size_t)getInputChannels() * p_batchSize
        );

        m_backpropDeltasKernel.setArg(14, p_previousLayerDeltas);
        
        cl::Event executionEvent;
        p_forwardBackpropQueue.enqueueNDRangeKernel(
            m_backpropDeltasKernel,
            cl::NullRange,
            globalSize,
            cl::NullRange,
            nullptr,
            &executionEvent
        );

        return executionEvent;
    }

    std::pair<cl::Event, cl::Event> ConvolutionalLayer::computeGradients(
        const cl::CommandQueue& p_queue,
        cl::Event& p_backpropEvent,
        const cl::Buffer& p_inputs,
        const size_t p_batchSize
    ) {
        if (m_batchSize < p_batchSize) setBatchSize(p_batchSize);

        std::vector<cl::Event> waitList;
        if (p_backpropEvent() != nullptr) {
            waitList.push_back(p_backpropEvent);
        }

        cl::NDRange globalSize(
            (size_t)m_filterDimensions.getWidth(), 
            (size_t)m_filterDimensions.getHeight(), 
            (size_t)getInputChannels() * getOutputChannels()
        );

        m_computeWeightsGradientsKernel.setArg(14, p_inputs);
        m_computeWeightsGradientsKernel.setArg(15, (int)p_batchSize);

        cl::Event weightsEvent;
        p_queue.enqueueNDRangeKernel(
            m_computeWeightsGradientsKernel,
            cl::NullRange,
            globalSize,
            cl::NullRange,
            &waitList,
            &weightsEvent
        );

        cl::NDRange biasGlobalSize(getOutputChannels());
        cl::Event biasEvent;
        m_computeBiasesGradientsKernel.setArg(5, (cl_int)p_batchSize);
        p_queue.enqueueNDRangeKernel(
            m_computeBiasesGradientsKernel,
            cl::NullRange,
            biasGlobalSize,
            cl::NullRange,
            &waitList,
            &biasEvent
        );
        return { weightsEvent, biasEvent  };
    }

    void ConvolutionalLayer::allocateConvolutionalLayerBuffers(const size_t p_batchSize) {
        size_t im2colRows = getInputChannels() * m_filterDimensions.getHeight() * m_filterDimensions.getWidth();
        size_t im2colCols = getOutputHeight() * getOutputWidth();
        size_t im2colBatchCols = im2colCols * p_batchSize;

        m_weightsGradients = cl::Buffer(
            m_sharedResources->getContext(), 
            CL_MEM_READ_WRITE, 
            (getWeightsSize()) * sizeof(float)
        );
        
        m_biasesGradients = cl::Buffer(
            m_sharedResources->getContext(), 
            CL_MEM_READ_WRITE, 
            (getBiasesSize()) * sizeof(float)
        );
        
        size_t requiredWorkspaceSize = std::max({
            (size_t)getOutputChannels() * im2colBatchCols,
            (size_t)getOutputChannels() * im2colRows
        });
    }

    Utils::Dimensions ConvolutionalLayer::calculateOutputDimensions(const Utils::Dimensions& p_inputDimensions, const Utils::FilterDimensions& p_filterDimensions, const Utils::StrideDimensions& p_strideDimensions, Utils::PaddingType p_paddingType) const {
        Utils::PaddingValues paddingValues = calculatePaddingValues(p_inputDimensions, p_filterDimensions, p_strideDimensions, p_paddingType);
        size_t inputHeight    = p_inputDimensions.getDimensions()[1];
        size_t inputWidth     = p_inputDimensions.getDimensions()[2];
        size_t filterHeight   = p_filterDimensions.getHeight();
        size_t filterWidth    = p_filterDimensions.getWidth();
        size_t strideHeight   = p_strideDimensions.getHeight();
        size_t strideWidth    = p_strideDimensions.getWidth();
        size_t padTop         = paddingValues.getTop();
        size_t padLeft        = paddingValues.getLeft();
        size_t padBottom      = paddingValues.getBottom();
        size_t padRight       = paddingValues.getRight();

        long numeratorHeight = static_cast<long>(inputHeight - filterHeight + padTop + padBottom);
        long numeratorWidth  = static_cast<long>(inputWidth - filterWidth + padLeft + padRight);

        size_t outputHeight = static_cast<size_t>(
            floor(static_cast<double>(numeratorHeight) / static_cast<double>(strideHeight)) + 1
        );
        
        size_t outputWidth = static_cast<size_t>(
            floor(static_cast<double>(numeratorWidth) / static_cast<double>(strideWidth)) + 1
        );
        
        if (outputHeight == 0 || outputWidth == 0) {
            std::cerr << "Error: Calculated output dimensions are invalid (zero)." << std::endl;
            throw std::runtime_error("Calculated output dimensions are invalid (zero). Check filter, stride, and padding settings.");
        }

        size_t outputChannels = p_filterDimensions.getOutputChannels();
        return Utils::Dimensions({ outputChannels, outputHeight, outputWidth });
    }

    Utils::Dimensions ConvolutionalLayer::calculateOutputDimensions() const {
        long inputHeight  = static_cast<long>(getInputHeight());
        long inputWidth   = static_cast<long>(getInputWidth());
        long filterHeight = static_cast<long>(m_filterDimensions.getHeight());
        long filterWidth  = static_cast<long>(m_filterDimensions.getWidth());
        long strideHeight = static_cast<long>(m_strideDimensions.getHeight());
        long strideWidth  = static_cast<long>(m_strideDimensions.getWidth());
        long padTop       = static_cast<long>(m_paddingValues.getTop());
        long padLeft      = static_cast<long>(m_paddingValues.getLeft());
        long padBottom    = static_cast<long>(m_paddingValues.getBottom());
        long padRight     = static_cast<long>(m_paddingValues.getRight());

        long numeratorHeight = inputHeight - filterHeight + padTop + padBottom;
        long numeratorWidth  = inputWidth - filterWidth + padLeft + padRight;

        size_t outputHeight = static_cast<size_t>(
            floor(static_cast<double>(numeratorHeight) / static_cast<double>(strideHeight)) + 1
        );
        
        size_t outputWidth = static_cast<size_t>(
            floor(static_cast<double>(numeratorWidth) / static_cast<double>(strideWidth)) + 1
        );
        
        if (outputHeight == 0 || outputWidth == 0) {
            std::cerr << "Error: Calculated output dimensions are invalid (zero)." << std::endl;
            throw std::runtime_error("Calculated output dimensions are invalid (zero). Check filter, stride, and padding settings.");
        }

        size_t outputChannels = m_filterDimensions.getOutputChannels();
        return Utils::Dimensions({ outputChannels, outputHeight, outputWidth });
    }

    Utils::PaddingValues ConvolutionalLayer::calculatePaddingValues(
        const Utils::Dimensions& p_inputDimensions, 
        const Utils::FilterDimensions& p_filterDimensions, 
        const Utils::StrideDimensions& p_strideDimensions, 
        const Utils::PaddingType p_paddingType) const 
    {
        size_t inputHeight = p_inputDimensions.getDimensions()[1]; 
        size_t inputWidth  = p_inputDimensions.getDimensions()[2]; 
        
        switch (p_paddingType) {
            case Utils::PaddingType::Valid: {
                return Utils::PaddingValues(0, 0, 0, 0);
            }
            case Utils::PaddingType::Same: {
                long long inputH_l = static_cast<long long>(inputHeight);
                long long inputW_l = static_cast<long long>(inputWidth);
                long long filterH_l = static_cast<long long>(p_filterDimensions.getHeight());
                long long filterW_l = static_cast<long long>(p_filterDimensions.getWidth());
                long long strideH_l = static_cast<long long>(p_strideDimensions.getHeight());
                long long strideW_l = static_cast<long long>(p_strideDimensions.getWidth());

                long long outputHeight = (inputH_l + strideH_l - 1) / strideH_l;
                long long outputWidth  = (inputW_l + strideW_l - 1) / strideW_l;

                long long totalPaddingHeight_l = (outputHeight - 1) * strideH_l + filterH_l - inputH_l;
                long long totalPaddingWidth_l  = (outputWidth - 1) * strideW_l + filterW_l - inputW_l;

                if (totalPaddingHeight_l < 0 || totalPaddingWidth_l < 0) {
                    std::cerr << "Error: Invalid convolution configuration. Same padding is insufficient." << std::endl;
                    std::cerr << "Required Padding H: " << totalPaddingHeight_l << ", W: " << totalPaddingWidth_l << std::endl;
                    throw std::invalid_argument("Input dimensions are too small for filter/stride combination, even with 'Same' padding.");
                }

                size_t totalPaddingHeight = static_cast<size_t>(totalPaddingHeight_l);
                size_t totalPaddingWidth  = static_cast<size_t>(totalPaddingWidth_l);
                
                size_t padTop    = totalPaddingHeight / 2;
                size_t padBottom = totalPaddingHeight - padTop;
                size_t padLeft   = totalPaddingWidth / 2;
                size_t padRight  = totalPaddingWidth - padLeft;
                
                return Utils::PaddingValues(padTop, padBottom, padLeft, padRight);
            }
            default: {
                std::cerr << "Warning: Unsupported padding type. Setting padding to zero." << std::endl;
                return Utils::PaddingValues(0, 0, 0, 0);
            }
        }
    }

    void ConvolutionalLayer::initializeWeightsAndBiases(std::mt19937& p_rng) {
        std::vector<float> h_weights(getWeightsSize());
        std::vector<float> h_biases(getBiasesSize());

        float fan =(float)m_filterDimensions.getHeight() * m_filterDimensions.getWidth();
        float limit = std::sqrt(6.0f / (getInputChannels() + getOutputChannels()) * fan);
        
        for (auto& weight : h_weights) {
            weight = getRandomValue(-limit, limit, p_rng);
        }
        
        for (auto& bias : h_biases) {
            bias = 0.0f;
        }
        
        m_weights = cl::Buffer(
            m_sharedResources->getContext(), 
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
            h_weights.size() * sizeof(float), 
            h_weights.data()
        );
        
        m_biases = cl::Buffer(
            m_sharedResources->getContext(), 
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
            h_biases.size() * sizeof(float), 
            h_biases.data()
        );
    }

    void ConvolutionalLayer::setupKernels() {
        setupTrainableKernels();
        cl_int err;

        m_biasKernel = cl::Kernel(m_sharedResources->getProgram(), "convolutionalBias", &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create convBias kernel");
        }
        m_biasKernel.setArg(0, getBiases());
        m_biasKernel.setArg(1, getOutputs());
        m_biasKernel.setArg(2, (int)getOutputHeight());
        m_biasKernel.setArg(3, (int)getOutputWidth());
        m_biasKernel.setArg(4, (int)getOutputChannels());

        m_backpropDeltasKernel = cl::Kernel(m_sharedResources->getProgram(), "convolutionalBackpropDeltas", &err);

        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create backprop kernel.");
        }
        m_backpropDeltasKernel.setArg(0, getWeights());
        m_backpropDeltasKernel.setArg(1, getDeltas());
        m_backpropDeltasKernel.setArg(2, (cl_int)getInputHeight());
        m_backpropDeltasKernel.setArg(3, (cl_int)getInputWidth());
        m_backpropDeltasKernel.setArg(4, (cl_int)getOutputHeight());
        m_backpropDeltasKernel.setArg(5, (cl_int)getOutputWidth());
        m_backpropDeltasKernel.setArg(6, (cl_int)m_filterDimensions.getHeight());
        m_backpropDeltasKernel.setArg(7, (cl_int)m_filterDimensions.getWidth());
        m_backpropDeltasKernel.setArg(8, (cl_int)m_strideDimensions.getHeight());
        m_backpropDeltasKernel.setArg(9, (cl_int)m_strideDimensions.getWidth());
        m_backpropDeltasKernel.setArg(10, (cl_int)m_paddingValues.getTop());
        m_backpropDeltasKernel.setArg(11, (cl_int)m_paddingValues.getLeft());
        m_backpropDeltasKernel.setArg(12, (cl_int)getInputChannels());
        m_backpropDeltasKernel.setArg(13, (cl_int)getOutputChannels());

        m_computeWeightsGradientsKernel = cl::Kernel(m_sharedResources->getProgram(), "convolutionalComputeWeightsGradients", &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create compute weights gradients kernel.");
        }

        m_computeWeightsGradientsKernel.setArg(0,  getDeltas());
        m_computeWeightsGradientsKernel.setArg(1,  getWeightsGradients());
        m_computeWeightsGradientsKernel.setArg(2,  (int)getInputChannels());
        m_computeWeightsGradientsKernel.setArg(3,  (int)getInputHeight());
        m_computeWeightsGradientsKernel.setArg(4,  (int)getInputWidth());
        m_computeWeightsGradientsKernel.setArg(5,  (int)getOutputChannels());
        m_computeWeightsGradientsKernel.setArg(6,  (int)getOutputHeight());
        m_computeWeightsGradientsKernel.setArg(7,  (int)getOutputWidth());
        m_computeWeightsGradientsKernel.setArg(8, (int)m_filterDimensions.getHeight());
        m_computeWeightsGradientsKernel.setArg(9, (int)m_filterDimensions.getWidth());
        m_computeWeightsGradientsKernel.setArg(10, (int)m_strideDimensions.getHeight());
        m_computeWeightsGradientsKernel.setArg(11, (int)m_strideDimensions.getWidth());
        m_computeWeightsGradientsKernel.setArg(12, (int)m_paddingValues.getTop());
        m_computeWeightsGradientsKernel.setArg(13, (int)m_paddingValues.getLeft());

        m_computeBiasesGradientsKernel = cl::Kernel(m_sharedResources->getProgram(), "convolutionalComputeBiasesGradients", &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create compute biases gradients kernel.");
        }
        m_computeBiasesGradientsKernel.setArg(0, getDeltas());
        m_computeBiasesGradientsKernel.setArg(1, getBiasesGradients());
        m_computeBiasesGradientsKernel.setArg(2, (int)getOutputChannels());
        m_computeBiasesGradientsKernel.setArg(3, (int)getOutputHeight());
        m_computeBiasesGradientsKernel.setArg(4, (int)getOutputWidth());
    }

    Utils::Dimensions ConvolutionalLayer::validateInputDimensions(
        const Utils::Dimensions& p_inputDimensions, 
        const Utils::FilterDimensions& p_filterDimensions, 
        const Utils::StrideDimensions& p_strideDimensions) const 
    {
        std::vector<size_t> dims = p_inputDimensions.getDimensions();
        size_t initial_dims = dims.size();
        Utils::Dimensions validDimensions;

        if (initial_dims == 1) {
            validDimensions = Utils::Dimensions({dims[0], 1, 1});
        } else if (initial_dims == 2) {
            validDimensions = Utils::Dimensions({dims[0], dims[1], 1});
        } else if (initial_dims == 3) {
            validDimensions = p_inputDimensions;
        } else {
            std::cerr << "Error: Input dimensions must be 1D, 2D, or 3D (Channels, Height, Width)." << std::endl;
            throw std::invalid_argument("Input dimensions must be 1D, 2D, or 3D.");
        }
        
        if (p_filterDimensions.getInputChannels() != validDimensions.getDimensions()[0]) {
            std::cerr << "Error: Filter's input channels (" << p_filterDimensions.getInputChannels() 
                    << ") do not match the input volume's channels (" << validDimensions.getDimensions()[0] << ")." << std::endl;
            throw std::invalid_argument("Input channels of filter dimensions must match the channels of input dimensions.");
        }
        
        if (p_filterDimensions.getHeight() <= 0 || p_filterDimensions.getWidth() <= 0) {
            std::cerr << "Error: Filter dimensions (" << p_filterDimensions.getHeight() << "x" << p_filterDimensions.getWidth() 
                    << ") must be strictly positive integers (> 0)." << std::endl;
            throw std::invalid_argument("Filter dimensions must be strictly positive.");
        }

        if (p_strideDimensions.getHeight() <= 0 || p_strideDimensions.getWidth() <= 0) {
            std::cerr << "Error: Stride dimensions (" 
                    << p_strideDimensions.getHeight() << "x" << p_strideDimensions.getWidth()
                    << ") must be strictly positive integers (> 0)." << std::endl;
            throw std::invalid_argument("Stride dimensions must be strictly positive.");
        }

        return validDimensions;
    }
}