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
        if (m_batchSize < p_batchSize) setBatchSize(p_batchSize);
        m_im2colKernel.setArg(12, p_inputs);

        size_t im2colRows = (size_t)getInputChannels() * m_filterDimensions.getHeight() * m_filterDimensions.getWidth();
        size_t im2colCols = (size_t)getOutputHeight() * getOutputWidth();

        cl_int err = p_forwardBackpropQueue.enqueueNDRangeKernel(
            m_im2colKernel, 
            cl::NullRange, 
            cl::NDRange(im2colRows, im2colCols, p_batchSize),
            cl::NullRange);

        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue im2col kernel.");
        }

        cl_command_queue raw_queue = p_forwardBackpropQueue.get();

        size_t N = im2colCols * p_batchSize;

        auto status = clblast::Gemm<float>(
            clblast::Layout::kRowMajor,
            clblast::Transpose::kNo,
            clblast::Transpose::kNo,
            getOutputChannels(),
            N,
            im2colRows,
            NO_SCALAR,
            getWeights()(), NO_OFFSET, im2colRows,
            m_im2colBuffer(), NO_OFFSET, N,
            CLEAR_C,
            getOutputs()(), NO_OFFSET, N,
            &raw_queue, nullptr,
            m_clblastWorkspace()
        );

        if (status != clblast::StatusCode::kSuccess) {
            std::cerr << "Forward CLBlast GEMM failed: " << static_cast<int>(status) << std::endl;
            throw std::runtime_error("CLBlast GEMM failed");
        }

        cl::Event returnEvent;

        err = p_forwardBackpropQueue.enqueueNDRangeKernel(
            m_biasKernel,
            cl::NullRange,
            cl::NDRange(getTotalOutputElements(), p_batchSize), 
            cl::NullRange,
            nullptr,
            &returnEvent
        );

        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue bias addition kernel.");
        }

        return returnEvent;
    }

    cl::Event ConvolutionalLayer::backpropDeltas(const cl::CommandQueue& p_forwardBackpropQueue, 
                                            const cl::Buffer& p_previousLayerDeltas,
                                            const size_t p_batchSize) {
        if (m_batchSize < p_batchSize) setBatchSize(p_batchSize);

        size_t M = getOutputChannels();
        size_t N = getOutputHeight() * getOutputWidth() * p_batchSize;
        size_t K = getInputChannels() * m_filterDimensions.getHeight() * m_filterDimensions.getWidth();

        cl_command_queue raw_queue = p_forwardBackpropQueue.get();
        
        auto status = clblast::Gemm<float>(
            clblast::Layout::kRowMajor,
            clblast::Transpose::kYes,
            clblast::Transpose::kNo,
            K, N, M,
            NO_SCALAR,
            getWeights()(), NO_OFFSET, K,
            getDeltas()(), NO_OFFSET, N,
            CLEAR_C,
            m_col2imBuffer(), NO_OFFSET, N,
            &raw_queue, nullptr,
            m_clblastDeltaWorkspace()
        );

        if (status != clblast::StatusCode::kSuccess) {
            std::cerr << "Backprop Delta CLBlast GEMM failed: " << static_cast<int>(status) << std::endl;
            throw std::runtime_error("CLBlast GEMM failed");
        }
        
        m_col2imKernel.setArg(12, p_previousLayerDeltas);

        cl::Event returnEvent;

        cl_int err = p_forwardBackpropQueue.enqueueNDRangeKernel(
                m_col2imKernel, 
                cl::NullRange,
                cl::NDRange(m_inputDimensions.getTotalElements(), p_batchSize),
                cl::NullRange,
                nullptr,
                &returnEvent
        );

        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue col2im kernel.");
        }

        return returnEvent;
    }

    std::pair<cl::Event, cl::Event> ConvolutionalLayer::computeGradients(const cl::CommandQueue& p_deltaToGradientQueue, 
                                                                            cl::Event& p_backpropEvent,
                                                                            const cl::Buffer& p_inputs,
                                                                            const size_t p_batchSize) {
        if (m_batchSize < p_batchSize) setBatchSize(p_batchSize);
        std::vector<cl::Event> deltaBackPropWaitList = { p_backpropEvent };
        p_deltaToGradientQueue.enqueueBarrierWithWaitList(&deltaBackPropWaitList);

        size_t im2colRows = (size_t)getInputChannels() * m_filterDimensions.getHeight() * m_filterDimensions.getWidth();
        size_t im2colCols = (size_t)getOutputHeight() * getOutputWidth();

        cl_event raw_gemm_event = nullptr;
        cl_command_queue raw_queue = p_deltaToGradientQueue.get();
        
        auto status = clblast::Gemm<float>(
            clblast::Layout::kRowMajor,
            clblast::Transpose::kNo,
            clblast::Transpose::kYes,
            getOutputChannels(),
            im2colRows,
            p_batchSize * im2colCols,
            NO_SCALAR,
            getDeltas()(), NO_OFFSET, p_batchSize * im2colCols,
            m_im2colBuffer(), NO_OFFSET, p_batchSize * im2colCols,
            CLEAR_C,
            getWeightsGradients()(), NO_OFFSET, im2colRows,
            &raw_queue, &raw_gemm_event,
            m_clblastWorkspace()
        );

        if (status != clblast::StatusCode::kSuccess) {
            std::cerr << "Weights Gradients CLBlast GEMM failed: " << static_cast<int>(status) << std::endl;
            throw std::runtime_error("CLBlast GEMM failed");
        }

        cl::Event gemmEvent(raw_gemm_event, true);

        cl_event raw_gemv_event = nullptr;
        
        status = clblast::Gemv<float>(
            clblast::Layout::kRowMajor,
            clblast::Transpose::kNo,
            getOutputChannels(),
            p_batchSize * im2colCols,
            NO_SCALAR,
            getDeltas()(), NO_OFFSET, p_batchSize * im2colCols,
            m_onesBuffer(), NO_OFFSET, 1,
            CLEAR_C,
            getBiasesGradients()(), NO_OFFSET, 1,
            &raw_queue,
            &raw_gemv_event
        );

        if (status != clblast::StatusCode::kSuccess) {
            std::cerr << "Bias Gradients CLBlast GEMM failed: " << static_cast<int>(status) << std::endl;
            throw std::runtime_error("CLBlast GEMV failed");
        }

        cl::Event gemvEvent(raw_gemv_event, true);

        return { gemmEvent, gemvEvent };
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
        
        m_im2colBuffer = cl::Buffer(
            m_sharedResources->getContext(), 
            CL_MEM_READ_WRITE, 
            im2colRows * im2colBatchCols * sizeof(float)
        );

        m_col2imBuffer = cl::Buffer(
            m_sharedResources->getContext(), 
            CL_MEM_READ_WRITE, 
            im2colRows * im2colBatchCols * sizeof(float)
        );
        
        m_onesBuffer = cl::Buffer(
            m_sharedResources->getContext(), 
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
            im2colBatchCols * sizeof(float),
            std::vector<float>(im2colBatchCols, 1.0f).data()
        );

        size_t requiredWorkspaceSize = std::max({
            (size_t)getOutputChannels() * im2colBatchCols,
            (size_t)getOutputChannels() * im2colRows
        });

        m_clblastWorkspace = cl::Buffer(
            m_sharedResources->getContext(),
            CL_MEM_READ_WRITE,
            std::max({ (size_t)getOutputChannels() * im2colBatchCols, (size_t)getOutputChannels() * im2colRows}) * sizeof(float)
        );

        m_clblastDeltaWorkspace = cl::Buffer(
            m_sharedResources->getContext(),
            CL_MEM_READ_WRITE,
            im2colRows * im2colBatchCols * sizeof(float)
        );
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

        m_im2colKernel = cl::Kernel(m_sharedResources->getProgram(), "convolutionalIm2col", &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create im2col kernel");
        }
        m_im2colKernel.setArg(0, m_im2colBuffer);
        m_im2colKernel.setArg(1, (cl_uint)getInputHeight());
        m_im2colKernel.setArg(2, (cl_uint)getInputWidth());
        m_im2colKernel.setArg(3, (cl_uint)getInputChannels());
        m_im2colKernel.setArg(4, (cl_uint)m_filterDimensions.getHeight());
        m_im2colKernel.setArg(5, (cl_uint)m_filterDimensions.getWidth());
        m_im2colKernel.setArg(6, (cl_uint)m_strideDimensions.getHeight());
        m_im2colKernel.setArg(7, (cl_uint)m_strideDimensions.getWidth());
        m_im2colKernel.setArg(8, (cl_uint)m_paddingValues.getTop());
        m_im2colKernel.setArg(9, (cl_uint)m_paddingValues.getLeft());
        m_im2colKernel.setArg(10, (cl_uint)getOutputHeight());
        m_im2colKernel.setArg(11, (cl_uint)getOutputWidth());

        m_biasKernel = cl::Kernel(m_sharedResources->getProgram(), "convolutionalBias", &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create convBias kernel");
        }
        m_biasKernel.setArg(0, getBiases());
        m_biasKernel.setArg(1, getOutputs());
        m_biasKernel.setArg(2, (cl_uint)getTotalOutputElements());
        m_biasKernel.setArg(3, (cl_uint)getOutputChannels());

        m_col2imKernel = cl::Kernel(m_sharedResources->getProgram(), "convolutionalCol2im", &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create col2im kernel");
        }
        m_col2imKernel.setArg(0, m_col2imBuffer);
        m_col2imKernel.setArg(1, (cl_uint)getInputHeight());
        m_col2imKernel.setArg(2, (cl_uint)getInputWidth());
        m_col2imKernel.setArg(3, (cl_uint)getInputChannels());
        m_col2imKernel.setArg(4, (cl_uint)m_filterDimensions.getHeight());
        m_col2imKernel.setArg(5, (cl_uint)m_filterDimensions.getWidth());
        m_col2imKernel.setArg(6, (cl_uint)m_strideDimensions.getHeight());
        m_col2imKernel.setArg(7, (cl_uint)m_strideDimensions.getWidth());
        m_col2imKernel.setArg(8, (cl_uint)m_paddingValues.getTop());
        m_col2imKernel.setArg(9, (cl_uint)m_paddingValues.getLeft());
        m_col2imKernel.setArg(10, (cl_uint)getOutputHeight());
        m_col2imKernel.setArg(11, (cl_uint)getOutputWidth());
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