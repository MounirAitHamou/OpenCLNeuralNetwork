#include "Layers/TrainableLayers/Convolutional/ConvolutionalLayer.hpp"
namespace Layers::Trainable
{
    ConvolutionalLayer::ConvolutionalLayer(const size_t p_layerId,
                                           std::shared_ptr<Utils::SharedResources> p_sharedResources,
                                           const Utils::Dimensions &p_inputDimensions,
                                           const Utils::FilterDimensions &p_filterDimensions,
                                           const Utils::StrideDimensions &p_strideDimensions,
                                           const Utils::PaddingType p_paddingType,
                                           const size_t p_batchSize,
                                           std::mt19937 &p_rng)
        : TrainableLayer(p_layerId, std::move(p_sharedResources), validateInputDimensions(p_inputDimensions, p_filterDimensions, p_strideDimensions), calculateOutputDimensions(validateInputDimensions(p_inputDimensions, p_filterDimensions, p_strideDimensions), p_filterDimensions, p_strideDimensions, p_paddingType), p_batchSize),
          m_filterDimensions(p_filterDimensions),
          m_strideDimensions(p_strideDimensions),
          m_paddingValues(calculatePaddingValues(m_inputDimensions, p_filterDimensions, p_strideDimensions, p_paddingType)),
          m_paddingType(p_paddingType)
    {

        initializeWeightsAndBiases(p_rng);
        allocateConvolutionalLayerBuffers();
        setupKernels();
    }

    ConvolutionalLayer::ConvolutionalLayer(const std::shared_ptr<Utils::SharedResources> &p_sharedResources,
                                           const H5::Group &p_layerGroup,
                                           const size_t p_batchSize)
        : TrainableLayer(p_sharedResources, p_layerGroup, p_batchSize)
    {
        m_filterDimensions = Utils::FilterDimensions(Utils::readVectorFromHDF5<size_t>(p_layerGroup, "filterDimensions"));
        m_strideDimensions = Utils::StrideDimensions(Utils::readVectorFromHDF5<size_t>(p_layerGroup, "strideDimensions"));
        m_paddingValues = Utils::PaddingValues(Utils::readVectorFromHDF5<size_t>(p_layerGroup, "paddingValues"));
        m_paddingType = Utils::paddingTypeFromUint(Utils::readValueFromHDF5<unsigned int>(p_layerGroup, "paddingType"));
        m_weights = Utils::loadBuffer(p_sharedResources->getContext(), p_layerGroup, "weights", getWeightsSize());
        m_biases = Utils::loadBuffer(p_sharedResources->getContext(), p_layerGroup, "biases", getBiasesSize());
        allocateConvolutionalLayerBuffers();
        setupKernels();
    }

    cl::Event ConvolutionalLayer::runForward(const cl::CommandQueue &p_forwardBackpropQueue,
                                             const cl::Buffer &p_inputs,
                                             const size_t p_batchSize)
    {
        if (m_batchSize < p_batchSize)
        {
            setBatchSize(p_batchSize);
        }

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
            &raw_queue, nullptr);

        if (status != clblast::StatusCode::kSuccess)
        {
            CLNN_FATAL("CLBlast Convgemm failed with status: " + std::to_string(static_cast<int>(status)));
        }

        cl::Event returnEvent;
        cl::NDRange globalSize(getOutputChannels(), getOutputHeight() * getOutputWidth(), p_batchSize);

        cl_int err = p_forwardBackpropQueue.enqueueNDRangeKernel(
            m_biasKernel,
            cl::NullRange,
            globalSize,
            cl::NullRange,
            nullptr,
            &returnEvent);

        if (err != CL_SUCCESS)
        {
            CLNN_FATAL("Failed to enqueue bias addition kernel.");
        }

        return returnEvent;
    }

    cl::Event ConvolutionalLayer::backpropDeltas(
        const cl::CommandQueue &p_forwardBackpropQueue,
        const cl::Buffer &p_previousLayerDeltas,
        size_t p_batchSize)
    {
        if (m_batchSize < p_batchSize)
        {
            setBatchSize(p_batchSize);
        }

        size_t globalWidth = (getInputWidth() + 1) / 2;

        cl::NDRange globalSize(
            globalWidth,
            getInputHeight(),
            getInputChannels() * p_batchSize);
        Utils::setKernelArgs(14, m_backpropDeltasKernel, p_previousLayerDeltas);

        cl::Event executionEvent;
        cl_int err = p_forwardBackpropQueue.enqueueNDRangeKernel(
            m_backpropDeltasKernel,
            cl::NullRange,
            globalSize,
            cl::NullRange,
            nullptr,
            &executionEvent);
        if (err != CL_SUCCESS)
        {
            CLNN_FATAL("Failed to enqueue backprop deltas kernel." + std::to_string(err));
        }

        return executionEvent;
    }

    std::pair<cl::Event, cl::Event> ConvolutionalLayer::computeGradients(
        const cl::CommandQueue &p_deltaToGradientQueue,
        cl::Event p_backpropEvent,
        const cl::Buffer &p_inputs,
        const size_t p_batchSize)
    {
        if (m_batchSize < p_batchSize)
        {
            setBatchSize(p_batchSize);
        }

        std::vector<cl::Event> waitList;
        if (p_backpropEvent() != nullptr)
        {
            waitList.push_back(p_backpropEvent);
        }

        cl::NDRange globalSize(
            m_filterDimensions.getWidth(),
            m_filterDimensions.getHeight(),
            getInputChannels() * getOutputChannels());

        Utils::setKernelArgs(14, m_computeWeightsGradientsKernel, p_inputs, static_cast<int>(p_batchSize));

        cl::Event weightsEvent;
        p_deltaToGradientQueue.enqueueNDRangeKernel(
            m_computeWeightsGradientsKernel,
            cl::NullRange,
            globalSize,
            cl::NullRange,
            &waitList,
            &weightsEvent);

        cl::NDRange biasGlobalSize(getOutputChannels());
        cl::Event biasEvent;
        Utils::setKernelArgs(5, m_computeBiasesGradientsKernel, static_cast<cl_int>(p_batchSize));
        p_deltaToGradientQueue.enqueueNDRangeKernel(
            m_computeBiasesGradientsKernel,
            cl::NullRange,
            biasGlobalSize,
            cl::NullRange,
            &waitList,
            &biasEvent);
        return {weightsEvent, biasEvent};
    }

    void ConvolutionalLayer::allocateConvolutionalLayerBuffers()
    {
        m_weightsGradients = cl::Buffer(
            m_sharedResources->getContext(),
            CL_MEM_READ_WRITE,
            (getWeightsSize()) * sizeof(float));

        m_biasesGradients = cl::Buffer(
            m_sharedResources->getContext(),
            CL_MEM_READ_WRITE,
            (getBiasesSize()) * sizeof(float));
    }

    Utils::Dimensions ConvolutionalLayer::calculateOutputDimensions(const Utils::Dimensions &p_inputDimensions, const Utils::FilterDimensions &p_filterDimensions, const Utils::StrideDimensions &p_strideDimensions, Utils::PaddingType p_paddingType)
    {
        Utils::PaddingValues paddingValues = calculatePaddingValues(p_inputDimensions, p_filterDimensions, p_strideDimensions, p_paddingType);
        size_t inputHeight = p_inputDimensions.getDimensions()[1];
        size_t inputWidth = p_inputDimensions.getDimensions()[2];
        size_t filterHeight = p_filterDimensions.getHeight();
        size_t filterWidth = p_filterDimensions.getWidth();
        size_t strideHeight = p_strideDimensions.getHeight();
        size_t strideWidth = p_strideDimensions.getWidth();
        size_t padTop = paddingValues.getTop();
        size_t padLeft = paddingValues.getLeft();
        size_t padBottom = paddingValues.getBottom();
        size_t padRight = paddingValues.getRight();

        auto numeratorHeight = static_cast<long>(inputHeight - filterHeight + padTop + padBottom);
        auto numeratorWidth = static_cast<long>(inputWidth - filterWidth + padLeft + padRight);

        auto outputHeight = static_cast<size_t>(
            floor(static_cast<double>(numeratorHeight) / static_cast<double>(strideHeight)) + 1);

        auto outputWidth = static_cast<size_t>(
            floor(static_cast<double>(numeratorWidth) / static_cast<double>(strideWidth)) + 1);

        if (outputHeight == 0 || outputWidth == 0)
        {
            std::cerr << "Error: Calculated output dimensions are invalid (zero)." << "\n";
            CLNN_FATAL("Calculated output dimensions are invalid (zero). Check filter, stride, and padding settings.");
        }

        size_t outputChannels = p_filterDimensions.getOutputChannels();
        return Utils::Dimensions({outputChannels, outputHeight, outputWidth});
    }

    Utils::Dimensions ConvolutionalLayer::calculateOutputDimensions() const
    {
        auto inputHeight = static_cast<long>(getInputHeight());
        auto inputWidth = static_cast<long>(getInputWidth());
        auto filterHeight = static_cast<long>(m_filterDimensions.getHeight());
        auto filterWidth = static_cast<long>(m_filterDimensions.getWidth());
        auto strideHeight = static_cast<long>(m_strideDimensions.getHeight());
        auto strideWidth = static_cast<long>(m_strideDimensions.getWidth());
        auto padTop = static_cast<long>(m_paddingValues.getTop());
        auto padLeft = static_cast<long>(m_paddingValues.getLeft());
        auto padBottom = static_cast<long>(m_paddingValues.getBottom());
        auto padRight = static_cast<long>(m_paddingValues.getRight());

        long numeratorHeight = inputHeight - filterHeight + padTop + padBottom;
        long numeratorWidth = inputWidth - filterWidth + padLeft + padRight;

        auto outputHeight = static_cast<size_t>(
            floor(static_cast<double>(numeratorHeight) / static_cast<double>(strideHeight)) + 1);

        auto outputWidth = static_cast<size_t>(
            floor(static_cast<double>(numeratorWidth) / static_cast<double>(strideWidth)) + 1);

        if (outputHeight == 0 || outputWidth == 0)
        {
            std::cerr << "Error: Calculated output dimensions are invalid (zero)." << "\n";
            CLNN_FATAL("Calculated output dimensions are invalid (zero). Check filter, stride, and padding settings.");
        }

        size_t outputChannels = m_filterDimensions.getOutputChannels();
        return Utils::Dimensions({outputChannels, outputHeight, outputWidth});
    }

    Utils::PaddingValues ConvolutionalLayer::calculatePaddingValues(
        const Utils::Dimensions &p_inputDimensions,
        const Utils::FilterDimensions &p_filterDimensions,
        const Utils::StrideDimensions &p_strideDimensions,
        const Utils::PaddingType p_paddingType)
    {
        size_t inputHeight = p_inputDimensions.getDimensions()[1];
        size_t inputWidth = p_inputDimensions.getDimensions()[2];

        switch (p_paddingType)
        {
        case Utils::PaddingType::Valid:
        {
            return {0, 0, 0, 0};
        }
        case Utils::PaddingType::Same:
        {
            auto inputH_l = static_cast<long long>(inputHeight);
            auto inputW_l = static_cast<long long>(inputWidth);
            auto filterH_l = static_cast<long long>(p_filterDimensions.getHeight());
            auto filterW_l = static_cast<long long>(p_filterDimensions.getWidth());
            auto strideH_l = static_cast<long long>(p_strideDimensions.getHeight());
            auto strideW_l = static_cast<long long>(p_strideDimensions.getWidth());

            long long outputHeight = (inputH_l + strideH_l - 1) / strideH_l;
            long long outputWidth = (inputW_l + strideW_l - 1) / strideW_l;

            long long totalPaddingHeight_l = ((outputHeight - 1) * strideH_l) + filterH_l - inputH_l;
            long long totalPaddingWidth_l = ((outputWidth - 1) * strideW_l) + filterW_l - inputW_l;

            if (totalPaddingHeight_l < 0 || totalPaddingWidth_l < 0)
            {
                std::cerr << "Error: Invalid convolution configuration. Same padding is insufficient." << "\n";
                std::cerr << "Required Padding H: " << totalPaddingHeight_l << ", W: " << totalPaddingWidth_l << "\n";
                CLNN_FATAL("Input dimensions are too small for filter/stride combination, even with 'Same' padding.");
            }

            auto totalPaddingHeight = static_cast<size_t>(totalPaddingHeight_l);
            auto totalPaddingWidth = static_cast<size_t>(totalPaddingWidth_l);

            size_t padTop = totalPaddingHeight / 2;
            size_t padBottom = totalPaddingHeight - padTop;
            size_t padLeft = totalPaddingWidth / 2;
            size_t padRight = totalPaddingWidth - padLeft;

            return {padTop, padBottom, padLeft, padRight};
        }
        default:
        {
            std::cerr << "Warning: Unsupported padding type. Setting padding to zero." << "\n";
            return {0, 0, 0, 0};
        }
        }
    }

    void ConvolutionalLayer::initializeWeightsAndBiases(std::mt19937 &p_rng)
    {
        std::vector<float> h_weights(getWeightsSize());
        std::vector<float> h_biases(getBiasesSize());

        auto fan = static_cast<float>(m_filterDimensions.getHeight() * m_filterDimensions.getWidth());
        float limit = std::sqrt(6.0F / (getInputChannels() + getOutputChannels()) * fan);

        for (auto &weight : h_weights)
        {
            weight = getRandomValue(-limit, limit, p_rng);
        }

        for (auto &bias : h_biases)
        {
            bias = 0.0F;
        }

        m_weights = cl::Buffer(
            m_sharedResources->getContext(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            h_weights.size() * sizeof(float),
            h_weights.data());

        m_biases = cl::Buffer(
            m_sharedResources->getContext(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            h_biases.size() * sizeof(float),
            h_biases.data());
    }

    void ConvolutionalLayer::setupKernels()
    {
        setupTrainableKernels();
        cl_int err;

        m_biasKernel = cl::Kernel(m_sharedResources->getProgram(), "convolutionalBias", &err);
        if (err != CL_SUCCESS)
        {
            CLNN_FATAL("Failed to create convBias kernel");
        }
        Utils::setKernelArgs(m_biasKernel,
                             getBiases(),
                             getOutputs(),
                             static_cast<cl_int>(getOutputHeight()),
                             static_cast<cl_int>(getOutputWidth()),
                             static_cast<cl_int>(getOutputChannels()));
        m_backpropDeltasKernel = cl::Kernel(m_sharedResources->getProgram(), "convolutionalBackpropDeltas", &err);

        if (err != CL_SUCCESS)
        {
            CLNN_FATAL("Failed to create backprop kernel.");
        }
        Utils::setKernelArgs(m_backpropDeltasKernel,
                             getWeights(),
                             getDeltas(),
                             static_cast<cl_int>(getInputHeight()),
                             static_cast<cl_int>(getInputWidth()),
                             static_cast<cl_int>(getOutputHeight()),
                             static_cast<cl_int>(getOutputWidth()),
                             static_cast<cl_int>(m_filterDimensions.getHeight()),
                             static_cast<cl_int>(m_filterDimensions.getWidth()),
                             static_cast<cl_int>(m_strideDimensions.getHeight()),
                             static_cast<cl_int>(m_strideDimensions.getWidth()),
                             static_cast<cl_int>(m_paddingValues.getTop()),
                             static_cast<cl_int>(m_paddingValues.getLeft()),
                             static_cast<cl_int>(getInputChannels()),
                             static_cast<cl_int>(getOutputChannels()));

        m_computeWeightsGradientsKernel = cl::Kernel(m_sharedResources->getProgram(), "convolutionalComputeWeightsGradients", &err);
        if (err != CL_SUCCESS)
        {
            CLNN_FATAL("Failed to create compute weights gradients kernel.");
        }

        Utils::setKernelArgs(m_computeWeightsGradientsKernel,
                             getDeltas(),
                             getWeightsGradients(),
                             static_cast<cl_int>(getInputChannels()),
                             static_cast<cl_int>(getInputHeight()),
                             static_cast<cl_int>(getInputWidth()),
                             static_cast<cl_int>(getOutputChannels()),
                             static_cast<cl_int>(getOutputHeight()),
                             static_cast<cl_int>(getOutputWidth()),
                             static_cast<cl_int>(m_filterDimensions.getHeight()),
                             static_cast<cl_int>(m_filterDimensions.getWidth()),
                             static_cast<cl_int>(m_strideDimensions.getHeight()),
                             static_cast<cl_int>(m_strideDimensions.getWidth()),
                             static_cast<cl_int>(m_paddingValues.getTop()),
                             static_cast<cl_int>(m_paddingValues.getLeft()));

        m_computeBiasesGradientsKernel = cl::Kernel(m_sharedResources->getProgram(), "convolutionalComputeBiasesGradients", &err);
        if (err != CL_SUCCESS)
        {
            CLNN_FATAL("Failed to create compute biases gradients kernel.");
        }

        Utils::setKernelArgs(
            m_computeBiasesGradientsKernel,
            getDeltas(),
            getBiasesGradients(),
            static_cast<cl_int>(getOutputChannels()),
            static_cast<cl_int>(getOutputHeight()),
            static_cast<cl_int>(getOutputWidth()));
    }

    Utils::Dimensions ConvolutionalLayer::validateInputDimensions(
        const Utils::Dimensions &p_inputDimensions,
        const Utils::FilterDimensions &p_filterDimensions,
        const Utils::StrideDimensions &p_strideDimensions)
    {
        std::vector<size_t> dims = p_inputDimensions.getDimensions();
        size_t initial_dims = dims.size();
        Utils::Dimensions validDimensions;

        if (initial_dims == 1)
        {
            validDimensions = Utils::Dimensions({dims[0], 1, 1});
        }
        else if (initial_dims == 2)
        {
            validDimensions = Utils::Dimensions({dims[0], dims[1], 1});
        }
        else if (initial_dims == 3)
        {
            validDimensions = p_inputDimensions;
        }
        else
        {
            std::cerr << "Error: Input dimensions must be 1D, 2D, or 3D (Channels, Height, Width)." << "\n";
            CLNN_FATAL("Input dimensions must be 1D, 2D, or 3D.");
        }

        if (p_filterDimensions.getInputChannels() != validDimensions.getDimensions()[0])
        {
            std::cerr << "Error: Filter's input channels (" << p_filterDimensions.getInputChannels()
                      << ") do not match the input volume's channels (" << validDimensions.getDimensions()[0] << ")." << "\n";
            CLNN_FATAL("Input channels of filter dimensions must match the channels of input dimensions.");
        }

        if (p_filterDimensions.getHeight() <= 0 || p_filterDimensions.getWidth() <= 0)
        {
            std::cerr << "Error: Filter dimensions (" << p_filterDimensions.getHeight() << "x" << p_filterDimensions.getWidth()
                      << ") must be strictly positive integers (> 0)." << "\n";
            CLNN_FATAL("Filter dimensions must be strictly positive.");
        }

        if (p_strideDimensions.getHeight() <= 0 || p_strideDimensions.getWidth() <= 0)
        {
            std::cerr << "Error: Stride dimensions ("
                      << p_strideDimensions.getHeight() << "x" << p_strideDimensions.getWidth()
                      << ") must be strictly positive integers (> 0)." << "\n";
            CLNN_FATAL("Stride dimensions must be strictly positive.");
        }

        return validDimensions;
    }
}