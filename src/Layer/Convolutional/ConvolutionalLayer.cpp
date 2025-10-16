#include "Layer/Convolutional/ConvolutionalLayer.hpp"


ConvolutionalLayer::ConvolutionalLayer(const size_t p_layerId, 
                                       std::shared_ptr<Utils::SharedResources> p_sharedResources,
                                       const Utils::Dimensions& p_inputDimensions,
                                       const Utils::FilterDimensions& p_filterDimensions,
                                       const Utils::StrideDimensions& p_strideDimensions,
                                       const Utils::PaddingType p_paddingType,
                                       const Utils::ActivationType p_activationType, 
                                       const size_t p_batchSize,
                                       std::mt19937& p_rng)
    : 
    TrainableLayer(
        p_layerId, 
        p_sharedResources, 
        validateInputDimensions(
            p_inputDimensions,
            p_filterDimensions,
            p_strideDimensions
        ),
        calculateOutputDimensions(
            validateInputDimensions(
                p_inputDimensions,
                p_filterDimensions,
                p_strideDimensions
            ),
            p_filterDimensions, 
            p_strideDimensions,
            p_paddingType
        ),
        p_activationType, 
        p_batchSize
    ),
    m_filterDimensions(p_filterDimensions),
    m_strideDimensions(p_strideDimensions),
    m_paddingValues(
        calculatePaddingValues(
            m_inputDimensions,
            p_filterDimensions,
            p_strideDimensions,
            p_paddingType
        )
    )
{
    try {
        initializeWeightsAndBiases(p_rng);
        allocateConvolutionalLayerBuffers();
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
    
    H5::DataSet inputDataset = p_layerGroup.openDataSet("inputDimensions");
    H5::DataSpace inputSpace = inputDataset.getSpace();

    hsize_t inputDims[1];
    inputSpace.getSimpleExtentDims(inputDims);

    std::vector<size_t> inputDimensionsVec(inputDims[0]);
    inputDataset.read(inputDimensionsVec.data(), H5::PredType::NATIVE_HSIZE);

    m_inputDimensions = Utils::Dimensions(inputDimensionsVec);

    H5::DataSet filterDataset = p_layerGroup.openDataSet("filterDimensions");
    H5::DataSpace filterSpace = filterDataset.getSpace();

    hsize_t filterDims[1];
    filterSpace.getSimpleExtentDims(filterDims);

    std::vector<size_t> filterDimensionsVec(filterDims[0]);
    filterDataset.read(filterDimensionsVec.data(), H5::PredType::NATIVE_HSIZE);

    m_filterDimensions = Utils::FilterDimensions(filterDimensionsVec);

    H5::DataSet strideDataset = p_layerGroup.openDataSet("strideDimensions");
    H5::DataSpace strideSpace = strideDataset.getSpace();
    
    hsize_t strideDims[1];
    strideSpace.getSimpleExtentDims(strideDims);

    std::vector<size_t> strideDimensionsVec(strideDims[0]);
    strideDataset.read(strideDimensionsVec.data(), H5::PredType::NATIVE_HSIZE);

    m_strideDimensions = Utils::StrideDimensions(strideDimensionsVec);


    H5::DataSet paddingDataset = p_layerGroup.openDataSet("paddingValues");
    H5::DataSpace paddingSpace = paddingDataset.getSpace();

    hsize_t paddingDims[1];
    paddingSpace.getSimpleExtentDims(paddingDims);

    std::vector<size_t> paddingValuesVec(paddingDims[0]);
    paddingDataset.read(paddingValuesVec.data(), H5::PredType::NATIVE_HSIZE);

    m_paddingValues = Utils::PaddingValues(paddingValuesVec);

    m_outputDimensions = calculateOutputDimensions();

    H5::DataSet weightsDataset = p_layerGroup.openDataSet("weights");
    H5::DataSpace weightsSpace = weightsDataset.getSpace();
    hsize_t weightsDims[1];
    weightsSpace.getSimpleExtentDims(weightsDims);
    std::vector<float> loadedWeights(weightsDims[0]);
    weightsDataset.read(loadedWeights.data(), H5::PredType::NATIVE_FLOAT);

    H5::DataSet biasesDataset = p_layerGroup.openDataSet("biases");
    H5::DataSpace biasesSpace = biasesDataset.getSpace();
    hsize_t biasesDims[1];
    biasesSpace.getSimpleExtentDims(biasesDims);
    std::vector<float> loadedBiases(biasesDims[0]);
    biasesDataset.read(loadedBiases.data(), H5::PredType::NATIVE_FLOAT);

    m_weights = cl::Buffer(
        m_sharedResources->getContext(),
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        loadedWeights.size() * sizeof(float),
        loadedWeights.data()
    );

    m_biases = cl::Buffer(
        m_sharedResources->getContext(),
        CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
        loadedBiases.size() * sizeof(float),
        loadedBiases.data()
    );

    allocateLayerBuffers();
    allocateConvolutionalLayerBuffers();
    setupKernels();
}

cl::Event ConvolutionalLayer::runForward(const cl::CommandQueue& p_forwardBackpropQueue,
                                         const cl::Buffer& p_inputs) {
    m_im2colKernel.setArg(12, p_inputs);

    size_t im2colRows = (size_t)getInputChannels() * m_filterDimensions.getHeight() * m_filterDimensions.getWidth();
    size_t im2colCols = (size_t)getOutputHeight() * getOutputWidth();

    cl::NDRange globalWorkSize(im2colRows, im2colCols, m_batchSize);

    cl_int err = p_forwardBackpropQueue.enqueueNDRangeKernel(
        m_im2colKernel, 
        cl::NullRange, 
        globalWorkSize, 
        cl::NullRange);

    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue im2col kernel.");
    }
    cl_command_queue raw_queue = p_forwardBackpropQueue.get();

    size_t N = im2colCols * m_batchSize;

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
        getPreActivations()(), NO_OFFSET, N,
        &raw_queue, nullptr,
        m_clblastWorkspace()
    );

    if (status != clblast::StatusCode::kSuccess) {
        std::cerr << "Forward CLBlast GEMM failed: " << static_cast<int>(status) << std::endl;
        throw std::runtime_error("CLBlast GEMM failed");
    }

    cl::Event activationEvent;

    p_forwardBackpropQueue.enqueueNDRangeKernel(
        m_convBiasActivationKernel, 
        cl::NullRange,
        cl::NDRange(getTotalOutputElements(), m_batchSize), 
        cl::NullRange,
        nullptr,
        &activationEvent);


    return activationEvent;
}

cl::Event ConvolutionalLayer::computeDeltas(const cl::CommandQueue& p_forwardBackpropQueue) {
    cl::Event activationDerivativeEvent;
    p_forwardBackpropQueue.enqueueNDRangeKernel(
        m_convBackpropActivationKernel, 
        cl::NullRange,
        cl::NDRange(getTotalOutputElements() * m_batchSize),
        cl::NullRange,
        nullptr,
        &activationDerivativeEvent
    );
    
    return activationDerivativeEvent;
}


void ConvolutionalLayer::backpropDeltas(const cl::CommandQueue& p_forwardBackpropQueue, 
                                         const cl::Buffer& p_previousLayerDeltas, 
                                         const Utils::Dimensions p_previousLayerOutputDimensions) {
    size_t M = getOutputChannels();
    size_t N = getOutputHeight() * getOutputWidth() * m_batchSize;
    size_t K = getInputChannels() * m_filterDimensions.getHeight() * m_filterDimensions.getWidth();

    if (m_clblastDeltaWorkspace() == nullptr) {
        size_t maxN = m_maxBatchSize * getOutputHeight() * getOutputWidth();
        size_t requiredWorkspaceSize = K * maxN * sizeof(float); 

        m_clblastDeltaWorkspace = cl::Buffer(
            m_sharedResources->getContext(),
            CL_MEM_READ_WRITE,
            requiredWorkspaceSize
        );
    }

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


    p_forwardBackpropQueue.enqueueNDRangeKernel(
            m_col2imKernel, 
            cl::NullRange,
            cl::NDRange(p_previousLayerOutputDimensions.getTotalElements(), m_batchSize)
    );
}

std::pair<cl::Event, cl::Event> ConvolutionalLayer::computeGradients(const cl::CommandQueue& p_deltaToGradientQueue, 
                                                                         const cl::CommandQueue& p_concurrentQueue, 
                                                                         cl::Event& p_deltaEvent, 
                                                                         const cl::Buffer& p_inputs) {
    std::vector<cl::Event> deltaBackPropWaitList = { p_deltaEvent };
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
        m_batchSize * im2colCols,
        NO_SCALAR,
        getDeltas().get(), NO_OFFSET, m_batchSize * im2colCols,
        m_im2colBuffer(), NO_OFFSET, m_batchSize * im2colCols,
        CLEAR_C,
        getWeightsGradients().get(), NO_OFFSET, im2colRows,
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
        m_batchSize * im2colCols,
        NO_SCALAR,
        getDeltas().get(), NO_OFFSET, m_batchSize * im2colCols,
        m_onesBuffer(), NO_OFFSET, 1,
        CLEAR_C,
        getBiasesGradients().get(), NO_OFFSET, 1,
        &raw_queue,
        &raw_gemv_event
    );

    if (status != clblast::StatusCode::kSuccess) {
        std::cerr << "Bias Gradients CLBlast GEMM failed: " << static_cast<int>(status) << std::endl;
        throw std::runtime_error("CLBlast GEMV failed");
    }

    cl::Event gemvEvent(raw_gemv_event, true);

    cl::Event weightsGradientsAverageEvent;
    std::vector<cl::Event> weightsGradientsAverageWaitList = {gemmEvent};
    
    m_convAverageWeightsGradientsKernel.setArg(1, (cl_uint)m_batchSize);

    p_concurrentQueue.enqueueNDRangeKernel(m_convAverageWeightsGradientsKernel, cl::NullRange,
        cl::NDRange(getWeightsSize()), cl::NullRange, &weightsGradientsAverageWaitList, &weightsGradientsAverageEvent);
    
    cl::Event biasesGradientsAverageEvent;
    std::vector<cl::Event> biasesGradientsAverageWaitList = {gemvEvent};

    m_convAverageBiasesGradientsKernel.setArg(1, (cl_uint)m_batchSize);
    
    p_concurrentQueue.enqueueNDRangeKernel(m_convAverageBiasesGradientsKernel, cl::NullRange,
        cl::NDRange(getOutputChannels()), cl::NullRange, &biasesGradientsAverageWaitList, &biasesGradientsAverageEvent);

    return { std::move(weightsGradientsAverageEvent), std::move(biasesGradientsAverageEvent) };
}

void ConvolutionalLayer::saveLayer(const cl::CommandQueue& p_forwardBackpropQueue, 
                                 H5::Group& p_layerGroup) const {
    p_layerGroup.createAttribute("layerId", H5::PredType::NATIVE_HSIZE, H5::DataSpace(H5S_SCALAR)).write(H5::PredType::NATIVE_HSIZE, &m_layerId);

    unsigned int layerType = static_cast<unsigned int>(getType());
    p_layerGroup.createAttribute("layerType", H5::PredType::NATIVE_UINT, H5::DataSpace(H5S_SCALAR))
        .write(H5::PredType::NATIVE_UINT, &layerType);
    
    cl_uint activationTypeUInt = (cl_uint)m_activationType;
    p_layerGroup.createAttribute("activationType", H5::PredType::NATIVE_UINT, H5::DataSpace(H5S_SCALAR)).write(H5::PredType::NATIVE_UINT, &activationTypeUInt);
    
    hsize_t inputDims[1] = { m_inputDimensions.getDimensions().size() };
    p_layerGroup.createDataSet("inputDimensions", H5::PredType::NATIVE_HSIZE, H5::DataSpace(1,inputDims)).write(m_inputDimensions.getDimensions().data(), H5::PredType::NATIVE_HSIZE);
    
    hsize_t filterDims[1] = { m_filterDimensions.getDimensions().size() };
    p_layerGroup.createDataSet("filterDimensions", H5::PredType::NATIVE_HSIZE, H5::DataSpace(1, filterDims)).write(m_filterDimensions.getDimensions().data(), H5::PredType::NATIVE_HSIZE);

    hsize_t strideDims[1] = { m_strideDimensions.getDimensions().size() };
    p_layerGroup.createDataSet("strideDimensions", H5::PredType::NATIVE_HSIZE, H5::DataSpace(1, strideDims)).write(m_strideDimensions.getDimensions().data(), H5::PredType::NATIVE_HSIZE);
    
    hsize_t paddingDims[1] = { m_paddingValues.getDimensions().size() };
    p_layerGroup.createDataSet("paddingValues", H5::PredType::NATIVE_HSIZE, H5::DataSpace(1, paddingDims)).write(m_paddingValues.getDimensions().data(), H5::PredType::NATIVE_HSIZE);

    
    std::vector<float> h_weights(getWeightsSize());
    p_forwardBackpropQueue.enqueueReadBuffer(m_weights, CL_TRUE, 0, h_weights.size() * sizeof(float), h_weights.data());
    hsize_t weightsDims[1] = { h_weights.size() };
    p_layerGroup.createDataSet("weights", H5::PredType::NATIVE_FLOAT, H5::DataSpace(1, weightsDims)).write(h_weights.data(), H5::PredType::NATIVE_FLOAT);

    std::vector<float> h_biases(getBiasesSize());
    p_forwardBackpropQueue.enqueueReadBuffer(m_biases, CL_TRUE, 0, h_biases.size() * sizeof(float), h_biases.data());
    hsize_t biasesDims[1] = { h_biases.size() };
    p_layerGroup.createDataSet("biases", H5::PredType::NATIVE_FLOAT, H5::DataSpace(1, biasesDims)).write(h_biases.data(), H5::PredType::NATIVE_FLOAT);
}

bool ConvolutionalLayer::equals(const cl::CommandQueue& p_queue, const Layer& p_other) const {
    if (p_other.getType() != Utils::LayerType::Convolutional) {
        return false;
    }

    const ConvolutionalLayer& otherConv = static_cast<const ConvolutionalLayer&>(p_other);

    if (m_inputDimensions != otherConv.m_inputDimensions ||
        m_outputDimensions != otherConv.m_outputDimensions ||
        m_filterDimensions != otherConv.m_filterDimensions ||
        m_strideDimensions != otherConv.m_strideDimensions ||
        m_paddingValues != otherConv.m_paddingValues ||
        m_activationType != otherConv.m_activationType ||
        !Utils::compareCLBuffers(p_queue, m_weights, otherConv.m_weights, getWeightsSize()) ||
        !Utils::compareCLBuffers(p_queue, m_biases, otherConv.m_biases, getBiasesSize())){
        return false;
    }

    return true;
}

void ConvolutionalLayer::print(const cl::CommandQueue& p_forwardBackpropQueue) const {
    std::cout << "----- Convolutional Layer Info -----\n";
    std::cout << "Convolutional Layer ID: " << m_layerId << "\n";
    std::cout << "Input Dimensions: " << m_inputDimensions.toString() << "\n";
    std::cout << "Filter Dimensions: " << m_filterDimensions.toString() << "\n";
    std::cout << "Stride Dimensions: " << m_strideDimensions.toString() << "\n";
    std::cout << "Padding Values (Top, Bottom, Left, Right): (" 
              << m_paddingValues.getTop() << ", "
              << m_paddingValues.getBottom() << ", "
              << m_paddingValues.getLeft() << ", "
              << m_paddingValues.getRight() << ")\n";
    std::cout << "Output Dimensions: " << m_outputDimensions.toString() << "\n";
    std::cout << "Activation Type: " << static_cast<unsigned int>(m_activationType) << "\n";

    Utils::printCLBuffer(p_forwardBackpropQueue, m_weights, getWeightsSize(), "Weights");
    Utils::printCLBuffer(p_forwardBackpropQueue, m_biases, getBiasesSize(), "Biases");
    Utils::printCLBuffer(p_forwardBackpropQueue, m_preActivations, m_batchSize * getTotalOutputElements(), "Pre-Activations");
    Utils::printCLBuffer(p_forwardBackpropQueue, m_outputs, m_batchSize * getTotalOutputElements(), "Outputs");
    Utils::printCLBuffer(p_forwardBackpropQueue, m_deltas, m_batchSize * m_inputDimensions.getTotalElements(), "Deltas");
    Utils::printCLBuffer(p_forwardBackpropQueue, m_weightsGradients, getWeightsSize(), "Weight Gradients");
    Utils::printCLBuffer(p_forwardBackpropQueue, m_biasesGradients, getBiasesSize(), "Bias Gradients");
    std::cout << "-------------------------------------\n";
}

void ConvolutionalLayer::allocateConvolutionalLayerBuffers() {
    size_t im2colRows = getInputChannels() * m_filterDimensions.getHeight() * m_filterDimensions.getWidth();
    size_t im2colCols = getOutputHeight() * getOutputWidth();
    size_t im2colBatchCols = im2colCols * m_maxBatchSize;
    
    m_preActivations = cl::Buffer(
        m_sharedResources->getContext(), 
        CL_MEM_READ_WRITE, 
        (getOutputChannels()) * im2colBatchCols * sizeof(float)
    );

    m_deltas = cl::Buffer(
        m_sharedResources->getContext(), 
        CL_MEM_READ_WRITE, 
        (getOutputChannels()) * im2colBatchCols * sizeof(float)
    );

    m_weightsGradients = cl::Buffer(
        m_sharedResources->getContext(), 
        CL_MEM_READ_WRITE, 
        (getOutputChannels()) * im2colRows * sizeof(float)
    );
    
    m_biasesGradients = cl::Buffer(
        m_sharedResources->getContext(), 
        CL_MEM_READ_WRITE, 
        (getOutputChannels()) * sizeof(float)
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
        requiredWorkspaceSize * sizeof(float)
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

    size_t outputHeight   = static_cast<size_t>(floor((static_cast<double>(inputHeight - filterHeight) + padTop + padBottom) / static_cast<double>(strideHeight)) + 1);

    size_t outputWidth    = static_cast<size_t>(floor((static_cast<double>(inputWidth - filterWidth) + padLeft + padRight) / static_cast<double>(strideWidth)) + 1);
    size_t outputChannels = p_filterDimensions.getOutputChannels();
    return Utils::Dimensions({ outputChannels, outputHeight, outputWidth });
}

Utils::Dimensions ConvolutionalLayer::calculateOutputDimensions() const {
    size_t inputHeight    = getInputHeight();
    size_t inputWidth     = getInputWidth();
    size_t filterHeight   = m_filterDimensions.getHeight();
    size_t filterWidth    = m_filterDimensions.getWidth();
    size_t strideHeight   = m_strideDimensions.getHeight();
    size_t strideWidth    = m_strideDimensions.getWidth();
    size_t padTop         = m_paddingValues.getTop();
    size_t padLeft        = m_paddingValues.getLeft();
    size_t padBottom      = m_paddingValues.getBottom();
    size_t padRight       = m_paddingValues.getRight();

    size_t outputHeight   = static_cast<size_t>(floor((static_cast<double>(inputHeight - filterHeight) + padTop + padBottom) / static_cast<double>(strideHeight)) + 1);
    size_t outputWidth    = static_cast<size_t>(floor((static_cast<double>(inputWidth - filterWidth) + padLeft + padRight) / static_cast<double>(strideWidth)) + 1);
    size_t outputChannels = m_filterDimensions.getOutputChannels();
    return Utils::Dimensions({ outputChannels, outputHeight, outputWidth });
}

Utils::PaddingValues ConvolutionalLayer::calculatePaddingValues(const Utils::Dimensions& p_inputDimensions, const Utils::FilterDimensions& p_filterDimensions, const Utils::StrideDimensions& p_strideDimensions, const Utils::PaddingType p_paddingType) const {
    switch (p_paddingType) {
        case Utils::PaddingType::Valid: {
            return Utils::PaddingValues(0, 0, 0, 0);
        }
        case Utils::PaddingType::Same: {
            size_t outputHeight       = (p_inputDimensions.getDimensions()[1] + p_strideDimensions.getHeight() - 1) / p_strideDimensions.getHeight();
            size_t outputWidth        = (p_inputDimensions.getDimensions()[2] + p_strideDimensions.getWidth() - 1) / p_strideDimensions.getWidth();


            size_t totalPaddingHeight = (outputHeight - 1) * p_strideDimensions.getHeight() + p_filterDimensions.getHeight() - p_inputDimensions.getDimensions()[1];
            size_t totalPaddingWidth  = (outputWidth - 1) * p_strideDimensions.getWidth() + p_filterDimensions.getWidth() - p_inputDimensions.getDimensions()[2];

            size_t padTop             = totalPaddingHeight / 2;
            size_t padBottom          = totalPaddingHeight - padTop;
            size_t padLeft            = totalPaddingWidth / 2;
            size_t padRight           = totalPaddingWidth - padLeft;
            return Utils::PaddingValues(padTop, padBottom, padLeft, padRight);
            
        }
        default: {
            std::cerr << "Warning: Unsupported padding type. Setting padding to zero." << std::endl;
            return Utils::PaddingValues(0, 0, 0, 0);
        }
    }
}

void ConvolutionalLayer::initializeWeightsAndBiases(std::mt19937& p_rng) {
    std::vector<float> h_weights((getOutputChannels()) * ((getInputChannels()) * (m_filterDimensions.getHeight()) * (m_filterDimensions.getWidth())));
    std::vector<float> h_biases(getOutputChannels());

    float fanIn = (float)getInputChannels() * m_filterDimensions.getHeight() * m_filterDimensions.getWidth();
    float fanOut = (float)getOutputChannels() * m_filterDimensions.getHeight() * m_filterDimensions.getWidth();
    float limit = std::sqrt(6.0f / (fanIn + fanOut));
    
    for (auto& weight : h_weights) {
        weight = getRandomValue(-limit, limit, p_rng);
    }
    
    for (auto& bias : h_biases) {
        bias = getRandomValue(0.0f, 0.0f, p_rng);
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

    m_convBiasActivationKernel = cl::Kernel(m_sharedResources->getProgram(), "convolutionalBiasActivation", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create convBiasActivation kernel");
    }
    m_convBiasActivationKernel.setArg(0, getPreActivations());
    m_convBiasActivationKernel.setArg(1, getBiases());
    m_convBiasActivationKernel.setArg(2, getOutputs());
    m_convBiasActivationKernel.setArg(3, (cl_uint)getTotalOutputElements());
    m_convBiasActivationKernel.setArg(4, (cl_uint)getOutputChannels());
    m_convBiasActivationKernel.setArg(5, (cl_uint)getActivationType());

    m_convBackpropActivationKernel = cl::Kernel(m_sharedResources->getProgram(), "convolutionalBackpropActivation", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create convBackpropActivationKernel");
    }
    m_convBackpropActivationKernel.setArg(0, getDeltas());
    m_convBackpropActivationKernel.setArg(1, m_preActivations);
    m_convBackpropActivationKernel.setArg(2, (cl_uint)m_activationType);

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

    m_convAverageWeightsGradientsKernel = cl::Kernel(m_sharedResources->getProgram(), "convolutionalAverageWeightsGradientsKernel");
    if (err != CL_SUCCESS) {
        throw std::runtime_error("convAverageWeightsGradientsKernel");
    }
    m_convAverageWeightsGradientsKernel.setArg(0, m_weightsGradients);

    m_convAverageBiasesGradientsKernel = cl::Kernel(m_sharedResources->getProgram(), "convolutionalAverageBiasesGradientsKernel");
    if (err != CL_SUCCESS) {
        throw std::runtime_error("convAverageBiasesGradientsKernel");
    }
    m_convAverageBiasesGradientsKernel.setArg(0, m_biasesGradients);
}

Utils::Dimensions ConvolutionalLayer::validateInputDimensions(const Utils::Dimensions& p_inputDimensions, const Utils::FilterDimensions& p_filterDimensions, const Utils::StrideDimensions& p_strideDimensions) const {
    Utils::Dimensions validDimensions(p_inputDimensions.getDimensions());
    if (p_inputDimensions.getDimensions().size() == 1){
        validDimensions = Utils::Dimensions({p_inputDimensions.getDimensions()[0], 1, 1});
    }

    if (p_filterDimensions.getInputChannels() != validDimensions.getDimensions()[0]) {
        std::cerr << "Error: Input channels of filter dimensions (" << p_filterDimensions.getInputChannels() 
                  << ") do not match the channels of input dimensions (" << validDimensions.getDimensions()[0] << ")." << std::endl;
        throw std::invalid_argument("Input channels of filter dimensions must match the channels of input dimensions.");
    }

    if (validDimensions.getDimensions().size() != 3) {
        std::cerr << "Error: Input dimensions must be 1D, 2D, or 3D (channels, height, width)." << std::endl;
        throw std::invalid_argument("Input dimensions must be 1D, 2D, or 3D (channels, height, width).");
    }

    if (validDimensions.getDimensions()[1] < p_filterDimensions.getHeight() || 
        validDimensions.getDimensions()[2] < p_filterDimensions.getWidth()) {
        std::cerr << "Error: Filter dimensions ("
                  << p_filterDimensions.getHeight() << "x" << p_filterDimensions.getWidth() 
                  << ") exceed input dimensions (" 
                  << validDimensions.getDimensions()[1] << "x" << validDimensions.getDimensions()[2] << ")." 
                  << std::endl;
         throw std::invalid_argument("Filter dimensions must not exceed input dimensions.");
    }
    
    if (p_strideDimensions.getHeight() <= 0 || p_strideDimensions.getWidth() <= 0) {
        std::cerr << "Error: Stride dimensions (" 
                  << p_strideDimensions.getHeight() << "x" << p_strideDimensions.getWidth()
                  << ") must be strictly positive integers (> 0)." << std::endl;
        throw std::invalid_argument("Stride dimensions must be strictly positive.");
    }

    return validDimensions;
}
