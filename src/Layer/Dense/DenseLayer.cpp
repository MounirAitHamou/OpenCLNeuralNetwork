#include "Layer/Dense/DenseLayer.hpp"

DenseLayer::DenseLayer(const size_t p_layerId, 
                       std::shared_ptr<Utils::SharedResources> p_sharedResources,
                       const Utils::Dimensions& p_inputDimensions, 
                       const Utils::Dimensions& p_outputDimensions,
                       const Utils::ActivationType p_activationType, 
                       const size_t p_batchSize,
                       std::mt19937& p_rng)
                       :
                       TrainableLayer(p_layerId, 
                                      p_sharedResources,
                                      p_inputDimensions,
                                      Utils::Dimensions::validateDenseDimensions(p_outputDimensions),
                                      p_activationType,
                                      p_batchSize) {
    initializeWeightsAndBiases(p_rng);
    allocateDenseLayerBuffers();
    setupKernels();
}

DenseLayer::DenseLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
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

    H5::DataSet outputDataset = p_layerGroup.openDataSet("outputDimensions");
    H5::DataSpace outputSpace = outputDataset.getSpace();

    hsize_t outputDims[1];
    outputSpace.getSimpleExtentDims(outputDims);

    std::vector<size_t> outputDimensionsVec(outputDims[0]);
    outputDataset.read(outputDimensionsVec.data(), H5::PredType::NATIVE_HSIZE);
    m_outputDimensions = Utils::Dimensions(outputDimensionsVec);

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

    size_t flatInputSize = m_inputDimensions.getTotalElements();
    size_t flatOutputSize = m_outputDimensions.getTotalElements();
        
    m_weights          = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, flatOutputSize * flatInputSize * sizeof(float), loadedWeights.data());
    m_biases           = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, flatOutputSize * sizeof(float), loadedBiases.data());
    allocateLayerBuffers();
    allocateDenseLayerBuffers();
    setupKernels();
}

cl::Event DenseLayer::runForward(const cl::CommandQueue& p_forwardBackpropQueue,
                                 const cl::Buffer& p_inputs) {
    size_t flatInputSize = m_inputDimensions.getTotalElements();
    size_t flatOutputSize = m_outputDimensions.getTotalElements();
    
    cl_command_queue raw_queue = p_forwardBackpropQueue.get();

    auto status = clblast::Gemm<float>(
        clblast::Layout::kRowMajor,
        clblast::Transpose::kNo, clblast::Transpose::kYes,
        m_batchSize, flatOutputSize, flatInputSize,
        NO_SCALAR,
        p_inputs(), NO_OFFSET, flatInputSize,
        getWeights()(), NO_OFFSET, flatInputSize,
        CLEAR_C,
        getPreActivations()(), NO_OFFSET, flatOutputSize,
        &raw_queue, nullptr,
        m_clblastWorkspace()
    );

    if (status != clblast::StatusCode::kSuccess) {
        std::cerr << "Forward CLBlast GEMM failed: " << static_cast<int>(status) << " for layer " << m_layerId << std::endl;
        throw std::runtime_error("CLBlast GEMM failed");
    }

    cl::Event kernelEvent;

    p_forwardBackpropQueue.enqueueNDRangeKernel(m_denseBiasActivationKernel, cl::NullRange,
                               cl::NDRange(flatOutputSize, m_batchSize), cl::NullRange, 
                               nullptr, &kernelEvent);

    return kernelEvent;
}

cl::Event DenseLayer::computeDeltas(const cl::CommandQueue& p_forwardBackpropQueue) {
    cl::Event kernelEvent;
    p_forwardBackpropQueue.enqueueNDRangeKernel(m_denseBackpropActivationKernel, cl::NullRange,
                               cl::NDRange(m_outputDimensions.getTotalElements(), m_batchSize), cl::NullRange,
                               nullptr, &kernelEvent);
    return kernelEvent;
}

void DenseLayer::backpropDeltas(const cl::CommandQueue& p_forwardBackpropQueue, const cl::Buffer& p_previousLayerDeltas, const Utils::Dimensions p_previousLayerOutputDimensions) {
    size_t previousLayerFlatOutputSize = p_previousLayerOutputDimensions.getTotalElements();
    size_t flatOutputSize = m_outputDimensions.getTotalElements();

    cl_command_queue raw_queue = p_forwardBackpropQueue.get();

    if (m_clblastDeltaWorkspace() == nullptr) {
        size_t requiredWorkspaceSize = m_maxBatchSize * flatOutputSize * sizeof(float);

        m_clblastDeltaWorkspace = cl::Buffer(
            m_sharedResources->getContext(),
            CL_MEM_READ_WRITE,
            requiredWorkspaceSize
        );
    }

    auto status = clblast::Gemm<float>(
        clblast::Layout::kRowMajor,
        clblast::Transpose::kNo, clblast::Transpose::kNo,
        m_batchSize, previousLayerFlatOutputSize, flatOutputSize,
        NO_SCALAR,
        m_deltas.get(), NO_OFFSET, flatOutputSize,
        m_weights.get(), NO_OFFSET, previousLayerFlatOutputSize,
        CLEAR_C,
        p_previousLayerDeltas.get(), NO_OFFSET, previousLayerFlatOutputSize,
        &raw_queue, nullptr,
        m_clblastDeltaWorkspace()
    );

    if (status != clblast::StatusCode::kSuccess) {
        std::cerr << "Backprop CLBlast GEMM failed: " << static_cast<int>(status) << " for layer " << m_layerId << std::endl;
        throw std::runtime_error("CLBlast GEMM failed");
    }
}

std::pair<cl::Event, cl::Event> DenseLayer::computeGradients(const cl::CommandQueue& p_deltaToGradientQueue, 
                                                               const cl::CommandQueue& p_concurrentQueue, 
                                                               cl::Event& p_deltaEvent, 
                                                               const cl::Buffer& p_inputs) {
    std::vector<cl::Event> deltaBackPropWaitList = { p_deltaEvent };

    p_deltaToGradientQueue.enqueueBarrierWithWaitList(&deltaBackPropWaitList);

    size_t flatInputSize = m_inputDimensions.getTotalElements();
    size_t flatOutputSize = m_outputDimensions.getTotalElements();
    
    cl_event raw_gemm_event = nullptr;
    cl_command_queue raw_queue = p_deltaToGradientQueue.get();

    auto status = clblast::Gemm<float>(
        clblast::Layout::kRowMajor,
        clblast::Transpose::kYes,
        clblast::Transpose::kNo,
        flatOutputSize, flatInputSize, m_batchSize,
        NO_SCALAR,
        getDeltas().get(), NO_OFFSET, flatOutputSize,
        p_inputs.get(), NO_OFFSET, flatInputSize,
        CLEAR_C,
        getWeightsGradients().get(), NO_OFFSET, flatInputSize,
        &raw_queue,
        &raw_gemm_event,
        m_clblastWorkspace()
    );

    if (status != clblast::StatusCode::kSuccess) {
        std::cerr << "Weight Gradients CLBlast GEMM failed: " << static_cast<int>(status) << " for layer " << m_layerId << std::endl;
        throw std::runtime_error("CLBlast GEMM failed");
    }

    cl::Event gemmEvent(raw_gemm_event, true);

    cl_event raw_gemv_event = nullptr;

    
    status = clblast::Gemv<float>(
        clblast::Layout::kRowMajor,
        clblast::Transpose::kYes,
        m_batchSize, flatOutputSize,
        NO_SCALAR,
        getDeltas().get(), NO_OFFSET, flatOutputSize,
        m_onesBuffer(), NO_OFFSET, 1,
        CLEAR_C,
        getBiasesGradients().get(), NO_OFFSET, 1,
        &raw_queue,
        &raw_gemv_event
    );

    if (status != clblast::StatusCode::kSuccess) {
        std::cerr << "Bias Gradients CLBlast GEMM failed: " << static_cast<int>(status) << " for layer " << m_layerId << std::endl;
        throw std::runtime_error("CLBlast GEMV failed");
    }

    cl::Event gemvEvent(raw_gemv_event, true);

    m_denseAverageWeightGradientsKernel.setArg(2, (cl_uint)m_batchSize);

    cl::Event weightGradientsAverageEvent;
    std::vector<cl::Event> weightGradientsAverageWaitList = {gemmEvent};
    
    p_concurrentQueue.enqueueNDRangeKernel(m_denseAverageWeightGradientsKernel, cl::NullRange,
                                    cl::NDRange(flatOutputSize, flatInputSize), cl::NullRange, &weightGradientsAverageWaitList, &weightGradientsAverageEvent);
    
    m_denseAverageBiasGradientsKernel.setArg(1, (cl_uint)m_batchSize);
    
    cl::Event biasGradientsAverageEvent;
    std::vector<cl::Event> biasGradientsAverageWaitList = {gemvEvent};

    p_concurrentQueue.enqueueNDRangeKernel(m_denseAverageBiasGradientsKernel, cl::NullRange,
                                    cl::NDRange(flatOutputSize), cl::NullRange, &biasGradientsAverageWaitList, &biasGradientsAverageEvent);

    return { std::move(weightGradientsAverageEvent), std::move(biasGradientsAverageEvent) };
}

void DenseLayer::saveLayer(const cl::CommandQueue& p_queue, 
                           H5::Group& p_layerGroup) const {
    H5::DataSpace scalarSpace(H5S_SCALAR);

    p_layerGroup.createAttribute("layerId", H5::PredType::NATIVE_HSIZE, scalarSpace)
        .write(H5::PredType::NATIVE_HSIZE, &m_layerId);

    unsigned int layerType = static_cast<unsigned int>(getType());
    p_layerGroup.createAttribute("layerType", H5::PredType::NATIVE_UINT, scalarSpace)
        .write(H5::PredType::NATIVE_UINT, &layerType);

    std::vector<hsize_t> inputDimensionsVec = m_inputDimensions.getDimensions();
    hsize_t inputDims[1] = { inputDimensionsVec.size() };
    H5::DataSpace inputDataspace(1, inputDims);

    H5::DataSet inputDataset = p_layerGroup.createDataSet(
        "inputDimensions", 
        H5::PredType::NATIVE_HSIZE, 
        inputDataspace
    );
    inputDataset.write(inputDimensionsVec.data(), H5::PredType::NATIVE_HSIZE);

    std::vector<hsize_t> outputDimensionsVec = m_outputDimensions.getDimensions();
    hsize_t outputDims[1] = { outputDimensionsVec.size() };
    H5::DataSpace outputDataspace(1, outputDims);
    H5::DataSet outputDataset = p_layerGroup.createDataSet(
        "outputDimensions", 
        H5::PredType::NATIVE_HSIZE, 
        outputDataspace
    );
    outputDataset.write(outputDimensionsVec.data(), H5::PredType::NATIVE_HSIZE);

    unsigned int activationTypeUInt = static_cast<unsigned int>(m_activationType);
    p_layerGroup.createAttribute(
        "activationType", H5::PredType::NATIVE_UINT, scalarSpace
    ).write(H5::PredType::NATIVE_UINT, &activationTypeUInt);
    
    Utils::saveBuffer(p_queue, m_weights, p_layerGroup, "weights", getWeightsSize());
    Utils::saveBuffer(p_queue, m_biases, p_layerGroup, "biases", getBiasesSize()); 
}


bool DenseLayer::equals(const cl::CommandQueue& p_queue, const Layer& p_other) const {
    if (p_other.getType() != Utils::LayerType::Dense) {
        return false;
    }

    const DenseLayer& otherDense = static_cast<const DenseLayer&>(p_other);

    if (m_layerId != otherDense.m_layerId ||
        m_inputDimensions != otherDense.m_inputDimensions ||
        m_outputDimensions != otherDense.m_outputDimensions ||
        m_activationType != otherDense.m_activationType ||
        m_batchSize != otherDense.m_batchSize ||
        !Utils::compareCLBuffers(p_queue, m_weights, otherDense.m_weights, getWeightsSize()) ||
        !Utils::compareCLBuffers(p_queue, m_biases, otherDense.m_biases, getBiasesSize())){
        return false;
    }

    return true;
}

void DenseLayer::print(const cl::CommandQueue& p_forwardBackpropQueue) const {
    std::cout << "----- Dense Layer Info -----\n";
    std::cout << "Dense Layer ID: " << m_layerId << "\n";
    std::cout << "Input Dimensions: " << m_inputDimensions.toString() << "\n";
    std::cout << "Output Dimensions: " << m_outputDimensions.toString() << "\n";
    std::cout << "Activation Type: " << static_cast<unsigned int>(m_activationType) << "\n";
    std::cout << "Weights Size: " << getWeightsSize() << "\n";
    std::cout << "Biases Size: " << getBiasesSize() << "\n";
    Utils::printCLBuffer(p_forwardBackpropQueue, m_weights, getWeightsSize(), "Weights");
    Utils::printCLBuffer(p_forwardBackpropQueue, m_biases, getBiasesSize(), "Biases");
    Utils::printCLBuffer(p_forwardBackpropQueue, m_preActivations, m_batchSize * getTotalOutputElements(), "Pre-activations");
    Utils::printCLBuffer(p_forwardBackpropQueue, m_outputs, m_batchSize * getTotalOutputElements(), "Outputs");
    Utils::printCLBuffer(p_forwardBackpropQueue, m_deltas, m_batchSize * getTotalOutputElements(), "Deltas");
    Utils::printCLBuffer(p_forwardBackpropQueue, m_weightsGradients, getWeightsSize(), "Weight Gradients");
    Utils::printCLBuffer(p_forwardBackpropQueue, m_biasesGradients, getBiasesSize(), "Bias Gradients");
    std::cout << "----------------------------------------\n";
}

void DenseLayer::allocateDenseLayerBuffers() {
    m_weightsGradients = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE, getWeightsSize() * sizeof(float));
    
    m_biasesGradients  = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE, getBiasesSize() * sizeof(float));
    
    m_preActivations   = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE, m_batchSize * getTotalOutputElements() * sizeof(float));

    m_onesBuffer        = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, m_batchSize * sizeof(float), std::vector<float>(m_batchSize, 1.0f).data());
    
    size_t flatInputSize = m_inputDimensions.getTotalElements();
    size_t flatOutputSize = m_outputDimensions.getTotalElements();

    size_t requiredWorkspaceSize = std::max({
        m_batchSize * flatOutputSize,
        flatOutputSize * flatInputSize
    });

    m_clblastWorkspace = cl::Buffer(
        m_sharedResources->getContext(),
        CL_MEM_READ_WRITE,
        requiredWorkspaceSize * sizeof(float)
    );
}

void DenseLayer::initializeWeightsAndBiases(std::mt19937& p_rng) {
    std::vector<float> h_weights(getWeightsSize());
    std::vector<float> h_biases(getBiasesSize());

    float fanIn = (float)getTotalInputElements();
    float fanOut = (float)getTotalOutputElements();
    float limit;

    if (Utils::isReluActivation(m_activationType)) {
        limit = std::sqrt(6.0f / fanIn);
    } else {
        limit = std::sqrt(6.0f / (fanIn + fanOut));
    }

    for (auto& weight : h_weights) {
        weight = getRandomValue(-limit, limit, p_rng);
    }
    
    for (auto& bias : h_biases) {
        bias = 0.0f;
    }

    m_weights = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_weights.size() * sizeof(float), h_weights.data());
    m_biases  = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, h_biases.size() * sizeof(float), h_biases.data());
}

void DenseLayer::setupKernels() {
    cl_int err;

    m_denseBiasActivationKernel = cl::Kernel(m_sharedResources->getProgram(), "denseBiasActivation", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create denseBiasActivation kernel");
    }
    m_denseBiasActivationKernel.setArg(0, getPreActivations());
    m_denseBiasActivationKernel.setArg(1, getBiases());
    m_denseBiasActivationKernel.setArg(2, getOutputs());
    m_denseBiasActivationKernel.setArg(3, (cl_uint)getTotalOutputElements());
    m_denseBiasActivationKernel.setArg(4, (cl_uint)getActivationType());

    m_denseBackpropActivationKernel = cl::Kernel(m_sharedResources->getProgram(), "denseBackpropActivation", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create denseBackpropActivation kernel");
    }
    m_denseBackpropActivationKernel.setArg(0, getDeltas());
    m_denseBackpropActivationKernel.setArg(1, getPreActivations());
    m_denseBackpropActivationKernel.setArg(2, (cl_uint)getTotalOutputElements());
    m_denseBackpropActivationKernel.setArg(3, (cl_uint)getActivationType());

    m_denseAverageWeightGradientsKernel = cl::Kernel(m_sharedResources->getProgram(), "denseAverageWeightsGradients", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create denseAverageWeightsGradients kernel");
    }
    m_denseAverageWeightGradientsKernel.setArg(0, getWeightsGradients());
    m_denseAverageWeightGradientsKernel.setArg(1, (cl_uint)getTotalInputElements());

    m_denseAverageBiasGradientsKernel = cl::Kernel(m_sharedResources->getProgram(), "denseAverageBiasesGradients", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create denseAverageBiasesGradients kernel");
    }
    m_denseAverageBiasGradientsKernel.setArg(0, getBiasesGradients());
}