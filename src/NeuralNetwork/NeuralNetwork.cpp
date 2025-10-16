#include "NeuralNetwork/NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(Utils::OpenCLResources&& p_oclResources, 
                             Utils::NetworkArgs p_networkArgs,
                             size_t p_seed)
    : m_batchSize(p_networkArgs.getBatchSize()),
      m_inputDimensions(p_networkArgs.getInitialInputDimensions()),
      m_lossFunctionType(p_networkArgs.getLossFunctionType())
{
    m_rng = std::mt19937(static_cast<unsigned long>(p_seed));
    m_oclResources = std::make_unique<Utils::OpenCLResources>(std::move(p_oclResources));
    Utils::Dimensions currentInputDimensions = m_inputDimensions;
    for (const auto& layerArgs : p_networkArgs.getLayersArguments()) {
        
        m_layers.emplace_back(layerArgs->createLayer(m_layers.size(), m_oclResources->getSharedResources(), currentInputDimensions, m_batchSize, m_rng));
        currentInputDimensions = m_layers.back()->getOutputDimensions();
    }
    m_optimizer = p_networkArgs.getOptimizerArguments()->createOptimizer(m_oclResources->getSharedResources());
    setupKernels();
}

std::vector<float> NeuralNetwork::predict(const cl::Buffer& p_inputBatch,
                                          size_t p_batchSize) {
    
    size_t originalBatchSize = m_batchSize;

    m_batchSize = p_batchSize;
    cl::Event forwardEvent = forward(p_inputBatch, m_batchSize);
    m_batchSize = originalBatchSize;

    cl::Buffer prediction = m_layers.back()->getOutputs();
    size_t predictionSize = p_batchSize * m_layers.back()->getTotalOutputElements();
    std::vector<float> predictionVec(predictionSize);
    
    std::vector<cl::Event> waitList = {forwardEvent};
    m_oclResources->getForwardBackpropQueue().enqueueReadBuffer(prediction, BLOCKING_READ, NO_OFFSET,
                            sizeof(float) * predictionSize, predictionVec.data(), &waitList);
    return predictionVec;
}

double NeuralNetwork::trainStepLoss(const Batch& p_batch, 
                                    bool p_lossReporting) {
    cl::Buffer inputs = p_batch.getInputs();
    cl::Buffer targets = p_batch.getTargets();
    size_t batchSize = p_batch.getSize();
    cl::Event forwardEvent = forward(inputs, batchSize);
    double loss = -1.0;
    std::future<double> lossFuture;
    if (p_lossReporting == true){
        lossFuture = std::async(std::launch::async, &NeuralNetwork::computeLossAsync, this, std::ref(forwardEvent), p_batch.getTargetsVector());
    }
    computeLossGradients(targets);
    backward(inputs);
    
    if (p_lossReporting == true){
        lossFuture.wait();
        loss = lossFuture.get();
    }

    return loss;
}

void NeuralNetwork::trainStepPolicy(const PolicyBatch& p_batch) {
    cl::Buffer inputs = p_batch.getInputs();
    cl::Buffer actions = p_batch.getActions();
    cl::Buffer rewards = p_batch.getRewards();
    size_t batchSize = p_batch.getSize();
    cl::Event forwardEvent = forward(inputs, batchSize);
    computePolicyGradients(actions, rewards);
    backward(inputs);
};

void NeuralNetwork::train(DataLoader& p_dataLoader, int p_epochs, bool p_lossReporting) {
    p_dataLoader.activateTrainPartition();
    double totalLoss;
    for (int epoch = 0; epoch < p_epochs; ++epoch) {
        p_dataLoader.shuffleCurrentPartition();
        for (const Batch& batch: p_dataLoader){
            totalLoss = trainStepLoss(batch, p_lossReporting);
        }
        if (p_lossReporting == true){
            std::cout << "Epoch " << (epoch + 1) << " | Loss: " << (totalLoss / (p_dataLoader.getActivePartition().size() * m_layers.back()->getTotalOutputElements())) << "\n";
        }   
    }   
}

NeuralNetwork& NeuralNetwork::addDense(const size_t p_numOutputNeurons, Utils::ActivationType p_activationType) {
    Utils::Dimensions outputDimensions = Utils::Dimensions::validateDenseDimensions({p_numOutputNeurons});
    Utils::Dimensions inputDimensions;
    if (m_layers.empty()){
        inputDimensions = m_inputDimensions;
    }
    else{
        inputDimensions = m_layers.back()->getOutputDimensions();
    }
    auto layerArgs = Utils::makeDenseLayerArgs(outputDimensions, p_activationType);
    m_layers.emplace_back(layerArgs->createLayer(m_layers.size(), m_oclResources->getSharedResources(), inputDimensions, m_batchSize, m_rng));
    setupKernels();
    return *this;
}

NeuralNetwork& NeuralNetwork::addConvolutional(const Utils::FilterDimensions& p_filterDimensions, const Utils::StrideDimensions& p_strideDimensions, const Utils::PaddingType p_paddingType, const Utils::ActivationType p_activationType) {
    Utils::Dimensions inputDimensions;
    if (m_layers.empty()){
        inputDimensions = m_inputDimensions;
    }
    else{
        inputDimensions = m_layers.back()->getOutputDimensions();
    }
    auto layerArgs = Utils::makeConvolutionalLayerArgs(p_filterDimensions, p_strideDimensions, p_paddingType, p_activationType);
    m_layers.emplace_back(layerArgs->createLayer(m_layers.size(), m_oclResources->getSharedResources(), inputDimensions, m_batchSize, m_rng));
    setupKernels();
    return *this;
}

void NeuralNetwork::saveNetwork(const std::string& p_fileName) const {
    try {
        H5::H5File file(p_fileName, H5F_ACC_TRUNC);
        H5::DataSpace scalarDataspace(H5S_SCALAR);

        std::ostringstream oss;
        oss << m_rng;
        std::string state = oss.str();

        hsize_t dims[1] = { state.size() };
        H5::DataSpace dataspace(1, dims);
        H5::DataSet dataset = file.createDataSet(
            "rngState", H5::PredType::NATIVE_CHAR, dataspace
        );
        dataset.write(state.data(), H5::PredType::NATIVE_CHAR);


        std::vector<hsize_t> inputDimensionsVec = m_inputDimensions.getDimensions();
        hsize_t numDimensions = inputDimensionsVec.size();

        H5::DataSpace inputDimensionsSpace(1, &numDimensions);

        H5::DataSet inputDataset = file.createDataSet(
            "inputDimensions", 
            H5::PredType::NATIVE_HSIZE, 
            inputDimensionsSpace
        );

        inputDataset.write(inputDimensionsVec.data(), H5::PredType::NATIVE_HSIZE);

        file.createAttribute(
            "batchSize", H5::PredType::NATIVE_HSIZE, scalarDataspace
        ).write(H5::PredType::NATIVE_HSIZE, &m_batchSize);

        unsigned int lossTypeUInt = static_cast<unsigned int>(m_lossFunctionType);
        file.createAttribute(
            "lossFunctionType", H5::PredType::NATIVE_UINT, scalarDataspace
        ).write(H5::PredType::NATIVE_UINT, &lossTypeUInt);

        H5::Group layersGroup(file.createGroup("/layers"));
        size_t numLayers = m_layers.size();
        layersGroup.createAttribute(
            "numLayers", H5::PredType::NATIVE_HSIZE, scalarDataspace
        ).write(H5::PredType::NATIVE_HSIZE, &numLayers);

        std::map<size_t, std::pair<size_t, size_t>> momentsSizes;
        size_t layerId;
        for (size_t i = 0; i < m_layers.size(); ++i) {
            layerId = m_layers[i]->getLayerId();
            if (m_layers[i]->isTrainable()) {
                auto& trainableLayer = static_cast<TrainableLayer&>(*m_layers[i]);
                momentsSizes[layerId] = {
                    trainableLayer.getWeightsSize(),
                    trainableLayer.getBiasesSize()
                };
            }
            H5::Group layerSubGroup(layersGroup.createGroup(std::to_string(layerId)));
            m_layers[i]->setBatchSize(m_batchSize);
            m_layers[i]->saveLayer(m_oclResources->getForwardBackpropQueue(), layerSubGroup);
        }

        if (m_optimizer) {
            H5::Group optimizerGroup(file.createGroup("/optimizer"));
            m_optimizer->saveOptimizer(m_oclResources->getForwardBackpropQueue(), optimizerGroup, momentsSizes);
        } else {
            std::cerr << "Warning: Optimizer is null, skipping its save operation." << std::endl;
        }

        file.close();
        std::cout << "Neural Network successfully saved to " << p_fileName << std::endl;
    } catch (const H5::Exception& error) {
        std::cerr << "HDF5 Exception in NeuralNetwork::saveNetwork: " << error.getFuncName()
                  << " -> " << error.getDetailMsg() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception in NeuralNetwork::saveNetwork: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown Exception in NeuralNetwork::saveNetwork." << std::endl;
    }
}

NeuralNetwork NeuralNetwork::loadNetwork(std::shared_ptr<Utils::SharedResources> p_sharedResources, const std::string& p_fileName) {
    if (!std::filesystem::exists(p_fileName)) {
        std::cerr << "Error: File does not exist: " << p_fileName << std::endl;
        throw std::runtime_error("File does not exist: " + p_fileName);
    }
    Utils::OpenCLResources oclResources = Utils::OpenCLResources::createOpenCLResources(p_sharedResources);
    H5::H5File file(p_fileName, H5F_ACC_RDONLY);
    NeuralNetwork network(std::move(oclResources), file);
    file.close();
    return network;
}

bool NeuralNetwork::equals(const NeuralNetwork& p_other) const {
    if (m_batchSize != p_other.m_batchSize ||
        m_inputDimensions != p_other.m_inputDimensions ||
        m_lossFunctionType != p_other.m_lossFunctionType ||
        m_layers.size() != p_other.m_layers.size()) {
        return false;
    }

    for (size_t i = 0; i < m_layers.size(); ++i) {
        if (!m_layers[i]->equals(m_oclResources->getForwardBackpropQueue(), *p_other.m_layers[i])) {
            std::cout << "Mismatch in layer " << i << ".\n";
            return false;
        }
    }

    if ((m_optimizer == nullptr) != (p_other.m_optimizer == nullptr)) {
        std::cout << "One network has an optimizer while the other does not.\n";
        return false;
    }
    if (m_optimizer && p_other.m_optimizer) {
        std::map<size_t, std::pair<size_t, size_t>> momentsSizes;
        for (const auto& layer : m_layers) {
            if (layer->isTrainable()) {
                auto& trainableLayer = static_cast<TrainableLayer&>(*layer);
                momentsSizes[layer->getLayerId()] = {
                    trainableLayer.getWeightsSize(),
                    trainableLayer.getBiasesSize()
                };
            }
        }
        if (!m_optimizer->equals(m_oclResources->getForwardBackpropQueue(), *p_other.m_optimizer, momentsSizes)) {
            std::cout << "Mismatch in optimizers.\n";
            return false;
        }
    }

    return true;
}

NeuralNetwork::NeuralNetwork(Utils::OpenCLResources&& p_oclResources, 
                             const H5::H5File& p_file) {
    m_oclResources = std::make_unique<Utils::OpenCLResources>(std::move(p_oclResources));

    try {
        H5::DataSet dataset = p_file.openDataSet("rngState");
        H5::DataSpace dataspace = dataset.getSpace();

        hsize_t dims[1];
        dataspace.getSimpleExtentDims(dims, nullptr);

        std::string state(dims[0], '\0');
        dataset.read(state.data(), H5::PredType::NATIVE_CHAR);

        std::istringstream iss(state);
        iss >> m_rng;

        H5::DataSet inputDataset = p_file.openDataSet("inputDimensions");
        H5::DataSpace inputSpace = inputDataset.getSpace();

        hsize_t inputDims[1];
        inputSpace.getSimpleExtentDims(inputDims);

        std::vector<size_t> inputDimensionsVec(inputDims[0]);
        inputDataset.read(inputDimensionsVec.data(), H5::PredType::NATIVE_HSIZE);

        m_inputDimensions = Utils::Dimensions(inputDimensionsVec);
    } catch (const H5::Exception& e) {
        std::cerr << "HDF5 error: " << e.getCDetailMsg() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Std error: " << e.what() << std::endl;
    }

    if (p_file.attrExists("batchSize")) p_file.openAttribute("batchSize").read(H5::PredType::NATIVE_HSIZE, &m_batchSize);
    else m_batchSize = 1;

    if (p_file.attrExists("lossFunctionType")){
        unsigned int lossFunctionUInt;
        p_file.openAttribute("lossFunctionType").read(H5::PredType::NATIVE_UINT, &lossFunctionUInt);
        m_lossFunctionType = Utils::lossFunctionTypeFromUint(lossFunctionUInt);

    }
    else m_lossFunctionType = Utils::LossFunctionType::MeanSquaredError;
    
    H5::Group layersGroup = p_file.openGroup("layers");
    size_t numLayers;
    if (layersGroup.attrExists("numLayers")) layersGroup.openAttribute("numLayers").read(H5::PredType::NATIVE_HSIZE, &numLayers);
    else numLayers = 0;

    for (size_t i = 0; i < numLayers; ++i) {
        std::string layerId = std::to_string(i);
        H5::Group layerGroup = layersGroup.openGroup(layerId);
        m_layers.emplace_back(Utils::loadLayer(m_oclResources->getSharedResources(), layerGroup, m_batchSize));
    }
    if (m_layers.front()->getInputDimensions() != m_inputDimensions) {
        throw std::runtime_error("Input dimensions of the first layer do not match the network's input dimensions.");
    }
    H5::Group optimizerGroup = p_file.openGroup("optimizer");
    m_optimizer = Utils::loadOptimizer(m_oclResources->getSharedResources(), optimizerGroup);
    setupKernels();
}

double NeuralNetwork::computeLossAsync(cl::Event& p_forwardEvent, const std::vector<float>& p_batchTargets) {
    size_t batchSize = m_layers.back()->getBatchSize();
    size_t flatOutputSize = m_layers.back()->getTotalOutputElements();
    size_t totalBatchElements = batchSize * flatOutputSize;

    double totalLoss = 0.0;
    std::vector<float> predictions(totalBatchElements);

    std::vector<cl::Event> waitList = {p_forwardEvent};

    m_oclResources->getConcurrentQueue().enqueueReadBuffer(m_layers.back()->getOutputs(), BLOCKING_READ, NO_OFFSET, sizeof(float) * totalBatchElements, predictions.data(),
                            &waitList);
                            
    for (size_t i = 0; i < totalBatchElements; ++i) {
        totalLoss += applyLossFunction(m_lossFunctionType, predictions[i], p_batchTargets[i]);
    }

    return totalLoss;
}

cl::Event NeuralNetwork::forward(const cl::Buffer& p_batchInputs, size_t p_batchSize) {
    cl::Buffer currentInput = p_batchInputs;
    cl::Event lastEvent;

    for (auto& layer : m_layers) {
        layer->setBatchSize(p_batchSize);
        lastEvent = layer->runForward(m_oclResources->getForwardBackpropQueue(), currentInput);
        currentInput = layer->getOutputs();
    }
    return lastEvent;
}

void NeuralNetwork::computeLossGradients(const cl::Buffer& p_batchTargets) {
    m_lossGradientKernel.setArg(4, p_batchTargets);
    m_oclResources->getForwardBackpropQueue().enqueueNDRangeKernel(
        m_lossGradientKernel,
        cl::NullRange,
        cl::NDRange(m_layers.back()->getTotalOutputElements(), m_layers.back()->getBatchSize()),
        cl::NullRange
    );
}

void NeuralNetwork::computePolicyGradients(const cl::Buffer& p_batchActions, const cl::Buffer& p_batchRewards) {
    m_policyGradientKernel.setArg(3, p_batchActions);
    m_policyGradientKernel.setArg(4, p_batchRewards);
    m_oclResources->getForwardBackpropQueue().enqueueNDRangeKernel(
        m_policyGradientKernel,
        cl::NullRange,
        cl::NDRange(m_layers.back()->getTotalOutputElements(), m_layers.back()->getBatchSize()),
        cl::NullRange
    );
}

void NeuralNetwork::backward(const cl::Buffer& p_batchInputs) {
    std::pair<cl::Event, cl::Event> gradientEvents;
    cl::Event deltaEvent;
    for (int l = static_cast<int>(m_layers.size()) - 1; l >= 0; --l) {
        auto& currentLayer = m_layers[l];
        deltaEvent = currentLayer->computeDeltas(m_oclResources->getForwardBackpropQueue());
        if (l > 0){
            auto& previousLayer = m_layers[l - 1];
            currentLayer->backpropDeltas(m_oclResources->getForwardBackpropQueue(), previousLayer->getDeltas(), previousLayer->getOutputDimensions());
        }

        if (currentLayer->isTrainable()) {
            const cl::Buffer& currentLayerInputs = (l == 0) ? p_batchInputs : m_layers[l - 1]->getOutputs();
            auto& trainableLayer = static_cast<TrainableLayer&>(*currentLayer);
            gradientEvents = trainableLayer.computeGradients(m_oclResources->getDeltaToGradientQueue(), m_oclResources->getConcurrentQueue(), deltaEvent, currentLayerInputs);
            size_t layerId = trainableLayer.getLayerId();
            m_optimizer->updateParameters(m_oclResources->getConcurrentQueue(), gradientEvents.first, std::to_string(layerId) + "Weights", trainableLayer.getWeights(), trainableLayer.getWeightsGradients(), trainableLayer.getWeightsSize());
            m_optimizer->updateParameters(m_oclResources->getConcurrentQueue(), gradientEvents.second, std::to_string(layerId) + "Biases", trainableLayer.getBiases(), trainableLayer.getBiasesGradients(), trainableLayer.getBiasesSize());
        }
    }
    m_oclResources->getConcurrentQueue().finish();
    m_optimizer->step();
}

void NeuralNetwork::setupKernels() {
    if (m_layers.empty()) {
        return;
    }
    cl_int err;
    m_lossGradientKernel = cl::Kernel(m_oclResources->getSharedResources()->getProgram(), "computeLossGradient", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create loss gradient kernel. Error code: " + std::to_string(err));
    }
    m_lossGradientKernel.setArg(0, m_layers.back()->getOutputs());
    m_lossGradientKernel.setArg(1, m_layers.back()->getDeltas());
    m_lossGradientKernel.setArg(3, (cl_uint)m_layers.back()->getTotalOutputElements());

    m_lossGradientKernel.setArg(2, (cl_uint)m_lossFunctionType);

    m_policyGradientKernel = cl::Kernel(m_oclResources->getSharedResources()->getProgram(), "computePolicyGradient", &err);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to create policy gradient kernel. Error code: " + std::to_string(err));
    }
    m_policyGradientKernel.setArg(0, m_layers.back()->getOutputs());
    m_policyGradientKernel.setArg(1, m_layers.back()->getDeltas());
    m_policyGradientKernel.setArg(2, (cl_uint)m_layers.back()->getTotalOutputElements());
}