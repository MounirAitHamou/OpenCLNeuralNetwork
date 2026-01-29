#include "Layers/TrainableLayers/Dense/DenseLayer.hpp"
namespace Layers::Trainable {
    DenseLayer::DenseLayer(const size_t p_layerId, 
                        std::shared_ptr<Utils::SharedResources> p_sharedResources,
                        const Utils::Dimensions& p_inputDimensions, 
                        const Utils::Dimensions& p_outputDimensions,
                        const size_t p_batchSize,
                        std::mt19937& p_rng)
            :
            TrainableLayer(p_layerId, p_sharedResources, p_inputDimensions, Utils::Dimensions::validateDenseDimensions(p_outputDimensions), p_batchSize) {
        try {
            initializeWeightsAndBiases(p_rng);
            allocateDenseLayerBuffers(p_batchSize);
            setupKernels();
        }
        catch (const std::exception& e) {
            std::cerr << "Error constructing DenseLayer: " << e.what() << std::endl;
            throw; 
        }
    }

    DenseLayer::DenseLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                        const H5::Group& p_layerGroup, 
                        const size_t p_batchSize)
    : TrainableLayer(p_sharedResources, p_layerGroup, p_batchSize) {
        m_weights = Utils::loadBuffer(p_sharedResources->getContext(), p_layerGroup, "weights", getWeightsSize());
        m_biases  = Utils::loadBuffer(p_sharedResources->getContext(), p_layerGroup, "biases", getBiasesSize());
        allocateDenseLayerBuffers(p_batchSize);
        setupKernels();
    }

    cl::Event DenseLayer::runForward(const cl::CommandQueue& p_forwardBackpropQueue,
                                    const cl::Buffer& p_inputs,
                                    const size_t p_batchSize) {
        if (m_batchSize < p_batchSize) setBatchSize(p_batchSize);

        size_t flatInputSize = m_inputDimensions.getTotalElements();
        size_t flatOutputSize = m_outputDimensions.getTotalElements();
        
        cl_command_queue raw_queue = p_forwardBackpropQueue.get();

        auto status = clblast::Gemm<float>(
            clblast::Layout::kRowMajor,
            clblast::Transpose::kNo, clblast::Transpose::kYes,
            p_batchSize, flatOutputSize, flatInputSize,
            NO_SCALAR,
            p_inputs(), NO_OFFSET, flatInputSize,
            getWeights()(), NO_OFFSET, flatInputSize,
            CLEAR_C,
            getOutputs()(), NO_OFFSET, flatOutputSize,
            &raw_queue, nullptr,
            m_clblastWorkspace()
        );
        if (status != clblast::StatusCode::kSuccess) {
            std::cerr << "Forward CLBlast GEMM failed: " << static_cast<int>(status) << " for layer " << m_layerId << std::endl;
            throw std::runtime_error("CLBlast GEMM failed");
        }

        cl::Event returnEvent;
        
        cl_int err = p_forwardBackpropQueue.enqueueNDRangeKernel(m_biasKernel, cl::NullRange,
                                cl::NDRange(flatOutputSize, p_batchSize), cl::NullRange, 
                                nullptr, &returnEvent);
        
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue bias addition kernel.");
        }

        return returnEvent;
    }

    cl::Event DenseLayer::backpropDeltas(
        const cl::CommandQueue& p_forwardBackpropQueue, 
        const cl::Buffer& p_previousLayerDeltas,
        const size_t p_batchSize) {
        if (m_batchSize < p_batchSize) setBatchSize(p_batchSize);

        size_t previousLayerFlatOutputSize = m_inputDimensions.getTotalElements();
        size_t flatOutputSize = m_outputDimensions.getTotalElements();

        cl_command_queue raw_queue = p_forwardBackpropQueue.get();
        cl_event raw_event = nullptr;

        auto status = clblast::Gemm<float>(
            clblast::Layout::kRowMajor,
            clblast::Transpose::kNo, clblast::Transpose::kNo,
            p_batchSize, previousLayerFlatOutputSize, flatOutputSize,
            NO_SCALAR,
            getDeltas()(), NO_OFFSET, flatOutputSize,
            getWeights()(), NO_OFFSET, previousLayerFlatOutputSize,
            CLEAR_C,
            p_previousLayerDeltas.get(), NO_OFFSET, previousLayerFlatOutputSize,
            &raw_queue, &raw_event,
            m_clblastDeltaWorkspace()
        );

        if (status != clblast::StatusCode::kSuccess) {
            std::cerr << "Backprop CLBlast GEMM failed: " << static_cast<int>(status) << " for layer " << m_layerId << std::endl;
            throw std::runtime_error("CLBlast GEMM failed");
        }

        return cl::Event(raw_event, true);
    }

    std::pair<cl::Event, cl::Event> DenseLayer::computeGradients(const cl::CommandQueue& p_deltaToGradientQueue,
                                                                cl::Event& p_backpropEvent, 
                                                                const cl::Buffer& p_inputs,
                                                                const size_t p_batchSize) {
        if (m_batchSize < p_batchSize) setBatchSize(p_batchSize);
        if (p_backpropEvent() != nullptr) {
            std::vector<cl::Event> deltaBackPropWaitList = { p_backpropEvent };
            p_deltaToGradientQueue.enqueueBarrierWithWaitList(&deltaBackPropWaitList);
        }

        size_t flatInputSize = m_inputDimensions.getTotalElements();
        size_t flatOutputSize = m_outputDimensions.getTotalElements();
        
        cl_event raw_gemm_event = nullptr;
        cl_command_queue raw_queue = p_deltaToGradientQueue.get();
        float alpha = 1.0f / static_cast<float>(p_batchSize);
        auto status = clblast::Gemm<float>(
            clblast::Layout::kRowMajor,
            clblast::Transpose::kYes,
            clblast::Transpose::kNo,
            flatOutputSize, flatInputSize, p_batchSize,
            alpha,
            getDeltas()(), NO_OFFSET, flatOutputSize,
            p_inputs(), NO_OFFSET, flatInputSize,
            CLEAR_C,
            getWeightsGradients()(), NO_OFFSET, flatInputSize,
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
            p_batchSize, flatOutputSize,
            alpha,
            getDeltas()(), NO_OFFSET, flatOutputSize,
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
        return { gemmEvent, gemvEvent };
    }

    void DenseLayer::allocateDenseLayerBuffers(const size_t p_batchSize) {
        m_weightsGradients = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE, getWeightsSize() * sizeof(float));
        m_biasesGradients  = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE, getBiasesSize() * sizeof(float));
        m_onesBuffer       = cl::Buffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, p_batchSize * sizeof(float), std::vector<float>(p_batchSize, 1.0f).data());
        
        size_t flatInputSize = m_inputDimensions.getTotalElements();
        size_t flatOutputSize = m_outputDimensions.getTotalElements();

        m_clblastWorkspace = cl::Buffer(
            m_sharedResources->getContext(),
            CL_MEM_READ_WRITE,
            std::max({p_batchSize * flatOutputSize, flatOutputSize * flatInputSize}) * sizeof(float)
        );

        m_clblastDeltaWorkspace = cl::Buffer(
            m_sharedResources->getContext(),
            CL_MEM_READ_WRITE,
            std::max({p_batchSize * flatInputSize, flatInputSize * flatOutputSize}) * sizeof(float)
        );
    }

    void DenseLayer::initializeWeightsAndBiases(std::mt19937& p_rng) {
        std::vector<float> h_weights(getWeightsSize());
        std::vector<float> h_biases(getBiasesSize());

        float fanIn = (float)getTotalInputElements();
        float fanOut = (float)getTotalOutputElements();
        float limit = std::sqrt(6.0f / (fanIn + fanOut));

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
        setupTrainableKernels();
        cl_int err;

        m_biasKernel = cl::Kernel(m_sharedResources->getProgram(), "denseBias", &err);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to create denseBias kernel");
        }
        m_biasKernel.setArg(0, getBiases());
        m_biasKernel.setArg(1, getOutputs());
        m_biasKernel.setArg(2, (cl_uint)getTotalOutputElements());
    }
}