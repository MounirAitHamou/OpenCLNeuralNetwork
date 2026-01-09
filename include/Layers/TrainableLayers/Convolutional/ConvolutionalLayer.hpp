#pragma once

#include "Layers/TrainableLayers/TrainableLayer.hpp"
#include "Utils/FilterDimensions.hpp"
#include "Utils/StrideDimensions.hpp"
#include "Utils/PaddingValues.hpp"
#include "Utils/PaddingType.hpp"
namespace Layers::Trainable {
    class ConvolutionalLayer : public TrainableLayer {
    public:

        ConvolutionalLayer(const size_t p_layerId, 
                        std::shared_ptr<Utils::SharedResources> p_sharedResources,
                        const Utils::Dimensions& p_inputDimensions,
                        const Utils::FilterDimensions& p_filterDimensions,
                        const Utils::StrideDimensions& p_strideDimensions,
                        const Utils::PaddingType p_paddingType,
                        const size_t p_batchSize,
                        std::mt19937& p_rng);
                        
        ConvolutionalLayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                        const H5::Group& p_layerGroup, 
                        const size_t p_batchSize);

        ~ConvolutionalLayer() = default;

        cl::Event runForward(const cl::CommandQueue& p_forwardBackpropQueue, const cl::Buffer& p_inputs, const size_t p_batchSize) final override;
        cl::Event backpropDeltas(const cl::CommandQueue& p_forwardBackpropQueue, const cl::Buffer& p_previousLayerDeltas, const size_t p_batchSize) final override;
        std::pair<cl::Event, cl::Event> computeGradients(const cl::CommandQueue& p_deltaToGradientQueue, cl::Event& p_backpropEvent, const cl::Buffer& p_inputs, const size_t p_batchSize) final override;

        Utils::LayerType getType() const final override { return Utils::LayerType::Convolutional; }

        size_t getWeightsSize() const final override { return getOutputChannels() * getInputChannels() * m_filterDimensions.getHeight() * m_filterDimensions.getWidth(); }
        size_t getBiasesSize() const final override { return getOutputChannels(); }
        const std::vector<float> getSerializedArgs() const final override {
            std::vector<float> layerArgs = getLayerSerializedArgs();
            layerArgs.push_back(static_cast<float>(m_filterDimensions.getHeight()));
            layerArgs.push_back(static_cast<float>(m_filterDimensions.getWidth()));
            layerArgs.push_back(static_cast<float>(m_filterDimensions.getInputChannels()));
            layerArgs.push_back(static_cast<float>(m_filterDimensions.getOutputChannels()));
            layerArgs.push_back(static_cast<float>(m_strideDimensions.getHeight()));
            layerArgs.push_back(static_cast<float>(m_strideDimensions.getWidth()));
            layerArgs.push_back(static_cast<float>(m_paddingType));
            return layerArgs;
        }

        void save(const cl::CommandQueue& p_queue, H5::Group& p_layerGroup) const final override { saveConvolutionalLayer(p_queue, p_layerGroup); }
        bool equals(const cl::CommandQueue& p_queue, const Layer& p_other) const final override { return convolutionalLayerEquals(p_queue, p_other); }
        void print(const cl::CommandQueue& p_queue, const size_t p_batchSize) const final override { printConvolutionalLayer(p_queue, p_batchSize); }


        void setBatchSize(const size_t p_batchSize) final override {
            allocateLayerBuffers(p_batchSize);
            size_t im2colRows = getInputChannels() * m_filterDimensions.getHeight() * m_filterDimensions.getWidth();
            size_t im2colCols = getOutputHeight() * getOutputWidth();
            size_t im2colBatchCols = im2colCols * p_batchSize;
            m_biasKernel.setArg(1, getOutputs());


            m_im2colBuffer = cl::Buffer(
                m_sharedResources->getContext(), 
                CL_MEM_READ_WRITE, 
                im2colRows * im2colBatchCols * sizeof(float)
            );

            m_im2colKernel.setArg(0, m_im2colBuffer);

            m_col2imBuffer = cl::Buffer(
                m_sharedResources->getContext(), 
                CL_MEM_READ_WRITE, 
                im2colRows * im2colBatchCols * sizeof(float)
            );
            m_col2imKernel.setArg(0, m_col2imBuffer);

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
    private:
        cl::Buffer m_im2colBuffer;
        cl::Buffer m_col2imBuffer;

        cl::Kernel m_im2colKernel;
        cl::Kernel m_col2imKernel;

        Utils::FilterDimensions m_filterDimensions;
        Utils::StrideDimensions m_strideDimensions;
        Utils::PaddingValues m_paddingValues;
        Utils::PaddingType m_paddingType;
        
        size_t getInputChannels() const { return m_inputDimensions.getDimensions()[0]; }
        size_t getInputHeight() const { return m_inputDimensions.getDimensions()[1]; }
        size_t getInputWidth() const { return m_inputDimensions.getDimensions()[2]; }

        size_t getOutputChannels() const { return m_outputDimensions.getDimensions()[0]; }
        size_t getOutputHeight() const { return m_outputDimensions.getDimensions()[1]; }
        size_t getOutputWidth() const { return m_outputDimensions.getDimensions()[2]; }

        void allocateConvolutionalLayerBuffers(const size_t p_batchSize);
        Utils::Dimensions calculateOutputDimensions(const Utils::Dimensions& p_inputDimensions, const Utils::FilterDimensions& p_filterDimensions, const Utils::StrideDimensions& p_strideDimensions, Utils::PaddingType p_paddingType) const;
        Utils::Dimensions calculateOutputDimensions() const;
        Utils::PaddingValues calculatePaddingValues(const Utils::Dimensions& p_inputDimensions, const Utils::FilterDimensions& p_filterDimensions, const Utils::StrideDimensions& p_strideDimensions, const Utils::PaddingType p_paddingType) const;
        void initializeWeightsAndBiases(std::mt19937& p_rng) final override;
        void setupKernels() final override;
        Utils::Dimensions validateInputDimensions(const Utils::Dimensions& p_inputDimensions, const Utils::FilterDimensions& p_filterDimensions, const Utils::StrideDimensions& p_strideDimensions) const;

        void saveConvolutionalLayer(const cl::CommandQueue& p_queue, H5::Group& p_layerGroup) const {
            saveTrainableLayer(p_queue, p_layerGroup);
            Utils::writeVectorToHDF5<size_t>(p_layerGroup, "filterDimensions", m_filterDimensions.getDimensions());
            Utils::writeVectorToHDF5<size_t>(p_layerGroup, "strideDimensions", m_strideDimensions.getDimensions());
            Utils::writeVectorToHDF5<size_t>(p_layerGroup, "paddingValues", m_paddingValues.getDimensions());
            Utils::writeValueToHDF5<unsigned int>(p_layerGroup, "paddingType", static_cast<unsigned int>(m_paddingType));
        }

        bool convolutionalLayerEquals(const cl::CommandQueue& p_queue, const Layer& p_other) const {
            if (!trainableLayerEquals(p_queue, p_other)) return false;
        
            const ConvolutionalLayer& otherConv = static_cast<const ConvolutionalLayer&>(p_other);

            return m_filterDimensions == otherConv.m_filterDimensions && 
                m_strideDimensions == otherConv.m_strideDimensions &&
                m_paddingValues == otherConv.m_paddingValues;
        }

        void printConvolutionalLayer(const cl::CommandQueue& p_queue, const size_t p_batchSize) const {
            size_t im2colSize = p_batchSize * getInputChannels() * m_filterDimensions.getHeight() * m_filterDimensions.getWidth() * getOutputHeight() * getOutputWidth();
            printTrainableLayer(p_queue, p_batchSize);
            Utils::printCLBuffer(p_queue, m_im2colBuffer, im2colSize, "Im2Col Buffer");
            Utils::printCLBuffer(p_queue, m_col2imBuffer, im2colSize, "Col2Im Buffer");
            std::cout << "Filter Dimensions: " << m_filterDimensions.toString() << "\n";
            std::cout << "Stride Dimensions: " << m_strideDimensions.toString() << "\n";
            std::cout << "Padding Values (Top, Bottom, Left, Right): (" 
                    << m_paddingValues.getTop() << ", "
                    << m_paddingValues.getBottom() << ", "
                    << m_paddingValues.getLeft() << ", "
                    << m_paddingValues.getRight() << ")\n";
        }
    };
}