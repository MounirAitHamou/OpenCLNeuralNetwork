#pragma once

#include "Layers/ActivationLayers/PreActivationLayers/PreActivationLayer.hpp"
namespace Layers::Activation {
    class LeakyReLULayer : public PreActivationLayer {
    public:
        LeakyReLULayer(const size_t p_layerId, 
                    std::shared_ptr<Utils::SharedResources> p_sharedResources,
                    const Utils::Dimensions& p_outputDimensions,
                    float p_alpha,
                    const size_t p_batchSize)
            : PreActivationLayer(p_layerId, p_sharedResources, p_outputDimensions, p_batchSize),
            m_alpha(p_alpha)
        {
            setupKernels();
        }

        LeakyReLULayer(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                    const H5::Group& p_layerGroup,
                    const size_t p_batchSize)
            : PreActivationLayer(p_sharedResources, p_layerGroup, p_batchSize)
        {
            m_alpha = Utils::readValueFromHDF5<float>(p_layerGroup, "alpha");
            setupKernels();
        }

        ~LeakyReLULayer() = default;

        Utils::LayerType getType() const override { return Utils::LayerType::LeakyReLU; }

        float getAlpha() const { return m_alpha; }

        void save(const cl::CommandQueue& p_queue, H5::Group& p_layerGroup) const final override { saveLeakyReLULayer(p_queue, p_layerGroup); }
        bool equals(const cl::CommandQueue& p_queue, const Layer& p_other) const final override { return leakyReLULayerEquals(p_queue, p_other); }
        void print(const cl::CommandQueue& p_queue, const size_t p_batchSize) const final override { printLeakyReLULayer(p_queue, p_batchSize); }

    private:
        void setupKernels() final override;

        float m_alpha;

        void saveLeakyReLULayer(const cl::CommandQueue& p_queue, H5::Group& p_layerGroup) const { 
            saveLayer(p_layerGroup);
            Utils::writeValueToHDF5<float>(p_layerGroup, "alpha", m_alpha);
        }

        bool leakyReLULayerEquals(const cl::CommandQueue& p_queue, const Layer& p_other) const {
            if (!layerEquals(p_queue, p_other)) return false;
            const LeakyReLULayer& otherLeakyReLU = static_cast<const LeakyReLULayer&>(p_other);
            return m_alpha == otherLeakyReLU.m_alpha;
        }

        void printLeakyReLULayer(const cl::CommandQueue& p_queue, const size_t p_batchSize) const { 
            printPreActivationLayer(p_queue, p_batchSize);
            std::cout << "Alpha: " << m_alpha << std::endl;
        }
    };
}