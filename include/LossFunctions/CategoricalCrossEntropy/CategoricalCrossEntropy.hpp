#pragma once 

#include "LossFunctions/LossFunction.hpp"

namespace LossFunctions {
    class CategoricalCrossEntropy : public LossFunction {
    public:
        CategoricalCrossEntropy(std::shared_ptr<Utils::SharedResources> p_sharedResources)
            : LossFunction(p_sharedResources) {setupKernel();}
        
        Utils::LossFunctionType getType() const override {
            return Utils::LossFunctionType::CategoricalCrossEntropy;
        }

        cl::Event computeLossGradient(const cl::CommandQueue& p_queue,
                                      const cl::Buffer& p_predictions, 
                                      const cl::Buffer& p_targets, 
                                      cl::Buffer& p_outputGradients, 
                                      const size_t p_outputElements, 
                                      const size_t p_batchSize) override;
    private:
        void setupKernel() override;
        float computeLoss(const std::vector<float>& p_predictions, 
                          const std::vector<float>& p_targets, 
                          size_t p_outputElements, 
                          size_t p_batchSize) override;
    };
}