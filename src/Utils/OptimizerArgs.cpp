#include "Utils/OptimizerArgs.hpp"

namespace Utils {
    std::unique_ptr<OptimizerArgs> makeSGDArgs(float p_learningRate, float p_weightDecayRate) {
        return std::make_unique<SGDOptimizerArgs>(p_learningRate, p_weightDecayRate);
    }

    std::unique_ptr<OptimizerArgs> makeAdamArgs(float p_learningRate, float p_weightDecayRate,
                                               float p_beta1, float p_beta2, float p_epsilon) {
        return std::make_unique<AdamOptimizerArgs>(p_learningRate, p_weightDecayRate, p_beta1, p_beta2, p_epsilon);
    }

     std::unique_ptr<OptimizerArgs> makeAdamWArgs(float p_learningRate, float p_weightDecayRate,
                                               float p_beta1, float p_beta2, float p_epsilon) {
        return std::make_unique<AdamWOptimizerArgs>(p_learningRate, p_weightDecayRate, p_beta1, p_beta2, p_epsilon);
    }

    std::unique_ptr<Optimizers::Optimizer> loadOptimizer(std::shared_ptr<Utils::SharedResources> p_oclResources,
                                             const H5::Group& p_optimizerGroup) {
        unsigned int optimizerType;
        p_optimizerGroup.openAttribute("optimizerType").read(H5::PredType::NATIVE_UINT, &optimizerType);

        switch (optimizerTypeFromUint(optimizerType)) {
            case OptimizerType::SGD:
                return std::make_unique<Optimizers::SGDOptimizer>(p_oclResources, p_optimizerGroup);
            case OptimizerType::Adam:
                return std::make_unique<Optimizers::AdamOptimizer>(p_oclResources, p_optimizerGroup);
            case OptimizerType::AdamW:
                return std::make_unique<Optimizers::AdamWOptimizer>(p_oclResources, p_optimizerGroup);
            default:
                throw std::runtime_error("Unsupported optimizer type in HDF5 file.");
        }
    }
}