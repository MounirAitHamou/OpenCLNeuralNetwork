#pragma once 

#include "Utils/Dimensions.hpp"
#include "Utils/OpenCLResources.hpp"
#include "Utils/LossFunctionType.hpp"

namespace LossFunctions {
    class LossFunction {
    public:
        LossFunction(std::shared_ptr<Utils::SharedResources> p_sharedResources)
            : m_sharedResources(p_sharedResources) {}

        virtual ~LossFunction() = default;

        virtual Utils::LossFunctionType getType() const = 0;

        float computeLoss(const cl::CommandQueue& queue,
                        std::vector<cl::Event>& waitList,
                        const cl::Buffer& predictions,
                        const cl::Buffer& targets,
                        size_t outputElements,
                        size_t batchSize) {
            std::vector<float> hostTargets(outputElements * batchSize);
            cl::Event readEvent;
            queue.enqueueReadBuffer(targets, NON_BLOCKING_READ, NO_OFFSET,
                                    hostTargets.size() * sizeof(float),
                                    hostTargets.data(),
                                    &waitList, &readEvent);
            waitList.push_back(readEvent);
            return computeLoss(queue, waitList, predictions, hostTargets, outputElements, batchSize);
        }

        float computeLoss(const cl::CommandQueue& queue,
                        std::vector<cl::Event>& waitList,
                        const cl::Buffer& predictions,
                        const std::vector<float>& targets,
                        size_t outputElements,
                        size_t batchSize) {
            std::vector<float> hostPredictions(outputElements * batchSize);
            cl::Event readEvent;
            queue.enqueueReadBuffer(predictions, NON_BLOCKING_READ, NO_OFFSET,
                                    hostPredictions.size() * sizeof(float),
                                    hostPredictions.data(),
                                    &waitList, &readEvent);
            waitList.push_back(readEvent);
            cl_int err = cl::Event::waitForEvents(waitList);
            return computeLoss(hostPredictions, targets, outputElements, batchSize);
        }
        
        virtual cl::Event computeLossGradient(const cl::CommandQueue& queue,
                                            const cl::Buffer& predictions,
                                            const cl::Buffer& targets,
                                            cl::Buffer& outputGradients,
                                            size_t outputElements,
                                            size_t batchSize) = 0;

        virtual bool equals(const LossFunction& other) const {
            return getType() == other.getType();
        }
        virtual void print() const {
            std::cout << "Loss Function Type: " << Utils::lossFunctionTypeToString(getType()) << "\n";
        }
        virtual void save(H5::Group& p_lossFunctionGroup) const {
            Utils::writeValueToHDF5<unsigned int>(p_lossFunctionGroup, "lossFunctionType", static_cast<unsigned int>(getType()));
        }

    protected:
        std::shared_ptr<Utils::SharedResources> m_sharedResources;
        cl::Kernel m_gradientKernel;

        virtual void setupKernel() = 0;

        virtual float computeLoss(const std::vector<float>& predictions,
                                const std::vector<float>& targets,
                                size_t outputElements,
                                size_t batchSize) = 0;
    };
}
