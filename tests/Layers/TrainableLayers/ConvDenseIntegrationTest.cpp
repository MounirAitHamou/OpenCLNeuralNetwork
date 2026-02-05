#include <gtest/gtest.h>
#include "Utils/LayerArgs.hpp"
#include "Utils/OpenCLResources.hpp"
#include <random>
#include <functional>
#include <map>
#include <string>
#include "Utils/LossFunctionArgs.hpp"
#include "Utils/Dimensions.hpp"
#include "Utils/FilterDimensions.hpp"
#include "Utils/StrideDimensions.hpp"
#include "Utils/PaddingValues.hpp"

using namespace Layers::Trainable;
using namespace Utils;
using namespace LossFunctions;


static std::vector<float> cpuFlatten(const std::vector<float>& x) {
    return x;
}

struct LayerDims {
    LayerType type;
    size_t batchSize;
    Dimensions inputDims;
    Dimensions outputDims;
    FilterDimensions filterDims;
    StrideDimensions strideDims;
    PaddingValues paddingVals;

    LayerDims() = default;
    LayerDims(ConvolutionalLayer* conv)
        : type(LayerType::Convolutional),
          batchSize(conv->getBatchSize()),
          inputDims(conv->getInputDimensions()),
          outputDims(conv->getOutputDimensions()),
          filterDims(conv->getFilterDimensions()),
          strideDims(conv->getStrideDimensions()),
          paddingVals(conv->getPaddingValues())
    {}

    LayerDims(DenseLayer* dense)
        : type(LayerType::Dense),
          batchSize(dense->getBatchSize()),
          inputDims(dense->getInputDimensions()),
          outputDims(dense->getOutputDimensions())
    {}

};


static std::vector<float> cpuDenseForward(
    const std::vector<float>& inputs,
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    LayerDims dims
) {
    size_t B = dims.batchSize;
    size_t in = dims.inputDims.getTotalElements();
    size_t out = dims.outputDims.getTotalElements();
    std::vector<float> y(B * out, 0.0f);
    for (size_t b = 0; b < B; ++b)
        for (size_t o = 0; o < out; ++o) {
            float acc = biases[o];
            for (size_t i = 0; i < in; ++i)
                acc += inputs[b * in + i] * weights[o * in + i];
            y[b * out + o] = acc;
        }
    return y;
}

static std::vector<float> cpuDenseBackpropDeltas(
    const std::vector<float>& deltas,
    const std::vector<float>& weights,
    LayerDims dims
) {
    size_t B = dims.batchSize;
    size_t in = dims.inputDims.getTotalElements();
    size_t out = dims.outputDims.getTotalElements();
    std::vector<float> prev(B * in, 0.0f);
    for (size_t b = 0; b < B; ++b)
        for (size_t i = 0; i < in; ++i)
            for (size_t o = 0; o < out; ++o)
                prev[b * in + i] += deltas[b * out + o] * weights[o * in + i];
    return prev;
}

static std::pair<std::vector<float>, std::vector<float>>
cpuDenseGradients(
    const std::vector<float>& inputs,
    const std::vector<float>& deltas,
    LayerDims dims
) {
    size_t B = dims.batchSize;
    size_t in = dims.inputDims.getTotalElements();
    size_t out = dims.outputDims.getTotalElements();
    std::vector<float> weightGrad(out * in, 0.0f);
    std::vector<float> biasGrad(out, 0.0f);

    for (size_t b = 0; b < B; ++b) {
        for (size_t o = 0; o < out; ++o) {
            float d = deltas[b * out + o];
            biasGrad[o] += d;

            for (size_t i = 0; i < in; ++i) {
                weightGrad[o * in + i] += d * inputs[b * in + i];
            }
        }
    }

    const float invB = 1.0f / static_cast<float>(B);
    for (auto& g : weightGrad) g *= invB;
    for (auto& g : biasGrad)   g *= invB;

    return { weightGrad, biasGrad };
}

static std::vector<float> cpuConvForward(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    LayerDims dims
) {
    size_t B = dims.batchSize;
    size_t IC = dims.inputDims.getDimensions()[0];
    size_t IH = dims.inputDims.getDimensions()[1];
    size_t IW = dims.inputDims.getDimensions()[2];
    size_t OC = dims.outputDims.getDimensions()[0];
    size_t FH = dims.filterDims.getHeight();
    size_t FW = dims.filterDims.getWidth();
    size_t OH = dims.outputDims.getDimensions()[1];
    size_t OW = dims.outputDims.getDimensions()[2];
    size_t strideH = dims.strideDims.getHeight();
    size_t strideW = dims.strideDims.getWidth();
    size_t padH = dims.paddingVals.getTop();
    size_t padW = dims.paddingVals.getLeft();
    std::vector<float> out(B * OC * OH * OW, 0.0f);

    for (size_t b = 0; b < B; ++b)
        for (size_t oc = 0; oc < OC; ++oc)
            for (size_t oh = 0; oh < OH; ++oh)
                for (size_t ow = 0; ow < OW; ++ow) {

                    float acc = bias[oc];

                    for (size_t ic = 0; ic < IC; ++ic)
                        for (size_t fh = 0; fh < FH; ++fh)
                            for (size_t fw = 0; fw < FW; ++fw) {

                                int ih = int(oh * strideH) - int(padH) + int(fh);
                                int iw = int(ow * strideW) - int(padW) + int(fw);

                                if (ih < 0 || ih >= int(IH) ||
                                    iw < 0 || iw >= int(IW))
                                    continue;

                                size_t inIdx =
                                    b * IC * IH * IW +
                                    ic * IH * IW +
                                    ih * IW +
                                    iw;

                                size_t wIdx =
                                    oc * IC * FH * FW +
                                    ic * FH * FW +
                                    fh * FW +
                                    fw;

                                acc += input[inIdx] * weights[wIdx];
                            }

                    out[
                        b * OC * OH * OW +
                        oc * OH * OW +
                        oh * OW +
                        ow
                    ] = acc;
                }

    return out;
}

static std::vector<float> cpuConvBackpropDeltas(
    const std::vector<float>& deltas,
    const std::vector<float>& weights,
    LayerDims dims
) {
    size_t B = dims.batchSize;
    size_t IC = dims.inputDims.getDimensions()[0];
    size_t IH = dims.inputDims.getDimensions()[1];
    size_t IW = dims.inputDims.getDimensions()[2];
    size_t OC = dims.outputDims.getDimensions()[0];
    size_t FH = dims.filterDims.getHeight();
    size_t FW = dims.filterDims.getWidth();
    size_t OH = dims.outputDims.getDimensions()[1];
    size_t OW = dims.outputDims.getDimensions()[2];
    size_t strideH = dims.strideDims.getHeight();
    size_t strideW = dims.strideDims.getWidth();
    size_t padH = dims.paddingVals.getTop();
    size_t padW = dims.paddingVals.getLeft();
    std::vector<float> prevDeltas(B * IC * IH * IW, 0.0f);

    for (size_t b = 0; b < B; ++b)
        for (size_t oc = 0; oc < OC; ++oc)
            for (size_t oh = 0; oh < OH; ++oh)
                for (size_t ow = 0; ow < OW; ++ow) {
                    float dOut = deltas[b * OC * OH * OW + oc * OH * OW + oh * OW + ow];
                    for (size_t ic = 0; ic < IC; ++ic)
                        for (size_t fh = 0; fh < FH; ++fh)
                            for (size_t fw = 0; fw < FW; ++fw) {
                                int ih = int(oh * strideH) - int(padH) + int(fh);
                                int iw = int(ow * strideW) - int(padW) + int(fw);

                                if (ih >= 0 && ih < int(IH) && iw >= 0 && iw < int(IW)) {
                                    size_t inIdx = b * IC * IH * IW + ic * IH * IW + ih * IW + iw;
                                    size_t wIdx = oc * IC * FH * FW + ic * FH * FW + fh * FW + fw;
                                    prevDeltas[inIdx] += dOut * weights[wIdx];
                                }
                            }
                }
    return prevDeltas;
}

static std::pair<std::vector<float>, std::vector<float>> cpuConvGradients(
    const std::vector<float>& inputs,
    const std::vector<float>& deltas,
    LayerDims dims
) {
    size_t B = dims.batchSize;
    size_t IC = dims.inputDims.getDimensions()[0];
    size_t IH = dims.inputDims.getDimensions()[1];
    size_t IW = dims.inputDims.getDimensions()[2];
    size_t OC = dims.outputDims.getDimensions()[0];
    size_t FH = dims.filterDims.getHeight();
    size_t FW = dims.filterDims.getWidth();
    size_t OH = dims.outputDims.getDimensions()[1];
    size_t OW = dims.outputDims.getDimensions()[2];
    size_t strideH = dims.strideDims.getHeight();
    size_t strideW = dims.strideDims.getWidth();
    size_t padH = dims.paddingVals.getTop();
    size_t padW = dims.paddingVals.getLeft();
    std::vector<float> dw(OC * IC * FH * FW, 0.0f);
    std::vector<float> db(OC, 0.0f);

    for (size_t b = 0; b < B; ++b)
        for (size_t oc = 0; oc < OC; ++oc)
            for (size_t oh = 0; oh < OH; ++oh)
                for (size_t ow = 0; ow < OW; ++ow) {
                    float dOut = deltas[b * OC * OH * OW + oc * OH * OW + oh * OW + ow];
                    db[oc] += dOut;
                    for (size_t ic = 0; ic < IC; ++ic)
                        for (size_t fh = 0; fh < FH; ++fh)
                            for (size_t fw = 0; fw < FW; ++fw) {
                                int ih = int(oh * strideH) - int(padH) + int(fh);
                                int iw = int(ow * strideW) - int(padW) + int(fw);

                                if (ih >= 0 && ih < int(IH) && iw >= 0 && iw < int(IW)) {
                                    size_t inIdx = b * IC * IH * IW + ic * IH * IW + ih * IW + iw;
                                    size_t wIdx = oc * IC * FH * FW + ic * FH * FW + fh * FW + fw;
                                    dw[wIdx] += dOut * inputs[inIdx];
                                }
                            }
                }

    for (auto& val : dw) val /= static_cast<float>(B);
    for (auto& val : db) val /= static_cast<float>(B);
    return {dw, db};
}

static std::vector<float> cpuForward(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    LayerDims dims) {
    if (dims.type == LayerType::Dense) {
        return cpuDenseForward(input, weights, bias, dims);
    }
    else if (dims.type == LayerType::Convolutional) {
        return cpuConvForward(input, weights, bias, dims);
    }
    throw std::invalid_argument("Unsupported layer type for cpuForward.");
}

static std::vector<float> cpuBackpropDeltas(
    const std::vector<float>& deltas,
    const std::vector<float>& weights,
    LayerDims dims) {
    if (dims.type == LayerType::Dense) {
        return cpuDenseBackpropDeltas(deltas, weights, dims);
    }
    else if (dims.type == LayerType::Convolutional) {
        return cpuConvBackpropDeltas(deltas, weights, dims);
    }
    throw std::invalid_argument("Unsupported layer type for cpuBackpropDeltas.");
}

static std::pair<std::vector<float>, std::vector<float>> cpuComputeGradients(
    const std::vector<float>& inputs,
    const std::vector<float>& deltas,
    LayerDims dims) {
    if (dims.type == LayerType::Dense) {
        return cpuDenseGradients(inputs, deltas, dims);
    }
    else if (dims.type == LayerType::Convolutional) {
        return cpuConvGradients(inputs, deltas, dims);
    }
    throw std::invalid_argument("Unsupported layer type for cpuComputeGradients.");
}

class ConvDenseIntegrationTest : public ::testing::Test {
protected:
    OpenCLResources ocl = OpenCLResources::createOpenCLResources();
    std::mt19937 rng{12345};

    ConvDenseIntegrationTest()
    {}

    std::vector<float> randomVec(size_t n) {
        std::uniform_real_distribution<float> dist(-1.f, 1.f);
        std::vector<float> v(n);
        for (auto& x : v) x = dist(rng);
        return v;
    }

    LayerDims generateLayerDims(TrainableLayer* layer) {
        if (layer->getType() == Utils::LayerType::Dense) {
            DenseLayer* dense = dynamic_cast<DenseLayer*>(layer);
            return LayerDims(dense);
        }
        if (layer->getType() == Utils::LayerType::Convolutional) {
            ConvolutionalLayer* conv = dynamic_cast<ConvolutionalLayer*>(layer);
            return LayerDims(conv);
        }
        throw std::invalid_argument("Unsupported layer type for generating LayerDims.");
    }

    std::vector<float> meanSquaredError(
        const std::vector<float>& predictions,
        const std::vector<float>& targets,
        size_t batchSize
    ) {
        size_t n = predictions.size();
        std::vector<float> lossGrad(n);
        for (size_t i = 0; i < n; ++i) {
            lossGrad[i] = 2.0f * (predictions[i] - targets[i]) / static_cast<float>(n / batchSize);
        }
        return lossGrad;
    }

    std::vector<float> cpuLossFunction(
        const std::vector<float>& predictions,
        const std::vector<float>& targets,
        Utils::LossFunctionType lossFunction,
        size_t batchSize
    ) {
        if (lossFunction == Utils::LossFunctionType::MeanSquaredError) {
            return meanSquaredError(predictions, targets, batchSize);
        }
        throw std::invalid_argument("Unsupported loss function type for cpuLossFunction.");
    }

    std::map<std::string, std::vector<float>> cpuForwardBackwardRun(
        TrainableLayer* layer1,
        TrainableLayer* layer2,
        Utils::LossFunctionType lossFunction,
        std::vector<float>& inputBuf,
        std::vector<float>& targetBuf,
        OpenCLResources& ocl,
        size_t B
    ) {
        std::map<std::string, std::vector<float>> results;

        LayerDims dims1 = generateLayerDims(layer1);
        LayerDims dims2 = generateLayerDims(layer2);

        std::vector<float> layer1Out = cpuForward(inputBuf, layer1->getWeightsCPU(ocl.getForwardBackpropQueue()), layer1->getBiasesCPU(ocl.getForwardBackpropQueue()), dims1);
        results["layer1ForwardOutput"] = layer1Out;

        std::vector<float> layer2Out = cpuForward(layer1Out, layer2->getWeightsCPU(ocl.getForwardBackpropQueue()), layer2->getBiasesCPU(ocl.getForwardBackpropQueue()), dims2);
        results["layer2ForwardOutput"] = layer2Out;

        std::vector<float> lossGrad = cpuLossFunction(layer2Out, targetBuf, lossFunction, B);
        results["lossGradient"] = lossGrad;

        std::vector<float> layer2Deltas = lossGrad;
        std::vector<float> layer1Deltas = cpuBackpropDeltas(layer2Deltas, layer2->getWeightsCPU(ocl.getForwardBackpropQueue()), dims2);
        results["layer1BackwardDeltas"] = layer1Deltas;

        std::vector<float> inputDeltas = cpuBackpropDeltas(layer1Deltas, layer1->getWeightsCPU(ocl.getForwardBackpropQueue()), dims1);
        results["initialInputBackwardDeltas"] = inputDeltas;

        auto [layer1WeightGrad, layer1BiasGrad] = cpuComputeGradients(inputBuf, layer1Deltas, dims1);
        results["layer1WeightGradients"] = layer1WeightGrad;
        results["layer1BiasGradients"] = layer1BiasGrad;

        auto [layer2WeightGrad, layer2BiasGrad] = cpuComputeGradients(layer1Out, layer2Deltas, dims2);
        results["layer2WeightGradients"] = layer2WeightGrad;
        results["layer2BiasGradients"] = layer2BiasGrad;

        return results;
    }

    
    std::map<std::string, std::vector<float>> gpuForwardBackwardRun(
        TrainableLayer* layer1,
        TrainableLayer* layer2,
        LossFunction* lossFunction,
        std::vector<float>& inputBuf,
        std::vector<float>& targetBuf,
        size_t B,
        OpenCLResources& ocl
    ) {
        std::map<std::string, std::vector<float>> results;
        auto& q = ocl.getForwardBackpropQueue();
        cl::Buffer inputCLBuf = cl::Buffer(
            ocl.getSharedResources()->getContext(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            inputBuf.size() * sizeof(float),
            inputBuf.data()
        );
        layer1->runForward(q, inputCLBuf, B).wait();
        cl::Buffer layer1Out = layer1->getOutputs();
        std::vector<float> layer1ForwardOutputVec(layer1->getTotalOutputElements() * B);
        q.enqueueReadBuffer(layer1Out, BLOCKING_READ, NO_OFFSET,
            layer1ForwardOutputVec.size() * sizeof(float), layer1ForwardOutputVec.data());
        results["layer1ForwardOutput"] = layer1ForwardOutputVec;
        layer2->runForward(q, layer1Out, B).wait();
        cl::Buffer layer2Out = layer2->getOutputs();
        std::vector<float> layer2ForwardOutputVec(layer2->getTotalOutputElements() * B);
        q.enqueueReadBuffer(layer2Out, BLOCKING_READ, NO_OFFSET,
            layer2ForwardOutputVec.size() * sizeof(float), layer2ForwardOutputVec.data());
        results["layer2ForwardOutput"] = layer2ForwardOutputVec;
        cl::Buffer targetCLBuf = cl::Buffer(
            ocl.getSharedResources()->getContext(),
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            targetBuf.size() * sizeof(float),
            targetBuf.data()
        );
        lossFunction->computeLossGradient(
            q,
            layer2Out,
            targetCLBuf,
            layer2->getDeltas(),
            layer2->getTotalOutputElements(),
            B
        ).wait();
        std::vector<float> lossGradVec(layer2->getTotalOutputElements() * B);
        q.enqueueReadBuffer(layer2->getDeltas(), BLOCKING_READ, NO_OFFSET,
            lossGradVec.size() * sizeof(float), lossGradVec.data());
        results["lossGradient"] = lossGradVec;
        layer2->backpropDeltas(q, layer1->getDeltas(), B).wait();
        std::vector<float> layer1BackwardDeltasVec(layer1->getTotalOutputElements() * B);
        q.enqueueReadBuffer(layer1->getDeltas(), BLOCKING_READ, NO_OFFSET,
            layer1BackwardDeltasVec.size() * sizeof(float), layer1BackwardDeltasVec.data());
        results["layer1BackwardDeltas"] = layer1BackwardDeltasVec;
        cl::Buffer outputDeltas = cl::Buffer(ocl.getSharedResources()->getContext(), CL_MEM_READ_WRITE, B * layer1->getTotalInputElements() * sizeof(float));
        layer1->backpropDeltas(q, outputDeltas, B).wait();
        size_t elems = layer1->getTotalInputElements() * B;
        std::vector<float> outputDeltasVec(elems);
        q.enqueueReadBuffer(outputDeltas, BLOCKING_READ, NO_OFFSET, elems * sizeof(float), outputDeltasVec.data());
        results["initialInputBackwardDeltas"] = outputDeltasVec;
        cl::Event emptyEvent1 = cl::Event{};
        layer1->computeGradients(
            q,
            emptyEvent1,
            inputCLBuf,
            B
        );
        cl::Event emptyEvent2 = cl::Event{};
        layer2->computeGradients(
            q,
            emptyEvent2,
            layer1Out,
            B
        );
        q.finish();
        std::vector<float> layer1WeightGradVec(layer1->getWeightsSize());
        ocl.getDeltaToGradientQueue().enqueueReadBuffer(
            layer1->getWeightsGradients(),
            BLOCKING_READ,
            NO_OFFSET,
            layer1WeightGradVec.size() * sizeof(float),
            layer1WeightGradVec.data()
        );
        results["layer1WeightGradients"] = layer1WeightGradVec;
        std::vector<float> layer1BiasGradVec(layer1->getBiasesSize());
        ocl.getDeltaToGradientQueue().enqueueReadBuffer(
            layer1->getBiasesGradients(),
            BLOCKING_READ,
            NO_OFFSET,
            layer1BiasGradVec.size() * sizeof(float),
            layer1BiasGradVec.data()
        );
        results["layer1BiasGradients"] = layer1BiasGradVec;
        std::vector<float> layer2WeightGradVec(layer2->getWeightsSize());
        ocl.getDeltaToGradientQueue().enqueueReadBuffer(
            layer2->getWeightsGradients(),
            BLOCKING_READ,
            NO_OFFSET,
            layer2WeightGradVec.size() * sizeof(float),
            layer2WeightGradVec.data()
        );
        results["layer2WeightGradients"] = layer2WeightGradVec;
        std::vector<float> layer2BiasGradVec(layer2->getBiasesSize());
        ocl.getDeltaToGradientQueue().enqueueReadBuffer(
            layer2->getBiasesGradients(),
            BLOCKING_READ,
            NO_OFFSET,
            layer2BiasGradVec.size() * sizeof(float),
            layer2BiasGradVec.data()
        );
        results["layer2BiasGradients"] = layer2BiasGradVec;
        return results;
    }

    std::pair<std::unique_ptr<TrainableLayer>, std::unique_ptr<TrainableLayer>> createTestLayers(
        std::pair<std::unique_ptr<Utils::LayerArgs>, std::unique_ptr<Utils::LayerArgs>> layerArgsPair,
        std::shared_ptr<Utils::SharedResources> sharedResources,
        Utils::Dimensions inputDimensions,
        size_t batchSize
    ) {
        auto layer1 = layerArgsPair.first->createLayer(
            0,
            sharedResources,
            inputDimensions,
            batchSize,
            rng
        );
        auto layer2 = layerArgsPair.second->createLayer(
            1,
            sharedResources,
            layer1->getOutputDimensions(),
            batchSize,
            rng
        );
        auto* tl1 = dynamic_cast<TrainableLayer*>(layer1.get());
        auto* tl2 = dynamic_cast<TrainableLayer*>(layer2.get());

        if (!tl1 || !tl2) {
            throw std::invalid_argument(
                "Both layers must be TrainableLayer for ConvDenseIntegrationTest."
            );
        }

        std::unique_ptr<TrainableLayer> trainableLayer1(
            static_cast<TrainableLayer*>(layer1.release())
        );

        std::unique_ptr<TrainableLayer> trainableLayer2(
            static_cast<TrainableLayer*>(layer2.release())
        );

        return std::make_pair(
            std::move(trainableLayer1),
            std::move(trainableLayer2)
        );
    }
    void runIntegrationTest(
        std::unique_ptr<Utils::LayerArgs> layer1Args,
        std::unique_ptr<Utils::LayerArgs> layer2Args,
        const Utils::Dimensions& inputDimensions,
        size_t batchSize
    ) {
        auto layerArgsPair = std::make_pair(
            std::move(layer1Args),
            std::move(layer2Args)
        );

        auto [layer1, layer2] = createTestLayers(
            std::move(layerArgsPair),
            ocl.getSharedResources(),
            inputDimensions,
            batchSize
        );

        std::vector<float> inputBuf =
            randomVec(batchSize * inputDimensions.getTotalElements());

        std::vector<float> targetBuf =
            randomVec(batchSize * layer2->getTotalOutputElements());

        auto cpuResults = cpuForwardBackwardRun(
            layer1.get(),
            layer2.get(),
            Utils::LossFunctionType::MeanSquaredError,
            inputBuf,
            targetBuf,
            ocl,
            batchSize
        );

        auto lossFunction =
            Utils::makeMeanSquaredErrorLossFunctionArgs()
                ->createLossFunction(ocl.getSharedResources());

        auto gpuResults = gpuForwardBackwardRun(
            layer1.get(),
            layer2.get(),
            lossFunction.get(),
            inputBuf,
            targetBuf,
            batchSize,
            ocl
        );

        float tolerance = 5e-4f;

        const std::vector<std::string> keys = {
            "layer1ForwardOutput",
            "layer2ForwardOutput",
            "lossGradient",
            "layer1BackwardDeltas",
            "initialInputBackwardDeltas",
            "layer1WeightGradients",
            "layer1BiasGradients",
            "layer2WeightGradients",
            "layer2BiasGradients"
        };

        for (const auto& key : keys) {
            const auto& cpuVec = cpuResults[key];
            const auto& gpuVec = gpuResults[key];

            ASSERT_EQ(cpuVec.size(), gpuVec.size())
                << "Size mismatch for " << key;

            for (size_t i = 0; i < cpuVec.size(); ++i) {
                if (std::abs(cpuVec[i] - gpuVec[i]) > tolerance) {
                    ADD_FAILURE()
                        << "Mismatch at index " << i << " for " << key
                        << " cpu=" << cpuVec[i]
                        << " gpu=" << gpuVec[i];
                    break;
                }
            }
        }
    }

};



TEST_F(ConvDenseIntegrationTest, ConvDense) {
    runIntegrationTest(
        Utils::makeConvolutionalLayerArgs(
            Utils::FilterDimensions(3, 3, 1, 2),
            Utils::StrideDimensions(1, 1),
            Utils::PaddingType::Same
        ),
        Utils::makeDenseLayerArgs(Utils::Dimensions({4})),
        Utils::Dimensions({1, 5, 5}),
        2
    );
}

TEST_F(ConvDenseIntegrationTest, DenseConv) {
    runIntegrationTest(
        Utils::makeDenseLayerArgs(Utils::Dimensions({25})),
        Utils::makeConvolutionalLayerArgs(
            Utils::FilterDimensions(3, 3, 25, 2),
            Utils::StrideDimensions(1, 1),
            Utils::PaddingType::Same
        ),
        Utils::Dimensions({25}),
        2
    );
}

TEST_F(ConvDenseIntegrationTest, ConvConv) {
    runIntegrationTest(
        Utils::makeConvolutionalLayerArgs(
            Utils::FilterDimensions(3, 3, 1, 2),
            Utils::StrideDimensions(1, 1),
            Utils::PaddingType::Same
        ),
        Utils::makeConvolutionalLayerArgs(
            Utils::FilterDimensions(3, 3, 2, 3),
            Utils::StrideDimensions(1, 1),
            Utils::PaddingType::Same
        ),
        Utils::Dimensions({1, 5, 5}),
        2
    );
}

TEST_F(ConvDenseIntegrationTest, DenseDense) {
    runIntegrationTest(
        Utils::makeDenseLayerArgs(Utils::Dimensions({16})),
        Utils::makeDenseLayerArgs(Utils::Dimensions({4})),
        Utils::Dimensions({16}),
        2
    );
}

struct ConvStressConfig {
    Utils::FilterDimensions filter;
    Utils::StrideDimensions stride;
    Utils::PaddingType padding;
    Utils::Dimensions inputDims;
    size_t batch;
};

class ConvStressTest : public ConvDenseIntegrationTest,
                       public ::testing::WithParamInterface<ConvStressConfig> {
};

TEST_P(ConvStressTest, ConvDenseGeometryStress) {
    const auto& cfg = GetParam();

    auto convArgs = Utils::makeConvolutionalLayerArgs(
        cfg.filter, cfg.stride, cfg.padding
    );

    auto denseArgs = Utils::makeDenseLayerArgs(
        Utils::Dimensions({13})
    );

    auto argsPair = std::make_pair(
        std::move(convArgs),
        std::move(denseArgs)
    );

    auto [layer1, layer2] = createTestLayers(
        std::move(argsPair),
        ocl.getSharedResources(),
        cfg.inputDims,
        cfg.batch
    );

    std::vector<float> input =
        randomVec(cfg.batch * cfg.inputDims.getTotalElements());

    std::vector<float> target =
        randomVec(cfg.batch * layer2->getTotalOutputElements());

    auto cpu = cpuForwardBackwardRun(
        layer1.get(), layer2.get(),
        Utils::LossFunctionType::MeanSquaredError,
        input, target, ocl, cfg.batch
    );

    auto lossFn =
        Utils::makeMeanSquaredErrorLossFunctionArgs()
            ->createLossFunction(ocl.getSharedResources());

    auto gpu = gpuForwardBackwardRun(
        layer1.get(), layer2.get(), lossFn.get(),
        input, target, cfg.batch, ocl
    );

    const float tol = 5e-4f;

    for (const auto& [key, cpuVec] : cpu) {
        const auto& gpuVec = gpu.at(key);
        ASSERT_EQ(cpuVec.size(), gpuVec.size()) << key;

        for (size_t i = 0; i < cpuVec.size(); ++i) {
            if (std::abs(cpuVec[i] - gpuVec[i]) > tol) {
                ADD_FAILURE()
                    << key << " mismatch at " << i
                    << " cpu=" << cpuVec[i]
                    << " gpu=" << gpuVec[i];
                break;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    UglyGeometry,
    ConvStressTest,
    ::testing::Values(
        ConvStressConfig{
            {3, 5, 7, 11}, {1, 1}, Utils::PaddingType::Same,
            {7, 13, 17}, 3
        },
        ConvStressConfig{
            {1, 7, 3, 5}, {2, 1}, Utils::PaddingType::Valid,
            {3, 19, 11}, 4
        },
        ConvStressConfig{
            {3, 3, 64, 128}, {1, 1}, Utils::PaddingType::Same,
            {64, 9, 9}, 2
        },
        ConvStressConfig{
            {5, 5, 3, 7}, {3, 3}, Utils::PaddingType::Valid,
            {3, 31, 29}, 5
        }
    )
);
