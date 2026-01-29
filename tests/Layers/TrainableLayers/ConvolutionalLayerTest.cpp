#include <gtest/gtest.h>
#include "Layers/TrainableLayers/Convolutional/ConvolutionalLayer.hpp"
#include "Utils/OpenCLResources.hpp"
#include "Utils/Dimensions.hpp"
#include "Utils/FilterDimensions.hpp"
#include "Utils/StrideDimensions.hpp"
#include "Utils/PaddingType.hpp"
#include <random>
#include <functional>

using namespace Layers::Trainable;
using namespace Utils;

static std::vector<float> cpuConvForward(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    const std::vector<float>& bias,
    size_t B,
    size_t IC, size_t IH, size_t IW,
    size_t OC,
    size_t FH, size_t FW,
    size_t OH, size_t OW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW
) {
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
    size_t B, size_t IC, size_t IH, size_t IW,
    size_t OC, size_t FH, size_t FW,
    size_t OH, size_t OW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW
) {
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
    size_t B, size_t IC, size_t IH, size_t IW,
    size_t OC, size_t FH, size_t FW,
    size_t OH, size_t OW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW
) {
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

class ConvolutionalLayerTest : public ::testing::Test {
protected:
    OpenCLResources ocl;
    std::mt19937 rng;

    const size_t B;
    const size_t IC;
    const size_t IH;
    const size_t IW;
    const size_t OC;
    const size_t FH;
    const size_t FW;
    const size_t strideH;
    const size_t strideW;
    const PaddingType padding;

    Dimensions inputDims;
    FilterDimensions filterDims;
    StrideDimensions strideDims;

    ConvolutionalLayer layer;

    ConvolutionalLayerTest()
        : ocl(Utils::OpenCLResources::createOpenCLResources()),
          rng(123),
          B(8), IC(3), IH(5), IW(5),
          OC(2), FH(3), FW(3),
          strideH(1), strideW(1),
          padding(PaddingType::Valid),
          inputDims{IC, IH, IW},
          filterDims{FH, FW, IC, OC},
          strideDims{strideH, strideW},
          layer{0, ocl.getSharedResources(), inputDims, filterDims, strideDims, padding, B, rng} 
    {}

    std::vector<float> randomVector(size_t size, float low=-1.0f, float high=1.0f) {
        std::uniform_real_distribution<float> dist(low, high);
        std::vector<float> v(size);
        for (auto& x : v) x = dist(rng);
        return v;
    }

    void checkForward(
        ConvolutionalLayer& layer,
        const std::vector<float>& inputs,
        size_t B
    ) {
        cl::Buffer inputBuf(
            ocl.getContext(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            inputs.size() * sizeof(float),
            const_cast<float*>(inputs.data())
        );

        layer.runForward(
            ocl.getForwardBackpropQueue(),
            inputBuf,
            B
        ).wait();

        const size_t OH = layer.getOutputHeight();
        const size_t OW = layer.getOutputWidth();

        std::vector<float> gpu(B * OC * OH * OW);
        ocl.getForwardBackpropQueue().enqueueReadBuffer(
            layer.getOutputs(),
            CL_TRUE,
            0,
            gpu.size() * sizeof(float),
            gpu.data()
        );

        auto cpu = cpuConvForward(
            inputs,
            layer.getWeightsCPU(ocl.getForwardBackpropQueue()),
            layer.getBiasesCPU(ocl.getForwardBackpropQueue()),
            B,
            IC, IH, IW,
            OC,
            FH, FW,
            OH, OW,
            strideH, strideW,
            layer.getPaddingValues().getTop(),
            layer.getPaddingValues().getLeft()
        );

        ASSERT_EQ(gpu.size(), cpu.size());

        for (size_t i = 0; i < gpu.size(); ++i)
            EXPECT_NEAR(gpu[i], cpu[i], 1e-4)
                << "Mismatch at index " << i;
    }

    void checkBackprop(
        ConvolutionalLayer& layer,
        const std::vector<float>& deltas,
        size_t B
    ) {
        cl::Buffer prevDeltaBuf(ocl.getContext(), CL_MEM_READ_WRITE, B * IC * IH * IW * sizeof(float));
        
        ocl.getForwardBackpropQueue().enqueueWriteBuffer(
            layer.getDeltas(), CL_TRUE, 0, deltas.size() * sizeof(float), deltas.data()
        );

        layer.backpropDeltas(ocl.getForwardBackpropQueue(), prevDeltaBuf, B).wait();

        std::vector<float> gpu(B * IC * IH * IW);
        ocl.getForwardBackpropQueue().enqueueReadBuffer(
            prevDeltaBuf, CL_TRUE, 0, gpu.size() * sizeof(float), gpu.data()
        );

        auto cpu = cpuConvBackpropDeltas(
            deltas, layer.getWeightsCPU(ocl.getForwardBackpropQueue()),
            B, IC, IH, IW, OC, FH, FW, layer.getOutputHeight(), layer.getOutputWidth(),
            strideH, strideW, layer.getPaddingValues().getTop(), layer.getPaddingValues().getLeft()
        );

        for (size_t i = 0; i < gpu.size(); ++i)
            EXPECT_NEAR(gpu[i], cpu[i], 1e-3);
    }

    void checkGradients(
        ConvolutionalLayer& layer,
        const std::vector<float>& inputs,
        const std::vector<float>& deltas,
        size_t B
    ) {
        cl::Buffer inputBuf(ocl.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            inputs.size() * sizeof(float), const_cast<float*>(inputs.data()));

        ocl.getForwardBackpropQueue().enqueueWriteBuffer(
            layer.getDeltas(), CL_TRUE, 0, deltas.size() * sizeof(float), deltas.data()
        );

        cl::Event empty;
        auto [wgEv, bgEv] = layer.computeGradients(ocl.getForwardBackpropQueue(), empty, inputBuf, B);
        wgEv.wait();
        bgEv.wait();

        std::vector<float> gpuW(OC * IC * FH * FW);
        std::vector<float> gpuB(OC);
        
        ocl.getForwardBackpropQueue().enqueueReadBuffer(layer.getWeightsGradients(), CL_TRUE, 0, gpuW.size() * sizeof(float), gpuW.data());
        ocl.getForwardBackpropQueue().enqueueReadBuffer(layer.getBiasesGradients(), CL_TRUE, 0, gpuB.size() * sizeof(float), gpuB.data());

        auto [cpuW, cpuB] = cpuConvGradients(
            inputs, deltas, B, IC, IH, IW, OC, FH, FW, layer.getOutputHeight(), layer.getOutputWidth(),
            strideH, strideW, layer.getPaddingValues().getTop(), layer.getPaddingValues().getLeft()
        );
        std::cout << "Weights Gradients Comparison:\n";
        for (size_t i = 0; i < gpuW.size(); ++i) EXPECT_NEAR(gpuW[i], cpuW[i], 1e-3);

        std::cout << "Biases Gradients Comparison:\n";
        for (size_t i = 0; i < gpuB.size(); ++i) EXPECT_NEAR(gpuB[i], cpuB[i], 1e-3);
    }

};

TEST_F(ConvolutionalLayerTest, ForwardRandom) {
    auto inputs = randomVector(B * IC * IH * IW);
    checkForward(layer, inputs, B);
}

TEST_F(ConvolutionalLayerTest, ForwardOnes) {
    std::vector<float> inputs(B * IC * IH * IW, 1.0f);
    checkForward(layer, inputs, B);
}

TEST_F(ConvolutionalLayerTest, ForwardBatch1) {
    auto inputs = randomVector(IC * IH * IW);
    checkForward(layer, inputs, 1);
}

TEST_F(ConvolutionalLayerTest, ForwardBatch4) {
    auto inputs = randomVector(4 * IC * IH * IW);
    checkForward(layer, inputs, 4);
}

TEST_F(ConvolutionalLayerTest, BackpropRandom) {
    auto deltas = randomVector(B * OC * layer.getOutputHeight() * layer.getOutputWidth());
    checkBackprop(layer, deltas, B);
}

TEST_F(ConvolutionalLayerTest, BackpropOnes) {
    std::vector<float> deltas(B * OC * layer.getOutputHeight() * layer.getOutputWidth(), 1.0f);
    checkBackprop(layer, deltas, B);
}

TEST_F(ConvolutionalLayerTest, BackpropBatch1) {
    auto deltas = randomVector(1 * OC * layer.getOutputHeight() * layer.getOutputWidth());
    checkBackprop(layer, deltas, 1);
}

TEST_F(ConvolutionalLayerTest, BackpropBatch4) {
    auto deltas = randomVector(4 * OC * layer.getOutputHeight() * layer.getOutputWidth());
    checkBackprop(layer, deltas, 4);
}


TEST_F(ConvolutionalLayerTest, BackpropBatch6) {
    auto deltas = randomVector(6 * OC * layer.getOutputHeight() * layer.getOutputWidth());
    checkBackprop(layer, deltas, 6);
}

TEST_F(ConvolutionalLayerTest, GradientsRandom) {
    auto inputs = randomVector(B * IC * IH * IW);
    auto deltas = randomVector(B * OC * layer.getOutputHeight() * layer.getOutputWidth());
    checkGradients(layer, inputs, deltas, B);
}

TEST_F(ConvolutionalLayerTest, GradientsOnes) {
    std::vector<float> inputs(B * IC * IH * IW, 1.0f);
    std::vector<float> deltas(B * OC * layer.getOutputHeight() * layer.getOutputWidth(), 1.0f);
    checkGradients(layer, inputs, deltas, B);
}

TEST_F(ConvolutionalLayerTest, GradientsBatch1) {
    auto inputs = randomVector(IC * IH * IW);
    auto deltas = randomVector(1 * OC * layer.getOutputHeight() * layer.getOutputWidth());
    checkGradients(layer, inputs, deltas, 1);
}

TEST_F(ConvolutionalLayerTest, GradientsBatch4) {
    auto inputs = randomVector(4 * IC * IH * IW);
    auto deltas = randomVector(4 * OC * layer.getOutputHeight() * layer.getOutputWidth());
    checkGradients(layer, inputs, deltas, 4);
}

TEST_F(ConvolutionalLayerTest, GradientsBatch6) {
    auto inputs = randomVector(6 * IC * IH * IW);
    auto deltas = randomVector(6 * OC * layer.getOutputHeight() * layer.getOutputWidth());
    checkGradients(layer, inputs, deltas, 6);
}
