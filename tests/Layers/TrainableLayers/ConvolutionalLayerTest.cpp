#include <gtest/gtest.h>
#include "Layers/TrainableLayers/Convolutional/ConvolutionalLayer.hpp"
#include "Utils/OpenCLResources.hpp"
#include "Utils/Dimensions.hpp"
#include "Utils/FilterDimensions.hpp"
#include "Utils/StrideDimensions.hpp"
#include "Utils/PaddingType.hpp"
#include <random>
#include <functional>
#include <utility>

using namespace Layers::Trainable;
using namespace Utils;

static std::vector<float> cpuConvForward(
    const std::vector<float> &input,
    const std::vector<float> &weights,
    const std::vector<float> &bias,
    size_t B,
    size_t IC, size_t IH, size_t IW,
    size_t OC,
    size_t FH, size_t FW,
    size_t OH, size_t OW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW)
{
    std::vector<float> out(B * OC * OH * OW, 0.0F);

    for (size_t b = 0; b < B; ++b)
    {
        for (size_t oc = 0; oc < OC; ++oc)
        {
            for (size_t oh = 0; oh < OH; ++oh)
            {
                for (size_t ow = 0; ow < OW; ++ow)
                {

                    float acc = bias[oc];

                    for (size_t ic = 0; ic < IC; ++ic)
                    {
                        for (size_t fh = 0; fh < FH; ++fh)
                        {
                            for (size_t fw = 0; fw < FW; ++fw)
                            {

                                int ih = static_cast<int>(oh * strideH) - static_cast<int>(padH) + static_cast<int>(fh);
                                int iw = static_cast<int>(ow * strideW) - static_cast<int>(padW) + static_cast<int>(fw);

                                if (ih < 0 || std::cmp_greater_equal(ih, IH) ||
                                    iw < 0 || std::cmp_greater_equal(iw, IW))
                                {
                                    continue;
                                }

                                size_t inIdx =
                                    (b * IC * IH * IW) +
                                    (ic * IH * IW) +
                                    (ih * IW) +
                                    iw;

                                size_t wIdx =
                                    (oc * IC * FH * FW) +
                                    (ic * FH * FW) +
                                    (fh * FW) +
                                    fw;

                                acc += input[inIdx] * weights[wIdx];
                            }
                        }
                    }

                    out[(b * OC * OH * OW) +
                        (oc * OH * OW) +
                        (oh * OW) +
                        ow] = acc;
                }
            }
        }
    }

    return out;
}

static std::vector<float> cpuConvBackpropDeltas(
    const std::vector<float> &deltas,
    const std::vector<float> &weights,
    size_t B, size_t IC, size_t IH, size_t IW,
    size_t OC, size_t FH, size_t FW,
    size_t OH, size_t OW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW)
{
    std::vector<float> prevDeltas(B * IC * IH * IW, 0.0F);

    for (size_t b = 0; b < B; ++b)
    {
        for (size_t oc = 0; oc < OC; ++oc)
        {
            for (size_t oh = 0; oh < OH; ++oh)
            {
                for (size_t ow = 0; ow < OW; ++ow)
                {
                    float dOut = deltas[(b * OC * OH * OW) + (oc * OH * OW) + (oh * OW) + ow];
                    for (size_t ic = 0; ic < IC; ++ic)
                    {
                        for (size_t fh = 0; fh < FH; ++fh)
                        {
                            for (size_t fw = 0; fw < FW; ++fw)
                            {
                                int ih = static_cast<int>(oh * strideH) - static_cast<int>(padH) + static_cast<int>(fh);
                                int iw = static_cast<int>(ow * strideW) - static_cast<int>(padW) + static_cast<int>(fw);

                                if (ih >= 0 && std::cmp_less(ih, IH) && iw >= 0 && std::cmp_less(iw, IW))
                                {
                                    size_t inIdx = (b * IC * IH * IW) + (ic * IH * IW) + (ih * IW) + iw;
                                    size_t wIdx = (oc * IC * FH * FW) + (ic * FH * FW) + (fh * FW) + fw;
                                    prevDeltas[inIdx] += dOut * weights[wIdx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return prevDeltas;
}

static std::pair<std::vector<float>, std::vector<float>> cpuConvGradients(
    const std::vector<float> &inputs,
    const std::vector<float> &deltas,
    size_t B, size_t IC, size_t IH, size_t IW,
    size_t OC, size_t FH, size_t FW,
    size_t OH, size_t OW,
    size_t strideH, size_t strideW,
    size_t padH, size_t padW)
{
    std::vector<float> dw(OC * IC * FH * FW, 0.0F);
    std::vector<float> db(OC, 0.0F);

    for (size_t b = 0; b < B; ++b)
    {
        for (size_t oc = 0; oc < OC; ++oc)
        {
            for (size_t oh = 0; oh < OH; ++oh)
            {
                for (size_t ow = 0; ow < OW; ++ow)
                {
                    float dOut = deltas[(b * OC * OH * OW) + (oc * OH * OW) + (oh * OW) + ow];
                    db[oc] += dOut;
                    for (size_t ic = 0; ic < IC; ++ic)
                    {
                        for (size_t fh = 0; fh < FH; ++fh)
                        {
                            for (size_t fw = 0; fw < FW; ++fw)
                            {
                                int ih = static_cast<int>(oh * strideH) - static_cast<int>(padH) + static_cast<int>(fh);
                                int iw = static_cast<int>(ow * strideW) - static_cast<int>(padW) + static_cast<int>(fw);

                                if (ih >= 0 && std::cmp_less(ih, IH) && iw >= 0 && std::cmp_less(iw, IW))
                                {
                                    size_t inIdx = (b * IC * IH * IW) + (ic * IH * IW) + (ih * IW) + iw;
                                    size_t wIdx = (oc * IC * FH * FW) + (ic * FH * FW) + (fh * FW) + fw;
                                    dw[wIdx] += dOut * inputs[inIdx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    for (auto &val : dw)
    {
        val /= static_cast<float>(B);
    }
    for (auto &val : db)
    {
        val /= static_cast<float>(B);
    }
    return {dw, db};
}

class ConvolutionalLayerTest : public ::testing::Test
{
protected:
    OpenCLResources ocl;
    std::mt19937 rng;

    const size_t B{1}; // B(1) to test setBatchSize functionality in forward/backprop/gradients checks
    const size_t IC{3};
    const size_t IH{5};
    const size_t IW{5};
    const size_t OC{2};
    const size_t FH{3};
    const size_t FW{3};
    const size_t strideH{1};
    const size_t strideW{1};
    const PaddingType padding{PaddingType::Valid};

    Dimensions inputDims;
    FilterDimensions filterDims;
    StrideDimensions strideDims;

    ConvolutionalLayer layer;

    ConvolutionalLayerTest()
        : ocl(Utils::OpenCLResources::createOpenCLResources()),
          rng(123),
          inputDims{IC, IH, IW},
          filterDims{FH, FW, IC, OC},
          strideDims{strideH, strideW},
          layer{0, ocl.getSharedResources(), inputDims, filterDims, strideDims, padding, B, rng}
    {
    }

    std::vector<float> randomVector(size_t size, float low = -1.0F, float high = 1.0F)
    {
        std::uniform_real_distribution<float> dist(low, high);
        std::vector<float> v(size);
        for (auto &x : v)
        {
            x = dist(rng);
        }
        return v;
    }

    void checkForward(
        ConvolutionalLayer &p_layer,
        const std::vector<float> &inputs,
        size_t p_B)
    {
        cl::Buffer inputBuf(
            ocl.getContext(),
            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
            inputs.size() * sizeof(float),
            const_cast<float *>(inputs.data()));

        p_layer.runForward(
                   ocl.getForwardBackpropQueue(),
                   inputBuf,
                   p_B)
            .wait();

        const size_t OH = p_layer.getOutputHeight();
        const size_t OW = p_layer.getOutputWidth();

        std::vector<float> gpu(p_B * OC * OH * OW);
        ocl.getForwardBackpropQueue().enqueueReadBuffer(
            p_layer.getOutputs(),
            BLOCKING,
            NO_OFFSET,
            gpu.size() * sizeof(float),
            gpu.data());

        auto cpu = cpuConvForward(
            inputs,
            p_layer.getWeightsCPU(ocl.getForwardBackpropQueue()),
            p_layer.getBiasesCPU(ocl.getForwardBackpropQueue()),
            p_B,
            IC, IH, IW,
            OC,
            FH, FW,
            OH, OW,
            strideH, strideW,
            p_layer.getPaddingValues().getTop(),
            p_layer.getPaddingValues().getLeft());

        ASSERT_EQ(gpu.size(), cpu.size());

        for (size_t i = 0; i < gpu.size(); ++i)
        {
            EXPECT_NEAR(gpu[i], cpu[i], 1e-4)
                << "Mismatch at index " << i;
        }
    }

    void checkBackprop(
        ConvolutionalLayer &p_layer,
        const std::vector<float> &deltas,
        size_t p_B)
    {
        cl::Buffer prevDeltaBuf(ocl.getContext(), CL_MEM_READ_WRITE, p_B * IC * IH * IW * sizeof(float));

        p_layer.enqueueWriteDeltasBuffer(ocl.getForwardBackpropQueue(), deltas, p_B);

        p_layer.backpropDeltas(ocl.getForwardBackpropQueue(), prevDeltaBuf, p_B).wait();

        std::vector<float> gpu(p_B * IC * IH * IW);
        ocl.getForwardBackpropQueue().enqueueReadBuffer(
            prevDeltaBuf, BLOCKING, NO_OFFSET, gpu.size() * sizeof(float), gpu.data());

        auto cpu = cpuConvBackpropDeltas(
            deltas, p_layer.getWeightsCPU(ocl.getForwardBackpropQueue()),
            p_B, IC, IH, IW, OC, FH, FW, p_layer.getOutputHeight(), p_layer.getOutputWidth(),
            strideH, strideW, p_layer.getPaddingValues().getTop(), p_layer.getPaddingValues().getLeft());

        for (size_t i = 0; i < gpu.size(); ++i)
        {
            EXPECT_NEAR(gpu[i], cpu[i], 1e-3);
        }
    }

    void checkGradients(
        ConvolutionalLayer &p_layer,
        const std::vector<float> &inputs,
        const std::vector<float> &deltas,
        size_t p_B)
    {
        cl::Buffer inputBuf(ocl.getContext(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                            inputs.size() * sizeof(float), const_cast<float *>(inputs.data()));

        p_layer.enqueueWriteDeltasBuffer(ocl.getForwardBackpropQueue(), deltas, p_B);

        cl::Event empty;
        auto [wgEv, bgEv] = p_layer.computeGradients(ocl.getForwardBackpropQueue(), empty, inputBuf, p_B);
        wgEv.wait();
        bgEv.wait();

        std::vector<float> gpuW(OC * IC * FH * FW);
        std::vector<float> gpuB(OC);

        ocl.getForwardBackpropQueue().enqueueReadBuffer(p_layer.getWeightsGradients(), BLOCKING, NO_OFFSET, gpuW.size() * sizeof(float), gpuW.data());
        ocl.getForwardBackpropQueue().enqueueReadBuffer(p_layer.getBiasesGradients(), BLOCKING, NO_OFFSET, gpuB.size() * sizeof(float), gpuB.data());

        auto [cpuW, cpuB] = cpuConvGradients(
            inputs, deltas, p_B, IC, IH, IW, OC, FH, FW, p_layer.getOutputHeight(), p_layer.getOutputWidth(),
            strideH, strideW, p_layer.getPaddingValues().getTop(), p_layer.getPaddingValues().getLeft());
        std::cout << "Weights Gradients Comparison:\n";
        for (size_t i = 0; i < gpuW.size(); ++i)
        {
            EXPECT_NEAR(gpuW[i], cpuW[i], 1e-3);
        }

        std::cout << "Biases Gradients Comparison:\n";
        for (size_t i = 0; i < gpuB.size(); ++i)
        {
            EXPECT_NEAR(gpuB[i], cpuB[i], 1e-3);
        }
    }
};

TEST_F(ConvolutionalLayerTest, ForwardRandom)
{
    auto inputs = randomVector(B * IC * IH * IW);
    checkForward(layer, inputs, B);
}

TEST_F(ConvolutionalLayerTest, ForwardOnes)
{
    std::vector<float> inputs(B * IC * IH * IW, 1.0F);
    checkForward(layer, inputs, B);
}

TEST_F(ConvolutionalLayerTest, ForwardBatch1)
{
    auto inputs = randomVector(IC * IH * IW);
    checkForward(layer, inputs, 1);
}

TEST_F(ConvolutionalLayerTest, ForwardBatch4)
{
    auto inputs = randomVector(4 * IC * IH * IW);
    checkForward(layer, inputs, 4);
}

TEST_F(ConvolutionalLayerTest, BackpropRandom)
{
    auto deltas = randomVector(B * OC * layer.getOutputHeight() * layer.getOutputWidth());
    checkBackprop(layer, deltas, B);
}

TEST_F(ConvolutionalLayerTest, BackpropOnes)
{
    std::vector<float> deltas(B * OC * layer.getOutputHeight() * layer.getOutputWidth(), 1.0F);
    checkBackprop(layer, deltas, B);
}

TEST_F(ConvolutionalLayerTest, BackpropBatch1)
{
    auto deltas = randomVector(1 * OC * layer.getOutputHeight() * layer.getOutputWidth());
    checkBackprop(layer, deltas, 1);
}

TEST_F(ConvolutionalLayerTest, BackpropBatch4)
{
    auto deltas = randomVector(4 * OC * layer.getOutputHeight() * layer.getOutputWidth());
    checkBackprop(layer, deltas, 4);
}

TEST_F(ConvolutionalLayerTest, BackpropBatch6)
{
    auto deltas = randomVector(6 * OC * layer.getOutputHeight() * layer.getOutputWidth());
    checkBackprop(layer, deltas, 6);
}

TEST_F(ConvolutionalLayerTest, GradientsRandom)
{
    auto inputs = randomVector(B * IC * IH * IW);
    auto deltas = randomVector(B * OC * layer.getOutputHeight() * layer.getOutputWidth());
    checkGradients(layer, inputs, deltas, B);
}

TEST_F(ConvolutionalLayerTest, GradientsOnes)
{
    std::vector<float> inputs(B * IC * IH * IW, 1.0F);
    std::vector<float> deltas(B * OC * layer.getOutputHeight() * layer.getOutputWidth(), 1.0F);
    checkGradients(layer, inputs, deltas, B);
}

TEST_F(ConvolutionalLayerTest, GradientsBatch1)
{
    auto inputs = randomVector(IC * IH * IW);
    auto deltas = randomVector(1 * OC * layer.getOutputHeight() * layer.getOutputWidth());
    checkGradients(layer, inputs, deltas, 1);
}

TEST_F(ConvolutionalLayerTest, GradientsBatch4)
{
    auto inputs = randomVector(4 * IC * IH * IW);
    auto deltas = randomVector(4 * OC * layer.getOutputHeight() * layer.getOutputWidth());
    checkGradients(layer, inputs, deltas, 4);
}

TEST_F(ConvolutionalLayerTest, GradientsBatch6)
{
    auto inputs = randomVector(6 * IC * IH * IW);
    auto deltas = randomVector(6 * OC * layer.getOutputHeight() * layer.getOutputWidth());
    checkGradients(layer, inputs, deltas, 6);
}
