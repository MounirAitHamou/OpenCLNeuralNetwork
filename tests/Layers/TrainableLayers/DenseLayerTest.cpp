#include <gtest/gtest.h>
#include "Layers/TrainableLayers/Dense/DenseLayer.hpp"
#include "Utils/OpenCLResources.hpp"
#include "Utils/Dimensions.hpp"
#include <random>
#include <functional>

using namespace Layers::Trainable;
using namespace Utils;

static std::vector<float> cpuDenseForward(
    const std::vector<float>& inputs,
    const std::vector<float>& weights,
    const std::vector<float>& biases,
    size_t B, size_t in, size_t out
) {
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

static std::vector<float> cpuBackpropDeltas(
    const std::vector<float>& deltas,
    const std::vector<float>& weights,
    size_t B, size_t in, size_t out
) {
    std::vector<float> prev(B * in, 0.0f);
    for (size_t b = 0; b < B; ++b)
        for (size_t i = 0; i < in; ++i)
            for (size_t o = 0; o < out; ++o)
                prev[b * in + i] += deltas[b * out + o] * weights[o * in + i];
    return prev;
}

static std::vector<float> cpuWeightGradients(
    const std::vector<float>& inputs,
    const std::vector<float>& deltas,
    size_t B, size_t in, size_t out
) {
    std::vector<float> grad(out * in, 0.0f);
    for (size_t o = 0; o < out; ++o)
        for (size_t i = 0; i < in; ++i)
            for (size_t b = 0; b < B; ++b)
                grad[o * in + i] += deltas[b * out + o] * inputs[b * in + i];
    for (auto& g : grad)
        g /= static_cast<float>(B);
    return grad;
}

static std::vector<float> cpuBiasGradients(
    const std::vector<float>& deltas, size_t B, size_t out
) {
    std::vector<float> grad(out, 0.0f);
    for (size_t b = 0; b < B; ++b)
        for (size_t o = 0; o < out; ++o)
            grad[o] += deltas[b * out + o];
    for (auto& g : grad)
        g /= static_cast<float>(B);
    return grad;
}

class DenseLayerTest : public ::testing::Test {
protected:
    OpenCLResources ocl = Utils::OpenCLResources::createOpenCLResources();
    std::mt19937 rng{123};
    const size_t IN  = 16;
    const size_t OUT = 8;
    const size_t B   = 8;
    DenseLayer layer{0,
                     ocl.getSharedResources(),
                     Utils::Dimensions({IN}),
                     Utils::Dimensions({OUT}),
                     B,
                     rng};

    std::vector<float> randomVector(size_t size, float low=-1.0f, float high=1.0f) {
        std::uniform_real_distribution<float> dist(low, high);
        std::vector<float> v(size);
        for (auto& x : v) x = dist(rng);
        return v;
    }

    void checkForward(DenseLayer& layer,
                      std::vector<float> inputs,
                      size_t B, size_t IN, size_t OUT) 
    {
        cl::Buffer inputBuf(ocl.getContext(),
                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            inputs.size() * sizeof(float),
                            const_cast<float*>(inputs.data()));

        layer.runForward(
            ocl.getForwardBackpropQueue(), inputBuf, B).wait();

        std::vector<float> gpu(B * OUT);
        ocl.getForwardBackpropQueue().enqueueReadBuffer(
            layer.getOutputs(), CL_TRUE, 0,
            gpu.size() * sizeof(float), gpu.data()
        );

        auto cpu = cpuDenseForward(
            inputs,
            layer.getWeightsCPU(ocl.getForwardBackpropQueue()),
            layer.getBiasesCPU(ocl.getForwardBackpropQueue()),
            B, IN, OUT
        );

        for (size_t i = 0; i < gpu.size(); ++i)
            EXPECT_NEAR(gpu[i], cpu[i], 1e-4);
    }

    void checkBackprop(DenseLayer& layer,
                       std::vector<float> deltas,
                       size_t B, size_t IN, size_t OUT)
    {
        cl::Buffer prevDeltaBuf(ocl.getContext(),
                                CL_MEM_READ_WRITE,
                                B * IN * sizeof(float));

        ocl.getForwardBackpropQueue().enqueueWriteBuffer(
            layer.getDeltas(),
            CL_TRUE, 0, deltas.size() * sizeof(float), deltas.data()
        );

        layer.backpropDeltas(
            ocl.getForwardBackpropQueue(), prevDeltaBuf, B).wait();

        std::vector<float> gpu(B * IN);
        ocl.getForwardBackpropQueue().enqueueReadBuffer(
            prevDeltaBuf, CL_TRUE, 0,
            gpu.size() * sizeof(float), gpu.data()
        );

        auto cpu = cpuBackpropDeltas(
            deltas,
            layer.getWeightsCPU(ocl.getForwardBackpropQueue()),
            B, IN, OUT
        );

        for (size_t i = 0; i < gpu.size(); ++i)
            EXPECT_NEAR(gpu[i], cpu[i], 1e-4);
    }

    void checkGradients(DenseLayer& layer,
                        std::vector<float> inputs,
                        std::vector<float> deltas,
                        size_t B, size_t IN, size_t OUT)
    {
        cl::Buffer inputBuf(ocl.getContext(),
                            CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                            inputs.size() * sizeof(float),
                            inputs.data());

        ocl.getForwardBackpropQueue().enqueueWriteBuffer(
            layer.getDeltas(),
            CL_TRUE, 0, deltas.size() * sizeof(float), deltas.data()
        );

        cl::Event placeHolder{};
        auto [wgEv, bgEv] = layer.computeGradients(
            ocl.getForwardBackpropQueue(), placeHolder, inputBuf, B
        );
        wgEv.wait();
        bgEv.wait();

        std::vector<float> wgpu(OUT * IN), bgpu(OUT);
        ocl.getForwardBackpropQueue().enqueueReadBuffer(
            layer.getWeightsGradients(), CL_TRUE, 0,
            wgpu.size() * sizeof(float), wgpu.data()
        );
        ocl.getForwardBackpropQueue().enqueueReadBuffer(
            layer.getBiasesGradients(), CL_TRUE, 0,
            bgpu.size() * sizeof(float), bgpu.data()
        );

        auto wcpu = cpuWeightGradients(inputs, deltas, B, IN, OUT);
        auto bcpu = cpuBiasGradients(deltas, B, OUT);

        for (size_t i = 0; i < wgpu.size(); ++i)
            EXPECT_NEAR(wgpu[i], wcpu[i], 1e-4);
        for (size_t i = 0; i < bgpu.size(); ++i)
            EXPECT_NEAR(bgpu[i], bcpu[i], 1e-4);
    }
};

TEST_F(DenseLayerTest, ForwardRandom) {
    checkForward(layer, randomVector(B*IN), B, IN, OUT);
}

TEST_F(DenseLayerTest, BackpropRandom) {
    checkBackprop(layer, randomVector(B*OUT), B, IN, OUT);
}

TEST_F(DenseLayerTest, GradientsRandom) {
    auto inputs = randomVector(B*IN);
    auto deltas = randomVector(B*OUT);
    checkGradients(layer, inputs, deltas, B, IN, OUT);
}

TEST_F(DenseLayerTest, ForwardZeros) {
    checkForward(layer, std::vector<float>(B*IN, 0.0f), B, IN, OUT);
}

TEST_F(DenseLayerTest, ForwardOnes) {
    checkForward(layer, std::vector<float>(B*IN, 1.0f), B, IN, OUT);
}

TEST_F(DenseLayerTest, BackpropZeros) {
    checkBackprop(layer, std::vector<float>(B*OUT, 0.0f), B, IN, OUT);
}

TEST_F(DenseLayerTest, GradientsZeros) {
    auto inputs = std::vector<float>(B*IN, 0.0f);
    auto deltas = std::vector<float>(B*OUT, 0.0f);
    checkGradients(layer, inputs, deltas, B, IN, OUT);
}

TEST_F(DenseLayerTest, ForwardBatch2) {
    checkForward(layer, randomVector(2*IN), 2, IN, OUT);
}

TEST_F(DenseLayerTest, ForwardBatch8) {
    checkForward(layer, randomVector(8*IN), 8, IN, OUT);
}

TEST_F(DenseLayerTest, BackpropBatch2) {
    checkBackprop(layer, randomVector(2*OUT), 2, IN, OUT);
}

TEST_F(DenseLayerTest, BackpropBatch8) {
    checkBackprop(layer, randomVector(8*OUT), 8, IN, OUT);
}

TEST_F(DenseLayerTest, GradientsBatch2) {
    auto inputs = randomVector(2*IN);
    auto deltas = randomVector(2*OUT);
    checkGradients(layer, inputs, deltas, 2, IN, OUT);
}

TEST_F(DenseLayerTest, GradientsBatch8) {
    auto inputs = randomVector(8*IN);
    auto deltas = randomVector(8*OUT);
    checkGradients(layer, inputs, deltas, 8, IN, OUT);
}
