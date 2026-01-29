#include "DataLoaders/BinImage/BinImageDataLoader.hpp"
#include <fstream>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <numeric>

namespace DataLoaders {
    BinImageDataLoader::BinImageDataLoader(
        std::shared_ptr<Utils::SharedResources> p_sharedResources,
        size_t p_batchSize,
        size_t p_width,
        size_t p_height,
        size_t p_channels,
        bool p_hasLabel,
        DataOrder p_inputOrder,
        DataOrder p_outputOrder,
        size_t p_numClasses
    )
        : DataLoader(p_sharedResources, p_batchSize),
        m_width(p_width),
        m_height(p_height),
        m_channels(p_channels),
        m_hasLabel(p_hasLabel),
        m_numClasses(p_numClasses),
        m_inputOrder(p_inputOrder),
        m_outputOrder(p_outputOrder)
    {}

    size_t BinImageDataLoader::index(int p_x, int p_y, int p_c, DataOrder p_o) const {
        const size_t W = m_width;
        const size_t H = m_height;
        const size_t C = m_channels;

        switch (p_o) {
            case DataOrder::CHW: return p_c*H*W + p_y*W + p_x;
            case DataOrder::HWC: return (p_y*W + p_x)*C + p_c;
            case DataOrder::CWH: return p_c*W*H + p_x*H + p_y;
            case DataOrder::WHC: return (p_x*H + p_y)*C + p_c;
            case DataOrder::HCW: return p_y*C*W + p_c*W + p_x;
            case DataOrder::WCH: return p_x*C*H + p_c*H + p_y;
        }
        return 0;
    }

    Utils::Dimensions BinImageDataLoader::getInputDimensions(const size_t p_channels, const size_t p_height, const size_t p_width, DataOrder p_o) const {
        switch(p_o) {
            case DataOrder::CHW: return Utils::Dimensions({p_channels, p_height, p_width});
            case DataOrder::HWC: return Utils::Dimensions({p_height, p_width, p_channels});
            case DataOrder::CWH: return Utils::Dimensions({p_channels, p_width, p_height});
            case DataOrder::WHC: return Utils::Dimensions({p_width, p_height, p_channels});
            case DataOrder::HCW: return Utils::Dimensions({p_height, p_channels, p_width});
            case DataOrder::WCH: return Utils::Dimensions({p_width, p_channels, p_height});
        }
        return Utils::Dimensions();
    }

    void BinImageDataLoader::loadData(const std::string& p_source) {
        std::ifstream f(p_source, std::ios::binary);
        if (!f)
            throw std::runtime_error("Failed to open binary image file");

        const size_t imageBytes = m_width * m_height * m_channels;
        const size_t recordBytes = imageBytes + (m_hasLabel ? 1 : 0);

        f.seekg(0, std::ios::end);
        const size_t fileSize = f.tellg();
        f.seekg(0, std::ios::beg);

        const size_t N = fileSize / recordBytes;

        size_t sampleSize = imageBytes + (m_hasLabel ? m_numClasses : 0);
        m_allData.resize(N, std::vector<float>(sampleSize));

        std::vector<uint8_t> tmp(imageBytes);

        for (size_t n = 0; n < N; ++n) {
            int label = 0;

            if (m_hasLabel) {
                uint8_t lbl;
                f.read(reinterpret_cast<char*>(&lbl), 1);
                label = static_cast<int>(lbl);
                std::vector<float>& sample = m_allData[n];
                for (size_t c = 0; c < m_numClasses; ++c)
                    sample[imageBytes + c] = (c == label ? 1.0f : 0.0f);
            }

            f.read(reinterpret_cast<char*>(tmp.data()), imageBytes);

            std::vector<float>& sample = m_allData[n];
            for (int y = 0; y < m_height; ++y)
            for (int x = 0; x < m_width;  ++x)
            for (int c = 0; c < m_channels; ++c) {
                size_t in  = index(x, y, c, m_inputOrder);
                size_t out = index(x, y, c, m_outputOrder);
                sample[out] = static_cast<float>(tmp[in]) / 255.0f;
            }
        }

        m_trainIndices.resize(N);
        std::iota(m_trainIndices.begin(), m_trainIndices.end(), 0);
        m_currentActiveIndices = &m_trainIndices;
    }

    Utils::Batch BinImageDataLoader::getBatch(size_t p_batchStart, size_t p_batchSize) const {
        const auto& idx = *m_currentActiveIndices;
        const size_t end = std::min(p_batchStart + p_batchSize, idx.size());

        size_t imageSize = m_width * m_height * m_channels;

        std::vector<float> inputs((end - p_batchStart) * imageSize);
        std::vector<float> targets;
        if (m_hasLabel)
            targets.resize((end - p_batchStart) * m_numClasses);

        for (size_t i = p_batchStart; i < end; ++i) {
            size_t id = idx[i];
            const auto& sample = m_allData[id];

            std::copy(
                sample.begin(),
                sample.begin() + imageSize,
                inputs.begin() + (i - p_batchStart) * imageSize
            );

            if (m_hasLabel) {
                std::copy(
                    sample.begin() + imageSize,
                    sample.end(),
                    targets.begin() + (i - p_batchStart) * m_numClasses
                );
            }
        }

        cl::Buffer inputBuffer = Utils::createCLBuffer(
            m_sharedResources->getContext(),
            inputs
        );

        cl::Buffer targetBuffer = Utils::createCLBuffer(
            m_sharedResources->getContext(),
            targets
        );

        return Utils::Batch(
            std::move(inputBuffer),
            std::move(targetBuffer),
            std::move(inputs),
            std::move(targets),
            end - p_batchStart,
            getInputDimensions(m_channels, m_height, m_width, m_inputOrder),
            Utils::Dimensions({m_hasLabel ? m_numClasses : 0})
        );
    }

    void BinImageDataLoader::splitData(float p_train, float p_val, size_t p_seed) {
        std::vector<size_t> all(m_allData.size());
        std::iota(all.begin(), all.end(), 0);

        std::mt19937 rng(static_cast<unsigned long>(p_seed));
        std::shuffle(all.begin(), all.end(), rng);

        size_t nTrain = static_cast<size_t>(all.size() * p_train);
        size_t nVal   = static_cast<size_t>(all.size() * p_val);
        size_t nTest  = all.size() - nTrain - nVal;

        m_trainIndices.assign(all.begin(), all.begin() + nTrain);
        m_validationIndices.assign(all.begin() + nTrain, all.begin() + nTrain + nVal);
        m_testIndices.assign(all.begin() + nTrain + nVal, all.end());
    }

    void BinImageDataLoader::shuffleCurrentPartition(std::mt19937& p_rng) {
        if (!m_currentActiveIndices)
            throw std::runtime_error("No active partition");
        std::shuffle(
            m_currentActiveIndices->begin(),
            m_currentActiveIndices->end(),
            p_rng
        );
    }

    const size_t BinImageDataLoader::getTotalSamples() const {
        return m_allData.size();
    }

    const size_t BinImageDataLoader::getInputSize() const {
        return m_width * m_height * m_channels;
    }

    const size_t BinImageDataLoader::getTargetSize() const {
        return m_hasLabel ? m_numClasses : 0;
    }

    const std::vector<size_t> BinImageDataLoader::getTrainIndices() const {
        return m_trainIndices;
    }

    const std::vector<size_t> BinImageDataLoader::getValidationIndices() const {
        return m_validationIndices;
    }

    const std::vector<size_t> BinImageDataLoader::getTestIndices() const {
        return m_testIndices;
    }

    void BinImageDataLoader::activateTrainPartition() {
        m_currentActiveIndices = &m_trainIndices;
    }

    void BinImageDataLoader::activateValidationPartition() {
        m_currentActiveIndices = &m_validationIndices;
    }

    void BinImageDataLoader::activateTestPartition() {
        m_currentActiveIndices = &m_testIndices;
    }

}
