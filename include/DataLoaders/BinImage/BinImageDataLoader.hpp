#pragma once

#include "DataLoaders/DataLoader.hpp"
#include <vector>
#include <cstdint>
namespace DataLoaders {
    class BinImageDataLoader : public DataLoader {
    public:
        enum class DataOrder {
            CHW, HWC, CWH, WHC, HCW, WCH
        };

        BinImageDataLoader(
            std::shared_ptr<Utils::SharedResources> p_sharedResources,
            size_t p_batchSize,
            size_t p_width,
            size_t p_height,
            size_t p_channels,
            bool p_hasLabel,
            DataOrder p_inputOrder,
            DataOrder p_outputOrder,
            size_t p_numClasses = 0
        );

        ~BinImageDataLoader() override = default;

        Utils::Batch getBatch(size_t p_batchStart, size_t p_batchSize) const override;

        void loadData(const std::string& p_source) override;

        void splitData(float p_trainRatio, float p_valRatio, size_t p_seed) override;
        void shuffleCurrentPartition(size_t p_seed) override {
            std::mt19937 rng(static_cast<unsigned long>(p_seed));
            shuffleCurrentPartition(rng);
        }
        void shuffleCurrentPartition(std::mt19937& p_rng) override;

        const size_t getTotalSamples() const override;
        const size_t getInputSize() const override;
        const size_t getTargetSize() const override;

        const std::vector<size_t> getTrainIndices() const override;
        const std::vector<size_t> getValidationIndices() const override;
        const std::vector<size_t> getTestIndices() const override;

        void activateTrainPartition() override;
        void activateValidationPartition() override;
        void activateTestPartition() override;

    private:

        size_t m_width;
        size_t m_height;
        size_t m_channels;
        bool m_hasLabel;
        size_t m_numClasses;
        DataOrder m_inputOrder;
        DataOrder m_outputOrder;

        size_t index(
            int x, int y, int c,
            DataOrder order
        ) const;

        Utils::Dimensions getInputDimensions(
            size_t p_channels,
            size_t p_height,
            size_t p_width,
            DataOrder p_o
        ) const;
    };
}