#pragma once

#include "DataLoaders/DataLoader.hpp"
namespace DataLoaders {
    class CSVNumericalLoader : public DataLoader {
    public:
        CSVNumericalLoader(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                           size_t p_batchSize,
                           std::vector<std::string> p_inputColumns,
                           std::vector<std::string> p_targetColumns);

        ~CSVNumericalLoader() override = default;

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

        void setInputColumns(const std::vector<std::string>& p_inputColumns) {
            m_inputColumns = p_inputColumns;
        }
        void setTargetColumns(const std::vector<std::string>& p_targetColumns) {
            m_targetColumns = p_targetColumns;
        }
        
    private:
        std::vector<std::string> m_inputColumns;
        std::vector<std::string> m_targetColumns;
        std::vector<size_t> m_inputColumnsIndices;
        std::vector<size_t> m_targetColumnsIndices;
        std::vector<std::string> m_header;
        size_t m_numInputFeatures = 0;
        size_t m_numTargetFeatures = 0;


        std::vector<float> parseCSVLine(const std::string& p_line) const;
        void processHeader(const std::string& p_headerLine);
    };
}