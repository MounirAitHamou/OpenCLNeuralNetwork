#pragma once

#include "DataLoaders/DataLoader.hpp"

class CSVNumericalLoader : public DataLoader {
public:
    CSVNumericalLoader(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                                size_t p_batchSize);

    ~CSVNumericalLoader() override = default;

    Batch getBatch(size_t p_batchStart, size_t p_batchSize) const override;

    void loadData(const std::string& p_source,
                  std::vector<std::string> p_inputColumns,
                  std::vector<std::string> p_targetColumns) override;

    void splitData(float p_trainRatio, float p_valRatio, size_t p_seed) override;
    void shuffleCurrentPartition() override;

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
    std::vector<std::vector<float>> m_allData;

    std::vector<float> parseCSVLine(const std::string& p_line) const;
    void processHeader(const std::string& p_headerLine,
                       std::vector<std::string> p_inputColumns,
                       std::vector<std::string> p_targetColumns);
};