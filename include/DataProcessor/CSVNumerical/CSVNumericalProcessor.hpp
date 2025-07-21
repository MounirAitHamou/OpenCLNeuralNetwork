#pragma once

#include "DataProcessor/DataProcessor.hpp"


class CSVNumericalProcessor : public DataProcessor {
private:
    std::vector<std::vector<float>> allData;
    std::vector<float> parseCSVLine(const std::string& line) const;
    void processHeader(const std::string& header_line, const std::vector<std::string>& input_columns, const std::vector<std::string>& target_columns);

public:
    CSVNumericalProcessor(const OpenCLSetup& ocl_setup, const size_t batch_size);

    ~CSVNumericalProcessor() override = default;


    void loadData(const std::string& source, const std::vector<std::string>& input_columns, const std::vector<std::string>& target_columns) override;
    size_t getTotalSamples() const override;
    size_t getInputSize() const override;
    size_t getTargetSize() const override;

    void splitData(float train_ratio, float val_ratio, unsigned int seed) override;
    Batch getBatch(size_t batch_start, size_t batch_size) override;

    const std::vector<size_t>& getTrainIndices() const override;
    const std::vector<size_t>& getValidationIndices() const override;
    const std::vector<size_t>& getTestIndices() const override;

    void shuffleCurrentPartition() override;



    void activateTrainPartition() override;
    void activateValidationPartition() override;
    void activateTestPartition() override;
};