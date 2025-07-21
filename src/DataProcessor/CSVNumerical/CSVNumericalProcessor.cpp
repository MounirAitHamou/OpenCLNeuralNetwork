#include "DataProcessor/CSVNumerical/CSVNumericalProcessor.hpp"

#include <sstream>
#include <iostream>

CSVNumericalProcessor::CSVNumericalProcessor(const OpenCLSetup& ocl_setup, const size_t batch_size)
    : DataProcessor(ocl_setup, batch_size) {}

void CSVNumericalProcessor::processHeader(const std::string& header_line, const std::vector<std::string>& input_columns, const std::vector<std::string>& target_columns) {
    header.clear();
    input_columns_indices.clear();
    target_columns_indices.clear();

    std::stringstream ss(header_line);
    std::string column;
    size_t index = 0;

    while (std::getline(ss, column, ',')) {
        header.push_back(column);
        if (std::find(input_columns.begin(), input_columns.end(), column) != input_columns.end()) {
            input_columns_indices.push_back(index);
        }
        if (std::find(target_columns.begin(), target_columns.end(), column) != target_columns.end()) {
            target_columns_indices.push_back(index);
        }
        index++;
    }

    numInputFeatures = input_columns_indices.size();
    numTargetFeatures = target_columns_indices.size();
}

std::vector<float> CSVNumericalProcessor::parseCSVLine(const std::string& line) const {
    std::vector<float> row;
    std::stringstream ss(line);
    std::string cell;

    while (std::getline(ss, cell, ',')) {
        try {
            row.push_back(std::stof(cell));
        } catch (const std::invalid_argument& e) {
            std::cerr << "Warning: Could not convert '" << cell << "' to float. Defaulting to 0.0. Error: " << e.what() << std::endl;
            row.push_back(0.0f);
        } catch (const std::out_of_range& e) {
            std::cerr << "Warning: Value '" << cell << "' out of float range. Defaulting to 0.0. Error: " << e.what() << std::endl;
            row.push_back(0.0f);
        }
    }
    return row;
}


void CSVNumericalProcessor::loadData(const std::string& source, const std::vector<std::string>& input_columns, const std::vector<std::string>& target_columns) {
    if (input_columns.empty() || target_columns.empty()) {
        throw std::invalid_argument("Input and target columns must be specified.");
    }
    filePath = source;
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file: " + filePath);
    }
    
    allData.clear();

    std::string line;
    bool header_processed = false;

    while (std::getline(file, line)) {

        if (!header_processed) {
            header_processed = true;
            processHeader(line, input_columns, target_columns);
            continue;
        }

        std::vector<float> row = parseCSVLine(line);
        if (!row.empty()) {
            allData.push_back(row);
        }
    }
    file.close();

    if (allData.empty()) {
        throw std::runtime_error("No data loaded from CSV file: " + filePath + ". File might be empty or malformed.");
    }

    if (allData[0].size() < 2) {
        throw std::runtime_error("CSV data must have at least one input and one target column.");
    }
}

size_t CSVNumericalProcessor::getTotalSamples() const {
    return allData.size();
}

size_t CSVNumericalProcessor::getInputSize() const {
    return numInputFeatures;
}

size_t CSVNumericalProcessor::getTargetSize() const {
    return numTargetFeatures;
}

void CSVNumericalProcessor::splitData(float train_ratio, float val_ratio, unsigned int seed) {
    if (train_ratio < 0.0f || val_ratio < 0.0f || train_ratio + val_ratio > 1.0f) {
        throw std::invalid_argument("Invalid train or validation ratios. They must be non-negative and sum to less than or equal to 1.0.");
    }

    std::vector<size_t> allIndices(getTotalSamples());
    std::iota(allIndices.begin(), allIndices.end(), 0);

    std::mt19937 g(seed);
    std::shuffle(allIndices.begin(), allIndices.end(), g);

    size_t total_samples = getTotalSamples();
    size_t num_train = static_cast<size_t>(total_samples * train_ratio);
    size_t num_val = static_cast<size_t>(total_samples * val_ratio);

    trainIndices.assign(allIndices.begin(), allIndices.begin() + num_train);
    validationIndices.assign(allIndices.begin() + num_train, allIndices.begin() + num_train + num_val);
    testIndices.assign(allIndices.begin() + num_train + num_val, allIndices.end());

    activateTrainPartition();
}

Batch CSVNumericalProcessor::getBatch(size_t batch_start, size_t batch_size) {
    if (!currentActiveIndices) {
        throw std::runtime_error("No data partition is active. Call activateTrainPartition, activateValidationPartition, or activateTestPartition before getting batches.");
    }

    std::vector<float> inputs;
    std::vector<float> targets;

    inputs.reserve(batch_size * numInputFeatures);
    targets.reserve(batch_size * numTargetFeatures);

    size_t current_partition_size = currentActiveIndices->size();
    size_t end_index = std::min(batch_start + batch_size, current_partition_size);

    size_t batch_actual_size = end_index - batch_start;

    for (size_t i = batch_start; i < end_index; ++i) {
        size_t sample_idx = (*currentActiveIndices)[i];
        const std::vector<float>& row = allData[sample_idx];

        for (auto& input_index : input_columns_indices) {
            if (input_index >= row.size()) {
                throw std::runtime_error("Input column index out of bounds for sample " + std::to_string(sample_idx));
            }
            inputs.push_back(row[input_index]);
        }

        for (auto& target_index : target_columns_indices) {
            if (target_index >= row.size()) {
                throw std::runtime_error("Target column index out of bounds for sample " + std::to_string(sample_idx));
            }
            targets.push_back(row[target_index]);
        }
    }

    cl::Buffer batch_inputs(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputs.size(), inputs.data());
    cl::Buffer batch_targets(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * targets.size(), targets.data());

    return Batch{batch_inputs, batch_targets, batch_actual_size};
}

const std::vector<size_t>& CSVNumericalProcessor::getTrainIndices() const {
    return trainIndices;
}

const std::vector<size_t>& CSVNumericalProcessor::getValidationIndices() const {
    return validationIndices;
}

const std::vector<size_t>& CSVNumericalProcessor::getTestIndices() const {
    return testIndices;
}

void CSVNumericalProcessor::shuffleCurrentPartition() {
    if (!currentActiveIndices) {
        throw std::runtime_error("No data partition is active to shuffle. Call activateTrainPartition, activateValidationPartition, or activateTestPartition first.");
    }

    std::random_device rd;
    std::mt19937 g(rd());

    std::shuffle(currentActiveIndices->begin(), currentActiveIndices->end(), g);
}

void CSVNumericalProcessor::activateTrainPartition() {
    currentActiveIndices = &trainIndices;
}

void CSVNumericalProcessor::activateValidationPartition() {
    currentActiveIndices = &validationIndices;
}

void CSVNumericalProcessor::activateTestPartition() {
    currentActiveIndices = &testIndices;
}