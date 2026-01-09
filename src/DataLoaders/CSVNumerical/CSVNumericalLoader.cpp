#include "DataLoaders/CSVNumerical/CSVNumericalLoader.hpp"
namespace DataLoaders {
    CSVNumericalLoader::CSVNumericalLoader(std::shared_ptr<Utils::SharedResources> p_sharedResources, const size_t p_batchSize)
        : DataLoader(p_sharedResources, p_batchSize) {}

    Utils::Batch CSVNumericalLoader::getBatch(size_t p_batchStart, size_t p_batchSize) const {
        if (!m_currentActiveIndices || m_currentActiveIndices == nullptr) {
            std::cerr << "Error: No active data partition is set. Call activateTrainPartition, activateValidationPartition, or activateTestPartition before getting batches." << std::endl;
            throw std::runtime_error("No data partition is active. Call activateTrainPartition, activateValidationPartition, or activateTestPartition before getting batches.");
        }

        std::vector<float> inputs;
        std::vector<float> targets;

        inputs.reserve(p_batchSize * m_numInputFeatures);
        targets.reserve(p_batchSize * m_numTargetFeatures);

        size_t currentPartitionSize = m_currentActiveIndices->size();
        size_t endIndex = std::min(p_batchStart + p_batchSize, currentPartitionSize);

        size_t batchActualSize = endIndex - p_batchStart;

        for (size_t i = p_batchStart; i < endIndex; ++i) {
            size_t sampleIdx = (*m_currentActiveIndices)[i];
            const std::vector<float>& row = m_allData[sampleIdx];

            for (auto& inputIndex : m_inputColumnsIndices) {
                if (inputIndex >= row.size()) {
                    throw std::runtime_error("Input column index out of bounds for sample " + std::to_string(sampleIdx));
                }
                inputs.push_back(row[inputIndex]);
            }

            for (auto& targetIndex : m_targetColumnsIndices) {
                if (targetIndex >= row.size()) {
                    throw std::runtime_error("Target column index out of bounds for sample " + std::to_string(sampleIdx));
                }
                targets.push_back(row[targetIndex]);
            }
        }
        
        cl::Buffer inputsBuffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputs.size(), inputs.data());
        cl::Buffer targetsBuffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * targets.size(), targets.data());

        return Utils::Batch(inputsBuffer, targetsBuffer, inputs, targets, batchActualSize, Utils::Dimensions({m_numInputFeatures}), Utils::Dimensions({m_numTargetFeatures}));
    }

    void CSVNumericalLoader::loadData(const std::string& p_source,
                                        std::vector<std::string> p_inputColumns,
                                        std::vector<std::string> p_targetColumns) {
        if (p_inputColumns.empty() || p_targetColumns.empty()) {
            throw std::invalid_argument("Input and target columns must be specified.");
        }

        m_filePath = p_source;
        std::ifstream file(m_filePath);
        if (!file) {
            throw std::runtime_error("Failed to open CSV file: " + m_filePath);
        }
        
        m_allData.clear();

        std::string line;
        bool headerProcessed = false;

        while (std::getline(file, line)) {
            if (!headerProcessed) {
                headerProcessed = true;
                processHeader(line, p_inputColumns, p_targetColumns);
                continue;
            }

            auto row = parseCSVLine(line);
            if (row.size() != m_numInputFeatures + m_numTargetFeatures) {
                std::cerr << "Warning: Row size mismatch. Expected "
                        << (m_numInputFeatures + m_numTargetFeatures)
                        << ", got " << row.size()
                        << ". Skipping row." << std::endl;
                continue;
            }

            m_allData.push_back(std::move(row));
        }

        if (m_allData.empty()) {
            throw std::runtime_error("No data loaded from CSV file: " + m_filePath +
                                    ". File might be empty or malformed.");
        }

        if (m_allData[0].size() < 2) {
            throw std::runtime_error("CSV data must have at least one input and one target column.");
        }
        file.close();
    }

    void CSVNumericalLoader::splitData(float p_trainRatio, float p_valRatio, size_t p_seed) {
        if (p_trainRatio < 0.0f || p_valRatio < 0.0f || p_trainRatio + p_valRatio > 1.0f) {
            throw std::invalid_argument("Invalid train or validation ratios. They must be non-negative and sum to less than or equal to 1.0.");
        }

        std::vector<size_t> allIndices(getTotalSamples());
        std::iota(allIndices.begin(), allIndices.end(), 0);


        std::mt19937 g(static_cast<unsigned long>(p_seed));
        std::shuffle(allIndices.begin(), allIndices.end(), g);

        size_t totalSamples = getTotalSamples();
        size_t numTrain = static_cast<size_t>(totalSamples * p_trainRatio);
        size_t numVal = static_cast<size_t>(totalSamples * p_valRatio);

        m_trainIndices.assign(allIndices.begin(), allIndices.begin() + numTrain);
        m_validationIndices.assign(allIndices.begin() + numTrain, allIndices.begin() + numTrain + numVal);
        m_testIndices.assign(allIndices.begin() + numTrain + numVal, allIndices.end());

        activateTrainPartition();
    }

    void CSVNumericalLoader::shuffleCurrentPartition() {
        if (!m_currentActiveIndices) {
            throw std::runtime_error("No data partition is active to shuffle. Call activateTrainPartition, activateValidationPartition, or activateTestPartition first.");
        }

        std::random_device randomDevice;
        std::mt19937 g(randomDevice());

        std::shuffle(m_currentActiveIndices->begin(), m_currentActiveIndices->end(), g);
    }

    const size_t CSVNumericalLoader::getTotalSamples() const {
        return m_allData.size();
    }

    const size_t CSVNumericalLoader::getInputSize() const {
        return m_numInputFeatures;
    }

    const size_t CSVNumericalLoader::getTargetSize() const {
        return m_numTargetFeatures;
    }

    const std::vector<size_t> CSVNumericalLoader::getTrainIndices() const {
        return m_trainIndices;
    }

    const std::vector<size_t> CSVNumericalLoader::getValidationIndices() const {
        return m_validationIndices;
    }

    const std::vector<size_t> CSVNumericalLoader::getTestIndices() const {
        return m_testIndices;
    }

    void CSVNumericalLoader::activateTrainPartition() {
        m_currentActiveIndices = &m_trainIndices;
    }

    void CSVNumericalLoader::activateValidationPartition() {
        m_currentActiveIndices = &m_validationIndices;
    }

    void CSVNumericalLoader::activateTestPartition() {
        m_currentActiveIndices = &m_testIndices;
    }

    std::vector<float> CSVNumericalLoader::parseCSVLine(const std::string& p_line) const {
        std::vector<float> row;
        std::stringstream ss(p_line);
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

    void CSVNumericalLoader::processHeader(const std::string& p_headerLine,
                        std::vector<std::string> p_inputColumns,
                        std::vector<std::string> p_targetColumns) {
        m_header.clear();
        m_inputColumnsIndices.clear();
        m_targetColumnsIndices.clear();

        std::stringstream ss(p_headerLine);
        std::string column;
        size_t index = 0;

        while (std::getline(ss, column, ',')) {
            m_header.push_back(column);
            if (std::find(p_inputColumns.begin(), p_inputColumns.end(), column) != p_inputColumns.end()) {
                m_inputColumnsIndices.push_back(index);
            }
            if (std::find(p_targetColumns.begin(), p_targetColumns.end(), column) != p_targetColumns.end()) {
                m_targetColumnsIndices.push_back(index);
            }
            index++;
        }

        m_numInputFeatures = m_inputColumnsIndices.size();
        m_numTargetFeatures = m_targetColumnsIndices.size();
    }
}