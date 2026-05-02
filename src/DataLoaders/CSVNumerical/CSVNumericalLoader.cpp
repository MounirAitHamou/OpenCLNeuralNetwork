#include "DataLoaders/CSVNumerical/CSVNumericalLoader.hpp"
namespace DataLoaders
{
    CSVNumericalLoader::CSVNumericalLoader(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                                           const size_t p_batchSize,
                                           std::vector<std::string> p_inputColumns,
                                           std::vector<std::string> p_targetColumns)
        : DataLoader(std::move(p_sharedResources), p_batchSize),
          m_inputColumns(std::move(p_inputColumns)),
          m_targetColumns(std::move(p_targetColumns)) {}

    Utils::Batch CSVNumericalLoader::getBatch(size_t p_batchStart, size_t p_batchSize) const
    {
        if (m_currentActiveIndices == nullptr)
        {
            std::cerr << "Error: No active data partition is set. Call activateTrainPartition, activateValidationPartition, or activateTestPartition before getting batches." << "\n";
            CLNN_FATAL("No data partition is active. Call activateTrainPartition, activateValidationPartition, or activateTestPartition before getting batches.");
        }

        std::vector<float> inputs;
        std::vector<float> targets;

        inputs.reserve(p_batchSize * m_numInputFeatures);
        targets.reserve(p_batchSize * m_numTargetFeatures);

        size_t currentPartitionSize = m_currentActiveIndices->size();
        size_t endIndex = std::min(p_batchStart + p_batchSize, currentPartitionSize);

        size_t batchActualSize = endIndex - p_batchStart;

        for (size_t i = p_batchStart; i < endIndex; ++i)
        {
            size_t sampleIdx = (*m_currentActiveIndices)[i];
            const std::vector<float> &row = m_allData[sampleIdx];

            for (const auto &inputIndex : m_inputColumnsIndices)
            {
                if (inputIndex >= row.size())
                {
                    CLNN_FATAL("Input column index out of bounds for sample " + std::to_string(sampleIdx));
                }
                inputs.push_back(row[inputIndex]);
            }

            for (const auto &targetIndex : m_targetColumnsIndices)
            {
                if (targetIndex >= row.size())
                {
                    CLNN_FATAL("Target column index out of bounds for sample " + std::to_string(sampleIdx));
                }
                targets.push_back(row[targetIndex]);
            }
        }

        cl::Buffer inputsBuffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * inputs.size(), inputs.data());
        cl::Buffer targetsBuffer(m_sharedResources->getContext(), CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(float) * targets.size(), targets.data());

        return Utils::Batch(inputsBuffer, targetsBuffer, inputs, targets, batchActualSize, Utils::Dimensions({m_numInputFeatures}), Utils::Dimensions({m_numTargetFeatures}));
    }

    void CSVNumericalLoader::loadData(const std::string &p_source)
    {
        if (m_inputColumns.empty() || m_targetColumns.empty())
        {
            CLNN_FATAL("Input and target columns must be specified.");
        }

        std::ifstream file(p_source);
        if (!file)
        {
            CLNN_FATAL("Failed to open CSV file: " + p_source);
        }

        m_allData.clear();

        std::string line;
        bool headerProcessed = false;

        while (std::getline(file, line))
        {
            if (!headerProcessed)
            {
                headerProcessed = true;
                processHeader(line);
                continue;
            }

            auto row = parseCSVLine(line);
            if (row.size() != m_numInputFeatures + m_numTargetFeatures)
            {
                std::cerr << "Warning: Row size mismatch. Expected "
                          << (m_numInputFeatures + m_numTargetFeatures)
                          << ", got " << row.size()
                          << ". Skipping row." << "\n";
                continue;
            }

            m_allData.push_back(std::move(row));
        }

        if (m_allData.empty())
        {
            CLNN_FATAL("No data loaded from CSV file: " + p_source +
                       ". File might be empty or malformed.");
        }

        if (m_allData[0].size() < 2)
        {
            CLNN_FATAL("CSV data must have at least one input and one target column.");
        }
        file.close();
    }

    void CSVNumericalLoader::splitData(float p_trainRatio, float p_valRatio, size_t p_seed)
    {
        if (p_trainRatio < 0.0F || p_valRatio < 0.0F || p_trainRatio + p_valRatio > 1.0F)
        {
            CLNN_FATAL("Invalid train or validation ratios. They must be non-negative and sum to less than or equal to 1.0.");
        }

        std::vector<size_t> allIndices(getTotalSamples());
        std::iota(allIndices.begin(), allIndices.end(), 0);

        std::mt19937 g(static_cast<unsigned long>(p_seed));
        std::shuffle(allIndices.begin(), allIndices.end(), g);

        size_t totalSamples = getTotalSamples();
        auto numTrain = static_cast<size_t>(totalSamples * p_trainRatio);
        auto numVal = static_cast<size_t>(totalSamples * p_valRatio);

        m_trainIndices.assign(allIndices.begin(), allIndices.begin() + numTrain);
        m_validationIndices.assign(allIndices.begin() + numTrain, allIndices.begin() + numTrain + numVal);
        m_testIndices.assign(allIndices.begin() + numTrain + numVal, allIndices.end());

        activateTrainPartition();
    }

    void CSVNumericalLoader::shuffleCurrentPartition(std::mt19937 &p_rng)
    {
        if (m_currentActiveIndices == nullptr)
        {
            CLNN_FATAL("No data partition is active to shuffle. Call activateTrainPartition, activateValidationPartition, or activateTestPartition first.");
        }
        std::shuffle(m_currentActiveIndices->begin(), m_currentActiveIndices->end(), p_rng);
    }

    size_t CSVNumericalLoader::getTotalSamples() const
    {
        return m_allData.size();
    }

    size_t CSVNumericalLoader::getInputSize() const
    {
        return m_numInputFeatures;
    }

    size_t CSVNumericalLoader::getTargetSize() const
    {
        return m_numTargetFeatures;
    }

    const std::vector<size_t> CSVNumericalLoader::getTrainIndices() const
    {
        return m_trainIndices;
    }

    const std::vector<size_t> CSVNumericalLoader::getValidationIndices() const
    {
        return m_validationIndices;
    }

    const std::vector<size_t> CSVNumericalLoader::getTestIndices() const
    {
        return m_testIndices;
    }

    void CSVNumericalLoader::activateTrainPartition()
    {
        m_currentActiveIndices = &m_trainIndices;
    }

    void CSVNumericalLoader::activateValidationPartition()
    {
        m_currentActiveIndices = &m_validationIndices;
    }

    void CSVNumericalLoader::activateTestPartition()
    {
        m_currentActiveIndices = &m_testIndices;
    }

    std::vector<float> CSVNumericalLoader::parseCSVLine(const std::string &p_line)
    {
        std::vector<float> row;
        std::stringstream ss(p_line);
        std::string cell;

        while (std::getline(ss, cell, ','))
        {
            float value = 0.0F;

            auto result = std::from_chars(
                cell.data(),
                cell.data() + cell.size(),
                value);

            if (result.ec == std::errc())
            {
                row.push_back(value);
            }
            else
            {
                std::cerr << "Warning: Could not convert '"
                          << cell
                          << "' to float. Defaulting to 0.0.\n";
                row.push_back(0.0F);
            }
        }
        return row;
    }

    void CSVNumericalLoader::processHeader(const std::string &p_headerLine)
    {
        m_header.clear();
        m_inputColumnsIndices.clear();
        m_targetColumnsIndices.clear();

        std::stringstream ss(p_headerLine);
        std::string column;
        size_t index = 0;

        while (std::getline(ss, column, ','))
        {
            m_header.push_back(column);
            if (std::ranges::find(m_inputColumns, column) != m_inputColumns.end())
            {
                m_inputColumnsIndices.push_back(index);
            }
            if (std::ranges::find(m_targetColumns, column) != m_targetColumns.end())
            {
                m_targetColumnsIndices.push_back(index);
            }
            index++;
        }

        m_numInputFeatures = m_inputColumnsIndices.size();
        m_numTargetFeatures = m_targetColumnsIndices.size();
    }
}