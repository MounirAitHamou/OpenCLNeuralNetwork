#pragma once
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iostream>

#include "Utils/OpenCLResources.hpp"
#include "Utils/Batch.hpp"

namespace DataLoaders {

    class DataLoaderIterator;

    class DataLoader {
    public:
        DataLoader(std::shared_ptr<Utils::SharedResources> p_sharedResources,
                size_t p_batchSize)
            : m_sharedResources(p_sharedResources),
            m_batchSize(p_batchSize) {}

        virtual ~DataLoader() = default;

        virtual Utils::Batch getBatch(size_t p_batchStart, size_t p_batchSize) const = 0;

        virtual void loadData(const std::string& p_source,
                            std::vector<std::string> p_inputColumns,
                            std::vector<std::string> p_targetColumns) = 0;

        virtual void splitData(float p_trainRatio, float p_valRatio, size_t p_seed) = 0;
        virtual void shuffleCurrentPartition() = 0;

        virtual const  size_t getTotalSamples() const = 0;
        virtual const  size_t getInputSize() const = 0;
        virtual const  size_t getTargetSize() const = 0;

        virtual const  std::vector<size_t> getTrainIndices() const = 0;
        virtual const  std::vector<size_t> getValidationIndices() const = 0;
        virtual const  std::vector<size_t> getTestIndices() const = 0;

        virtual void activateTrainPartition() = 0;
        virtual void activateValidationPartition() = 0;
        virtual void activateTestPartition() = 0;

        virtual std::vector<size_t> getActivePartition() const {
            if (m_currentActiveIndices) {
                return *m_currentActiveIndices;
            }
            throw std::runtime_error("No active partition is set.");
        }

        size_t getBatchSize() const { return m_batchSize; }
        void setBatchSize(size_t p_size) { m_batchSize = p_size; }

        DataLoaderIterator begin();
        DataLoaderIterator end();

    protected:
        size_t m_batchSize;
        std::vector<size_t>* m_currentActiveIndices = nullptr;

        std::string m_filePath;

        std::shared_ptr<Utils::SharedResources> m_sharedResources;

        std::vector<size_t> m_trainIndices;
        std::vector<size_t> m_validationIndices;
        std::vector<size_t> m_testIndices;
        std::vector<size_t> m_inputColumnsIndices;
        std::vector<size_t> m_targetColumnsIndices;

        std::vector<std::string> m_header;

        size_t m_numInputFeatures = 0;
        size_t m_numTargetFeatures = 0;
    };

    class DataLoaderIterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type        = Utils::Batch;
        using difference_type   = std::ptrdiff_t;
        using pointer           = void;
        using reference         = Utils::Batch;

        DataLoaderIterator(DataLoader* p_loader, size_t p_pos)
            : m_loader(p_loader), m_pos(p_pos) {}

        bool operator==(const DataLoaderIterator& p_other) const {
            return m_pos == p_other.m_pos;
        }

        bool operator!=(const DataLoaderIterator& p_other) const {
            return m_pos != p_other.m_pos;
        }

        Utils::Batch operator*() const {
            if (!m_loader || m_pos >= m_loader->getTotalSamples()) {
                throw std::out_of_range("Batch position out of range");
            }
            return m_loader->getBatch(m_pos, m_loader->getBatchSize());
        }

        DataLoaderIterator& operator++() {
            if (!m_loader) return *this;
            m_pos = std::min(m_pos + m_loader->getBatchSize(),
                            m_loader->getActivePartition().size());
            return *this;
        }

    private:
        DataLoader* m_loader = nullptr;
        size_t m_pos = 0;
    };
}