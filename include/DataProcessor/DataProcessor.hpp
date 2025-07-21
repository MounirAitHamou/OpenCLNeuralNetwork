#pragma once
#include <CL/opencl.hpp>
#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <iostream>

#include "Utils/OpenCLSetup.hpp"

struct Batch {
    cl::Buffer inputs;
    cl::Buffer targets;
    size_t batch_actual_size;
};

class DataProcessorIterator;

class DataProcessor {
protected:
    size_t batch_size;
    std::vector<size_t>* currentActiveIndices = nullptr;

    std::string filePath;
    
    cl::Context context;
    cl::CommandQueue queue;

    std::vector<size_t> trainIndices;
    std::vector<size_t> validationIndices;
    std::vector<size_t> testIndices;

    std::vector<std::string> header;
    std::vector<size_t> input_columns_indices;
    std::vector<size_t> target_columns_indices;

    
    size_t numInputFeatures = 0;
    size_t numTargetFeatures = 0;
public:


    DataProcessor(const OpenCLSetup& ocl_setup, const size_t batch_size)
        : context(ocl_setup.context), queue(ocl_setup.queue), batch_size(batch_size) {}

    virtual ~DataProcessor() = default;
    
    virtual void loadData(const std::string& source, const std::vector<std::string>& input_columns, const std::vector<std::string>& target_columns) = 0;
    virtual size_t getTotalSamples() const = 0;
    virtual size_t getInputSize() const = 0;
    virtual size_t getTargetSize() const = 0;
    

    virtual void splitData(float train_ratio, float val_ratio, unsigned int seed) = 0;
    virtual Batch getBatch(size_t batch_start, size_t batch_size) = 0;

    virtual const std::vector<size_t>& getTrainIndices() const = 0;
    virtual const std::vector<size_t>& getValidationIndices() const = 0;
    virtual const std::vector<size_t>& getTestIndices() const = 0;

    virtual void shuffleCurrentPartition() = 0;

    virtual void activateTrainPartition() = 0;
    virtual void activateValidationPartition() = 0;
    virtual void activateTestPartition() = 0;

    virtual const std::vector<size_t>& getActivePartition() const {
        if (currentActiveIndices) {
            return *currentActiveIndices;
        }
        throw std::runtime_error("No active partition is set.");
    }

    size_t getBatchSize() const {
        return batch_size;
    }
    void setBatchSize(size_t size) {
        batch_size = size;
    }

    DataProcessorIterator begin();

    DataProcessorIterator end();
};



class DataProcessorIterator {
public:
    DataProcessorIterator(DataProcessor* processor, size_t pos)
        : processor(processor), pos(pos) {}

    bool operator==(const DataProcessorIterator& other) const {
        return pos == other.pos;
    }

    bool operator!=(const DataProcessorIterator& other) const {
        return pos != other.pos;
    }

    Batch operator*() const {
        if (pos >= processor->getTotalSamples()) {
            throw std::out_of_range("Batch position out of range");
        }
        return processor->getBatch(pos, processor->getBatchSize());
    }

    DataProcessorIterator& operator++() {
        pos = std::min(pos + processor->getBatchSize(), processor->getActivePartition().size());
        return *this;
    }

private:
    DataProcessor* processor;
    size_t pos;
};

