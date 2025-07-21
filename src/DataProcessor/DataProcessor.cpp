#include "DataProcessor/DataProcessor.hpp"

DataProcessorIterator DataProcessor::begin() {
    return DataProcessorIterator(this, 0);
}


DataProcessorIterator DataProcessor::end() {
    return DataProcessorIterator(this, getActivePartition().size());
}