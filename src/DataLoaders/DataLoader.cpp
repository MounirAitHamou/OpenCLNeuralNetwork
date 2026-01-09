#include "DataLoaders/DataLoader.hpp"
namespace DataLoaders {
    DataLoaderIterator DataLoader::begin() {
        return DataLoaderIterator(this, 0);
    }

    DataLoaderIterator DataLoader::end() {
        return DataLoaderIterator(this, getActivePartition().size());
    }
}