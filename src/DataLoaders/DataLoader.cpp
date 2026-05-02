#include "DataLoaders/DataLoader.hpp"
namespace DataLoaders
{
    DataLoaderIterator DataLoader::begin()
    {
        return {this, 0};
    }

    DataLoaderIterator DataLoader::end()
    {
        return {this, getActivePartition().size()};
    }
}