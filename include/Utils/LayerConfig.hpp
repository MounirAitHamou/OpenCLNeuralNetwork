#pragma once

#include "Layer/AllLayers.hpp"
#include "Utils/LossFunctionType.hpp"

#include <memory>
#include <stdexcept>
#include <vector>
#include <iterator>

namespace LayerConfig {

    struct LayerArgs {
 
        Dimensions output_dims;

        ActivationType activation_type;

        size_t batch_size;

        virtual ~LayerArgs() = default;

        LayerArgs(Dimensions out_dims, ActivationType activation_type, size_t batch_size)
            : output_dims(out_dims), activation_type(activation_type), batch_size(batch_size) {}

        virtual std::unique_ptr<Layer> createLayer(const size_t layer_id, const OpenCLSetup& ocl_setup, Dimensions in_dims) const = 0;

        virtual LayerType getLayerType() const = 0;
    };

    struct DenseLayerArgs : public LayerArgs {

        DenseLayerArgs(Dimensions out_dims, ActivationType activation_type, size_t batch_size)
            : LayerArgs(out_dims, activation_type, batch_size) {}

        std::unique_ptr<Layer> createLayer(const size_t layer_id,  const OpenCLSetup& ocl_setup, Dimensions input_dims) const override {
            return std::make_unique<DenseLayer>(layer_id, ocl_setup, input_dims, output_dims, activation_type, batch_size);
        }

        LayerType getLayerType() const override {
            return LayerType::Dense;
        }
    };

    std::unique_ptr<DenseLayerArgs> makeDenseLayerArgs(
        const Dimensions& layer_dimensions,
        ActivationType activation_type = ActivationType::ReLU,
        size_t batch_size = 1
    );

    std::unique_ptr<Layer> loadLayer(const OpenCLSetup& ocl_setup, const H5::Group& layer_group, const size_t batch_size);
}