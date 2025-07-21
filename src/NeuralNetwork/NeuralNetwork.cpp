#include "NeuralNetwork/NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(const OpenCLSetup& ocl_setup, NetworkConfig::NetworkArgs network_args)
    : context(ocl_setup.context), queue(ocl_setup.queue), program(ocl_setup.program), ocl_setup(ocl_setup), 
    input_dims(network_args.initial_input_dims), batch_size(network_args.batch_size), loss_function_type(network_args.loss_function_type) {    
    Dimensions current_input_dims = input_dims;
    if (!network_args.layer_arguments.empty()){
        for (const auto& layer_args : network_args.layer_arguments) {
            layer_args->batch_size = batch_size;
            layers.emplace_back(layer_args->createLayer(layers.size(), ocl_setup, current_input_dims));
            current_input_dims = layers.back()->output_dims;
        }
    }
    optimizer = network_args.optimizer_parameters->createOptimizer(ocl_setup);
}

NeuralNetwork::NeuralNetwork(const OpenCLSetup& ocl_setup, const H5::H5File& file)
: context(ocl_setup.context), queue(ocl_setup.queue), program(ocl_setup.program), ocl_setup(ocl_setup) {
    if (!file.attrExists("input_dims")) {
        throw std::runtime_error("HDF5 file does not contain 'input_dims' attribute.");
    }
    try {
        H5::Attribute attr = file.openAttribute("input_dims");
        H5::DataSpace dataspace = attr.getSpace();

        if (dataspace.getSimpleExtentNdims() != 1) {
            throw std::runtime_error("Expected 1D attribute for input_dims");
        }

        hsize_t dims[1];
        dataspace.getSimpleExtentDims(dims);  // Could crash if dataspace is invalid

        std::vector<hsize_t> input_dims_vector(dims[0]);

        attr.read(H5::PredType::NATIVE_HSIZE, input_dims_vector.data());

        input_dims = Dimensions(input_dims_vector);
    } catch (const H5::Exception& e) {
        std::cerr << "HDF5 error: " << e.getCDetailMsg() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Std error: " << e.what() << std::endl;
    }



    if (file.attrExists("batch_size")) file.openAttribute("batch_size").read(H5::PredType::NATIVE_HSIZE, &batch_size);
    else batch_size = 1;

    if (file.attrExists("loss_function_type")){
        unsigned int loss_function_uint;
        file.openAttribute("loss_function_type").read(H5::PredType::NATIVE_UINT, &loss_function_uint);
        loss_function_type = lossFunctionTypeFromUint(loss_function_uint);
    }
    else loss_function_type = LossFunctionType::MeanSquaredError;
    
    
    H5::Group layers_group = file.openGroup("layers");
    size_t num_layers;
    if (layers_group.attrExists("num_layers")) layers_group.openAttribute("num_layers").read(H5::PredType::NATIVE_HSIZE, &num_layers);
    else num_layers = 0;

    for (size_t i = 0; i < num_layers; ++i) {
        std::string layer_id = std::to_string(i);
        H5::Group layer_group = layers_group.openGroup(layer_id);
        layers.emplace_back(LayerConfig::loadLayer(ocl_setup, layer_group, batch_size));
        if (layers.size() == 1 && layers.back()->input_dims != input_dims) {
            throw std::runtime_error("First layer input dimensions do not match the network's input dimensions.");
            
        }
    }
    
    H5::Group optimizer_group = file.openGroup("optimizer");
    optimizer = OptimizerConfig::loadOptimizer(ocl_setup, optimizer_group);
}

NeuralNetwork& NeuralNetwork::addDense(const size_t output_dims_size_t, ActivationType activation_type) {
    Dimensions output_dims = Dimensions({output_dims_size_t});
    Dimensions input_dims = this->input_dims;
    if (!layers.empty()){
        input_dims = layers.back()->output_dims;
    }
    auto layer_args = LayerConfig::makeDenseLayerArgs(output_dims, activation_type, batch_size);
    layers.emplace_back(layer_args->createLayer(layers.size(), ocl_setup, input_dims));
    
    return *this;
}

int NeuralNetwork::findNextTrainableLayer(int start_idx) const {
    for (int i = start_idx + 1; i < static_cast<int>(layers.size()); ++i) {
        if (layers[i]->isTrainable()) {
            return i;
        }
    }
    return -1;
}

cl::Buffer NeuralNetwork::forward(const cl::Buffer& initial_input_batch, size_t current_batch_actual_size) {
    cl::Buffer current_input_buffer = initial_input_batch;

    for (size_t i = 0; i < layers.size(); ++i) {
        auto& layer = layers[i];
        layer->setBatchSize(current_batch_actual_size);
        layer->runForward(current_input_buffer);
        current_input_buffer = layer->outputs;
    }
    return current_input_buffer;
}

void NeuralNetwork::backprop(const cl::Buffer& original_network_input_batch, const cl::Buffer& target_batch_buf) {
    size_t output_size = layers.back()->output_dims.getTotalElements();

    auto& output_layer = layers.back();

    output_layer->computeOutputDeltas(target_batch_buf, loss_function_type);

    // Backprop deltas
    cl::Buffer* weights_of_next_layer_ptr = nullptr;
    size_t next_layer_output_size; 
    for (int l = static_cast<int>(layers.size()) - 2; l >= 0; --l) {
        auto& current_layer = layers[l];
        auto& next_layer = layers[l + 1];
        next_layer_output_size = next_layer->getTotalOutputElements();
        if (next_layer->isTrainable()) {
            weights_of_next_layer_ptr =  &static_cast<TrainableLayer&>(*next_layer).getWeights();
        }
        else {
            weights_of_next_layer_ptr = nullptr;
        }
        current_layer->backpropDeltas(next_layer->getDeltas(), weights_of_next_layer_ptr, next_layer_output_size);
    }

    // Calculate gradients for trainable layers
    for (int l = static_cast<int>(layers.size()) - 1; l >= 0; --l) {
        auto& layer = layers[l];
        const cl::Buffer& inputs_to_current_layer = (l == 0) ? original_network_input_batch : layers[l - 1]->outputs;
        if (layer->isTrainable()) {
            auto& trainable = static_cast<TrainableLayer&>(*layer);
            trainable.calculateWeightGradients(inputs_to_current_layer);
            trainable.calculateBiasGradients();
        }
    }

    // Update parameters using the optimizer
    for (auto& layer : layers) {
        if (layer->isTrainable()){
            auto& trainable_layer = static_cast<TrainableLayer&>(*layer);
            std::string weights_id = std::to_string(trainable_layer.layer_id) + "_weights";
            std::string biases_id  = std::to_string(trainable_layer.layer_id) + "_biases";
            optimizer->updateParameters(weights_id, trainable_layer.getWeights(), trainable_layer.getWeightGradients(), trainable_layer.getWeightsSize());
            optimizer->updateParameters(biases_id, trainable_layer.getBiases(), trainable_layer.getBiasGradients(), trainable_layer.getBiasesSize());
        }
    }

    optimizer->step();
}

void NeuralNetwork::train(DataProcessor& dataProcessor, int epochs) {

    dataProcessor.activateTrainPartition();

    size_t output_single_size = dataProcessor.getTargetSize();
    std::vector<float> network_output_host(batch_size * output_single_size);
    std::vector<float> current_batch_target_host(batch_size * output_single_size);

    for (int epoch = 0; epoch < epochs; ++epoch) { 
        float total_loss = 0.0f;  
        dataProcessor.shuffleCurrentPartition();
        for (const Batch& batch: dataProcessor){
            cl::Buffer inputs = batch.inputs;
            cl::Buffer targets = batch.targets;
            size_t current_batch_actual_size = batch.batch_actual_size;

            queue.enqueueReadBuffer(targets, CL_TRUE, 0,
                                     sizeof(float) * current_batch_actual_size * output_single_size,
                                     current_batch_target_host.data());

            cl::Buffer final_output_buf = forward(inputs, current_batch_actual_size);
            
            queue.enqueueReadBuffer(final_output_buf, CL_TRUE, 0,
                                    sizeof(float) * current_batch_actual_size * output_single_size,
                                    network_output_host.data());

            for (size_t b = 0; b < current_batch_actual_size; ++b) {
                for (size_t j = 0; j < output_single_size; ++j) {
                    float diff = network_output_host[b * output_single_size + j] -
                                 current_batch_target_host[b * output_single_size + j];
                    total_loss += diff * diff;
                }
            }

            backprop(inputs, targets);
            queue.finish();

        }
        float average_loss = total_loss / (dataProcessor.getActivePartition().size() * output_single_size);
        std::cout << "Epoch " << (epoch + 1) << " | Loss: " << average_loss << std::endl;
    }
}

void NeuralNetwork::saveNetwork(const std::string& filename) const {
    try {
        H5::H5File file(filename, H5F_ACC_TRUNC);
        H5::DataSpace scalar_dataspace(H5S_SCALAR);

        const std::vector<hsize_t>& dims = input_dims.dims;

        hsize_t len = dims.size();
        H5::DataSpace vector_dataspace(1, &len);

        file.createAttribute(
            "input_dims", H5::PredType::NATIVE_HSIZE, vector_dataspace
        ).write(H5::PredType::NATIVE_HSIZE, dims.data());

        file.createAttribute(
            "batch_size", H5::PredType::NATIVE_HSIZE, scalar_dataspace
        ).write(H5::PredType::NATIVE_HSIZE, &batch_size);

        unsigned int loss_type_int = static_cast<unsigned int>(loss_function_type);
        file.createAttribute(
            "loss_function_type", H5::PredType::NATIVE_UINT, scalar_dataspace
        ).write(H5::PredType::NATIVE_UINT, &loss_type_int);

        H5::Group layers_parent_group(file.createGroup("/layers"));
        size_t num_layers = layers.size();
        layers_parent_group.createAttribute(
            "num_layers", H5::PredType::NATIVE_HSIZE, scalar_dataspace
        ).write(H5::PredType::NATIVE_HSIZE, &num_layers);

        std::map<size_t, std::pair<size_t, size_t>> moments_sizes;
        for (size_t i = 0; i < layers.size(); ++i) {
            if (layers[i]) {
                if (layers[i]->isTrainable()) {
                    auto& trainable_layer = static_cast<TrainableLayer&>(*layers[i]);
                    moments_sizes[trainable_layer.layer_id] = {
                        trainable_layer.getWeightsSize(),
                        trainable_layer.getBiasesSize()
                    };
                }
                

                std::string layer_group_name = std::to_string(layers[i]->layer_id);
                H5::Group layer_subgroup(layers_parent_group.createGroup(layer_group_name));

                layer_subgroup.createAttribute(
                    "layer_id", H5::PredType::NATIVE_HSIZE, scalar_dataspace
                ).write(H5::PredType::NATIVE_HSIZE, &layers[i]->layer_id);
                
                unsigned int layer_type = static_cast<unsigned int>(layers[i]->getType());
                layer_subgroup.createAttribute(
                    "layer_type", H5::PredType::NATIVE_UINT, scalar_dataspace
                ).write(H5::PredType::NATIVE_UINT, &layer_type);
                
                layers[i]->saveLayer(layer_subgroup);
            } else {
                std::cerr << "Warning: Layer at index " << i << " is null, skipping its save operation." << std::endl;
            }
        }

        if (optimizer) {
            H5::Group optimizer_group(file.createGroup("/optimizer"));
            unsigned int optimizer_type = static_cast<unsigned int>(optimizer->getType());
            optimizer_group.createAttribute(
                "optimizer_type", H5::PredType::NATIVE_UINT, scalar_dataspace
            ).write(H5::PredType::NATIVE_UINT, &optimizer_type);
            optimizer->saveOptimizer(optimizer_group, moments_sizes);
        } else {
            std::cerr << "Warning: Optimizer is null, skipping its save operation." << std::endl;
        }
        file.close();
        std::cout << "Neural Network successfully saved to " << filename << std::endl;
    } catch (const H5::Exception& error) {
        std::cerr << "HDF5 Exception in NeuralNetwork::saveNetwork: " << error.getFuncName()
                  << " -> " << error.getDetailMsg() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Standard Exception in NeuralNetwork::saveNetwork: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "Unknown Exception in NeuralNetwork::saveNetwork." << std::endl;
    }
}

NeuralNetwork NeuralNetwork::loadNetwork(const OpenCLSetup& ocl_setup, const std::string& file_name) {
    if (!std::filesystem::exists(file_name)) {
        throw std::runtime_error("File does not exist: " + file_name);
    }

    H5::H5File file(file_name, H5F_ACC_RDONLY);
    NeuralNetwork network(ocl_setup, file);
    file.close();
    return network;
}