#include "Layer/Dense/DenseLayer.hpp"

DenseLayer::DenseLayer(const size_t layer_id, const OpenCLSetup& ocl_setup,
                       const Dimensions& input_dims, const Dimensions& output_dims,
                       ActivationType act_type, size_t batch_size): 
                       TrainableLayer(layer_id, ocl_setup, input_dims, output_dims, batch_size),
                        activation_type(act_type){
                        allocatePreActivationBuffer();
                        initializeWeightsAndBiases();
                        }

DenseLayer::DenseLayer(const OpenCLSetup& ocl_setup, const H5::Group& layer_group, const size_t batch_size)
: TrainableLayer(ocl_setup, batch_size) {
    layer_group.openAttribute("layer_id").read(H5::PredType::NATIVE_HSIZE, &layer_id);

    unsigned int input_dims_uint;
    layer_group.openAttribute("input_dims").read(H5::PredType::NATIVE_UINT, &input_dims_uint);
    input_dims = Dimensions({input_dims_uint});

    unsigned int output_dims_uint;
    layer_group.openAttribute("output_dims").read(H5::PredType::NATIVE_UINT, &output_dims_uint);
    output_dims = Dimensions({output_dims_uint});

    unsigned int activation_type_uint;
    layer_group.openAttribute("activation_type").read(H5::PredType::NATIVE_UINT, &activation_type_uint);
    activation_type = activationTypeFromUint(activation_type_uint);

    std::vector<float> loaded_weights(getWeightsSize());
    std::vector<float> loaded_biases(getBiasesSize());

    layer_group.openDataSet("weights").read(loaded_weights.data(), H5::PredType::NATIVE_FLOAT);
    layer_group.openDataSet("biases").read(loaded_biases.data(), H5::PredType::NATIVE_FLOAT);

    size_t flat_input_size = input_dims.getTotalElements();
    size_t flat_output_size = output_dims.getTotalElements();

    weights = cl::Buffer(context, CL_MEM_READ_WRITE, flat_input_size * flat_output_size * sizeof(float));
    biases = cl::Buffer(context, CL_MEM_READ_WRITE, flat_output_size * sizeof(float));
    pre_activations = cl::Buffer(context, CL_MEM_READ_WRITE, batch_size * flat_output_size * sizeof(float));
    outputs = cl::Buffer(context, CL_MEM_READ_WRITE, batch_size * flat_output_size * sizeof(float));
    deltas = cl::Buffer(context, CL_MEM_READ_WRITE, batch_size * flat_output_size * sizeof(float));
    weight_gradients = cl::Buffer(context, CL_MEM_READ_WRITE, flat_input_size * flat_output_size * sizeof(float));
    bias_gradients = cl::Buffer(context, CL_MEM_READ_WRITE, flat_output_size * sizeof(float));

    queue.enqueueWriteBuffer(weights, CL_TRUE, 0, loaded_weights.size() * sizeof(float), loaded_weights.data());
    queue.enqueueWriteBuffer(biases, CL_TRUE, 0, loaded_biases.size() * sizeof(float), loaded_biases.data());
}

void DenseLayer::initializeWeightsAndBiases() {
    size_t flat_input_size = input_dims.getTotalElements();
    size_t flat_output_size = output_dims.getTotalElements();

    std::vector<float> h_weights(flat_input_size * flat_output_size);
    std::vector<float> h_biases(flat_output_size);

    float limit = std::sqrt(6.0f / (flat_input_size + flat_output_size));

    for (auto& weight : h_weights) {
        weight = getRandomWeight(-limit, limit);
    }
    for (auto& bias : h_biases) {
        bias = getRandomWeight(-0.5f, 0.5f);
    }

    queue.enqueueWriteBuffer(weights, CL_TRUE, 0, h_weights.size() * sizeof(float), h_weights.data());
    queue.enqueueWriteBuffer(biases, CL_TRUE, 0, h_biases.size() * sizeof(float), h_biases.data());
}

void DenseLayer::runForward(const cl::Buffer& input_buffer) {
    size_t flat_input_size = input_dims.getTotalElements();
    size_t flat_output_size = output_dims.getTotalElements();

    cl::Kernel kernel(program, "dense_forward_batch");

    kernel.setArg(0, weights);
    kernel.setArg(1, biases);
    kernel.setArg(2, input_buffer);
    kernel.setArg(3, outputs);
    kernel.setArg(4, pre_activations);
    kernel.setArg(5, (cl_uint)flat_input_size);
    kernel.setArg(6, (cl_uint)flat_output_size);
    kernel.setArg(7, (cl_uint)batch_size);
    kernel.setArg(8, (cl_uint)activation_type);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(flat_output_size, batch_size), cl::NullRange);
}

void DenseLayer::computeOutputDeltas(const cl::Buffer& true_labels_buffer,
                                     const LossFunctionType& loss_function_type) {
    size_t flat_output_size = output_dims.getTotalElements();

    cl::Kernel kernel(program, "dense_compute_output_deltas_batch");

    kernel.setArg(0, pre_activations);
    kernel.setArg(1, outputs);
    kernel.setArg(2, true_labels_buffer);
    kernel.setArg(3, deltas);
    kernel.setArg(4, (cl_uint)flat_output_size);
    kernel.setArg(5, (cl_uint)batch_size);
    kernel.setArg(6, (cl_uint)activation_type);
    kernel.setArg(7, (cl_uint)loss_function_type);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(flat_output_size, batch_size), cl::NullRange);
}

void DenseLayer:: backpropDeltas(const cl::Buffer& next_layer_deltas,
                                 const cl::Buffer* next_layer_weights_ptr,
                                 const size_t next_layer_output_size) {

    size_t flat_output_size = output_dims.getTotalElements();
    cl::Kernel kernel;
    if (next_layer_weights_ptr != nullptr) {
        kernel = cl::Kernel(program, "dense_backprop_deltas_with_weights_batch");
        kernel.setArg(0, *next_layer_weights_ptr);
        kernel.setArg(1, next_layer_deltas);
        kernel.setArg(2, deltas);
        kernel.setArg(3, pre_activations);
        kernel.setArg(4, (cl_uint)flat_output_size);
        kernel.setArg(5, (cl_uint)(next_layer_output_size));
        kernel.setArg(6, (cl_uint)batch_size);
        kernel.setArg(7, (cl_uint)activation_type);
    }
    else {
        kernel = cl::Kernel(program, "dense_backprop_deltas_no_weights_batch");
        kernel.setArg(0, next_layer_deltas);
        kernel.setArg(1, deltas);
        kernel.setArg(2, pre_activations);
        kernel.setArg(3, (cl_uint)flat_output_size);
        kernel.setArg(4, (cl_uint)batch_size);
        kernel.setArg(5, (cl_uint)activation_type);
    }
    

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(flat_output_size, batch_size), cl::NullRange);
}

void DenseLayer::calculateWeightGradients(const cl::Buffer& inputs_to_current_layer) {
    size_t flat_input_size = input_dims.getTotalElements();
    size_t flat_output_size = output_dims.getTotalElements();

    cl::Kernel kernel(program, "dense_calculate_weight_gradients_batch");

    kernel.setArg(0, inputs_to_current_layer);
    kernel.setArg(1, deltas);
    kernel.setArg(2, weight_gradients);
    kernel.setArg(3, (cl_uint)flat_input_size);
    kernel.setArg(4, (cl_uint)flat_output_size);
    kernel.setArg(5, (cl_uint)batch_size);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(flat_output_size, flat_input_size), cl::NullRange);
}

void DenseLayer::calculateBiasGradients() {
    size_t flat_output_size = output_dims.getTotalElements();

    cl::Kernel kernel(program, "dense_calculate_bias_gradients_batch");

    kernel.setArg(0, deltas);
    kernel.setArg(1, bias_gradients);
    kernel.setArg(2, (cl_uint)flat_output_size);
    kernel.setArg(3, (cl_uint)batch_size);

    queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                               cl::NDRange(flat_output_size), cl::NullRange);
}

void DenseLayer::saveLayer(H5::Group& layer_group) const {
    H5::DataSpace scalar_space(H5S_SCALAR);

    size_t input_dims_size = input_dims.getTotalElements();
    layer_group.createAttribute(
        "input_dims", H5::PredType::NATIVE_HSIZE, scalar_space
    ).write(H5::PredType::NATIVE_HSIZE, &input_dims_size);

    size_t output_dims_size = output_dims.getTotalElements();
    layer_group.createAttribute(
        "output_dims", H5::PredType::NATIVE_HSIZE, scalar_space
    ).write(H5::PredType::NATIVE_HSIZE, &output_dims_size);

    unsigned int activation_type_int = static_cast<unsigned int>(activation_type);
    layer_group.createAttribute(
        "activation_type", H5::PredType::NATIVE_UINT, scalar_space
    ).write(H5::PredType::NATIVE_UINT, &activation_type_int);
    
    saveBuffer(weights, layer_group, "weights", getWeightsSize());
    saveBuffer(biases, layer_group, "biases", getBiasesSize());   
}