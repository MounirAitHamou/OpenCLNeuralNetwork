#include "clnn/clnn.hpp"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;

clnn::Shape array_shape(const py::buffer_info& information) {
    clnn::Shape shape;
    shape.reserve(static_cast<std::size_t>(information.ndim));
    for (const auto dimension : information.shape) {
        if (dimension < 0)
            throw std::invalid_argument("array dimensions cannot be negative");
        shape.push_back(static_cast<std::size_t>(dimension));
    }
    return shape;
}

std::vector<float> array_values(const FloatArray& array) {
    const auto information = array.request();
    const auto* first = static_cast<const float*>(information.ptr);
    return {first, first + information.size};
}

clnn::Tensor tensor_from_array(const FloatArray& array, const bool requires_grad, std::string name,
                               const clnn::Device device) {
    const auto information = array.request();
    return clnn::Tensor(array_values(array), array_shape(information), requires_grad,
                        std::move(name), device);
}

py::array_t<float> tensor_array(const clnn::Tensor& tensor) {
    std::vector<py::ssize_t> shape;
    shape.reserve(tensor.ndim());
    for (const auto dimension : tensor.shape()) {
        shape.push_back(static_cast<py::ssize_t>(dimension));
    }
    py::array_t<float> result(shape);
    const auto& values = tensor.data();
    std::memcpy(result.mutable_data(), values.data(), values.size() * sizeof(float));
    return result;
}

py::object tensor_gradient(const clnn::Tensor& tensor) {
    if (!tensor.has_grad())
        return py::none();
    std::vector<py::ssize_t> shape;
    for (const auto dimension : tensor.shape()) {
        shape.push_back(static_cast<py::ssize_t>(dimension));
    }
    py::array_t<float> result(shape);
    const auto& gradient = tensor.grad();
    std::memcpy(result.mutable_data(), gradient.data(), gradient.size() * sizeof(float));
    return result;
}

py::tuple shape_tuple(const clnn::Shape& shape) {
    py::tuple result(shape.size());
    for (std::size_t index = 0; index < shape.size(); ++index) {
        result[index] = py::int_(shape[index]);
    }
    return result;
}

std::array<std::size_t, 2> pair_from_python(const std::array<std::size_t, 2>& value) {
    if (value[0] == 0 || value[1] == 0)
        throw std::invalid_argument("values must be positive");
    return value;
}

std::unique_ptr<clnn::nn::Linear> make_linear(const std::size_t input_features,
                                              const std::size_t output_features, const bool bias,
                                              const std::uint32_t seed, const clnn::Device device) {
    std::mt19937 generator(seed);
    return std::make_unique<clnn::nn::Linear>(input_features, output_features, generator, bias,
                                              device);
}

std::unique_ptr<clnn::nn::Conv2d>
make_convolution(const std::size_t input_channels, const std::size_t output_channels,
                 const std::array<std::size_t, 2> kernel_size,
                 const std::array<std::size_t, 2> stride, const py::object& padding,
                 const bool bias, const std::uint32_t seed, const clnn::Device device) {
    std::mt19937 generator(seed);
    const auto kernel = pair_from_python(kernel_size);
    const auto strides = pair_from_python(stride);
    if (py::isinstance<py::str>(padding)) {
        const auto mode = padding.cast<std::string>();
        if (mode == "same") {
            return std::make_unique<clnn::nn::Conv2d>(input_channels, output_channels, kernel,
                                                      generator, strides,
                                                      clnn::nn::PaddingMode::same, bias, device);
        }
        if (mode == "valid") {
            return std::make_unique<clnn::nn::Conv2d>(input_channels, output_channels, kernel,
                                                      generator, strides,
                                                      clnn::nn::PaddingMode::valid, bias, device);
        }
        throw std::invalid_argument("padding must be 'same', 'valid', or a pair");
    }
    return std::make_unique<clnn::nn::Conv2d>(input_channels, output_channels, kernel, generator,
                                              strides, padding.cast<std::array<std::size_t, 2>>(),
                                              bias, device);
}

class PythonNoGrad final {
  public:
    PythonNoGrad& enter() {
        if (guard_)
            throw std::logic_error("no_grad context cannot be entered twice");
        guard_ = std::make_unique<clnn::NoGradGuard>();
        return *this;
    }
    void exit(const py::object&, const py::object&, const py::object&) {
        guard_.reset();
    }

  private:
    std::unique_ptr<clnn::NoGradGuard> guard_;
};

clnn::data::TensorDataset
dataset_from_arrays(const FloatArray& inputs, const FloatArray& targets,
                    const std::optional<std::vector<std::size_t>>& labels) {
    const auto input_information = inputs.request();
    const auto target_information = targets.request();
    if (input_information.ndim < 1 || target_information.ndim < 1 ||
        input_information.shape.front() != target_information.shape.front()) {
        throw std::invalid_argument("inputs and targets require the same leading sample dimension");
    }
    auto input_shape = array_shape(input_information);
    auto target_shape = array_shape(target_information);
    input_shape.erase(input_shape.begin());
    target_shape.erase(target_shape.begin());
    if (input_shape.empty())
        input_shape.push_back(1);
    if (target_shape.empty())
        target_shape.push_back(1);
    return clnn::data::TensorDataset(array_values(inputs), std::move(input_shape),
                                     array_values(targets), std::move(target_shape),
                                     labels.value_or(std::vector<std::size_t>{}));
}

} // namespace

PYBIND11_MODULE(_clnn, module) {
    module.doc() = "OpenCL-accelerated dynamic autograd for C++ and Python";
    module.attr("__version__") = "2.0.0";

    py::enum_<clnn::DeviceType>(module, "DeviceType")
        .value("CPU", clnn::DeviceType::cpu)
        .value("OPENCL", clnn::DeviceType::opencl);

    py::class_<clnn::KernelProfile>(module, "KernelProfile")
        .def_readonly("operation", &clnn::KernelProfile::operation)
        .def_readonly("calls", &clnn::KernelProfile::calls)
        .def_readonly("total_milliseconds", &clnn::KernelProfile::total_milliseconds)
        .def_readonly("minimum_milliseconds", &clnn::KernelProfile::minimum_milliseconds)
        .def_readonly("maximum_milliseconds", &clnn::KernelProfile::maximum_milliseconds);
    py::class_<clnn::OpenCLRuntimeStatistics>(module, "OpenCLRuntimeStatistics")
        .def_readonly("buffer_allocations", &clnn::OpenCLRuntimeStatistics::buffer_allocations)
        .def_readonly("buffer_reuses", &clnn::OpenCLRuntimeStatistics::buffer_reuses)
        .def_readonly("pooled_buffers", &clnn::OpenCLRuntimeStatistics::pooled_buffers)
        .def_readonly("cached_kernels", &clnn::OpenCLRuntimeStatistics::cached_kernels);

    py::class_<clnn::Device>(module, "Device")
        .def_static("cpu", &clnn::Device::cpu)
        .def_static("opencl", &clnn::Device::opencl, py::arg("platform_index") = 0,
                    py::arg("device_index") = 0)
        .def_property_readonly("type", &clnn::Device::type)
        .def_property_readonly("platform_index", &clnn::Device::platform_index)
        .def_property_readonly("device_index", &clnn::Device::device_index)
        .def_property_readonly("name", &clnn::Device::name)
        .def("synchronize", &clnn::Device::synchronize, py::call_guard<py::gil_scoped_release>())
        .def("set_profiling", &clnn::Device::set_profiling, py::arg("enabled"))
        .def("profile", &clnn::Device::profile, py::arg("reset") = true,
             py::call_guard<py::gil_scoped_release>())
        .def("runtime_statistics", &clnn::Device::runtime_statistics)
        .def("clear_memory_pool", &clnn::Device::clear_memory_pool,
             py::call_guard<py::gil_scoped_release>())
        .def("__repr__", [](const clnn::Device& self) { return "Device('" + self.name() + "')"; })
        .def(py::self == py::self);

    module.def("opencl_available", &clnn::opencl_available);
    module.def("grad_enabled", &clnn::grad_enabled);
    module.def("numel", &clnn::numel, py::arg("shape"));
    module.def("shape_string", &clnn::shape_string, py::arg("shape"));

    py::class_<PythonNoGrad>(module, "_NoGrad")
        .def(py::init<>())
        .def("__enter__", &PythonNoGrad::enter, py::return_value_policy::reference_internal)
        .def("__exit__", &PythonNoGrad::exit);
    module.def("no_grad", [] { return PythonNoGrad{}; });

    py::class_<clnn::Tensor>(module, "Tensor")
        .def(py::init(&tensor_from_array), py::arg("data"), py::arg("requires_grad") = false,
             py::arg("name") = "", py::arg("device") = clnn::Device::cpu())
        .def_static("scalar", &clnn::Tensor::scalar, py::arg("value"),
                    py::arg("requires_grad") = false, py::arg("name") = "",
                    py::arg("device") = clnn::Device::cpu())
        .def_static("zeros", &clnn::Tensor::zeros, py::arg("shape"),
                    py::arg("requires_grad") = false, py::arg("name") = "",
                    py::arg("device") = clnn::Device::cpu())
        .def_static("ones", &clnn::Tensor::ones, py::arg("shape"), py::arg("requires_grad") = false,
                    py::arg("name") = "", py::arg("device") = clnn::Device::cpu())
        .def_static(
            "randn",
            [](const clnn::Shape& shape, const float mean, const float standard_deviation,
               const bool requires_grad, const std::uint32_t seed, std::string name,
               const clnn::Device device) {
                std::mt19937 generator(seed);
                return clnn::Tensor::randn(shape, generator, mean, standard_deviation,
                                           requires_grad, std::move(name), device);
            },
            py::arg("shape"), py::arg("mean") = 0.0F, py::arg("standard_deviation") = 1.0F,
            py::arg("requires_grad") = false, py::arg("seed") = 0, py::arg("name") = "",
            py::arg("device") = clnn::Device::cpu())
        .def_property_readonly("shape",
                               [](const clnn::Tensor& self) { return shape_tuple(self.shape()); })
        .def_property_readonly("defined", &clnn::Tensor::defined)
        .def_property_readonly("is_scalar", &clnn::Tensor::is_scalar)
        .def_property_readonly("ndim", &clnn::Tensor::ndim)
        .def_property_readonly("size", &clnn::Tensor::size)
        .def_property_readonly("requires_grad", &clnn::Tensor::requires_grad)
        .def_property_readonly("is_leaf", &clnn::Tensor::is_leaf)
        .def_property_readonly("name", &clnn::Tensor::name)
        .def_property_readonly("device", &clnn::Tensor::device)
        .def_property_readonly("grad", &tensor_gradient)
        .def("numpy", &tensor_array)
        .def("item", &clnn::Tensor::item)
        .def("backward", &clnn::Tensor::backward, py::arg("gradient") = py::none(),
             py::arg("retain_graph") = false, py::call_guard<py::gil_scoped_release>())
        .def("zero_grad", &clnn::Tensor::zero_grad)
        .def("set_data", [](clnn::Tensor& self,
                            const FloatArray& values) { self.set_data(array_values(values)); })
        .def("detach", &clnn::Tensor::detach)
        .def("to", &clnn::Tensor::to, py::arg("device"))
        .def("reshape", &clnn::reshape, py::arg("shape"))
        .def("transpose", &clnn::transpose)
        .def("sum", py::overload_cast<const clnn::Tensor&>(&clnn::sum))
        .def("sum", py::overload_cast<const clnn::Tensor&, std::size_t, bool>(&clnn::sum),
             py::arg("dimension"), py::arg("keep_dimensions") = false)
        .def("mean", py::overload_cast<const clnn::Tensor&>(&clnn::mean))
        .def("mean", py::overload_cast<const clnn::Tensor&, std::size_t, bool>(&clnn::mean),
             py::arg("dimension"), py::arg("keep_dimensions") = false)
        .def("__repr__", [](const clnn::Tensor& self) { return clnn::inspect(self); })
        .def("__len__",
             [](const clnn::Tensor& self) {
                 if (self.ndim() == 0)
                     throw py::type_error("len() of a scalar tensor");
                 return self.shape().front();
             })
        .def("__add__", py::overload_cast<const clnn::Tensor&, const clnn::Tensor&>(&clnn::add),
             py::is_operator())
        .def(
            "__add__", [](const clnn::Tensor& self, const float value) { return self + value; },
            py::is_operator())
        .def(
            "__radd__", [](const clnn::Tensor& self, const float value) { return value + self; },
            py::is_operator())
        .def("__sub__",
             py::overload_cast<const clnn::Tensor&, const clnn::Tensor&>(&clnn::subtract),
             py::is_operator())
        .def(
            "__sub__", [](const clnn::Tensor& self, const float value) { return self - value; },
            py::is_operator())
        .def(
            "__rsub__", [](const clnn::Tensor& self, const float value) { return value - self; },
            py::is_operator())
        .def("__mul__",
             py::overload_cast<const clnn::Tensor&, const clnn::Tensor&>(&clnn::multiply),
             py::is_operator())
        .def(
            "__mul__", [](const clnn::Tensor& self, const float value) { return self * value; },
            py::is_operator())
        .def(
            "__rmul__", [](const clnn::Tensor& self, const float value) { return value * self; },
            py::is_operator())
        .def("__truediv__",
             py::overload_cast<const clnn::Tensor&, const clnn::Tensor&>(&clnn::divide),
             py::is_operator())
        .def(
            "__truediv__", [](const clnn::Tensor& self, const float value) { return self / value; },
            py::is_operator())
        .def(
            "__rtruediv__",
            [](const clnn::Tensor& self, const float value) { return value / self; },
            py::is_operator())
        .def("__neg__", &clnn::negate, py::is_operator());

    module.def("matmul", &clnn::matmul, py::call_guard<py::gil_scoped_release>());
    module.def("conv2d", &clnn::conv2d, py::arg("input"), py::arg("weight"),
               py::arg("bias") = std::nullopt, py::arg("stride") = std::array<std::size_t, 2>{1, 1},
               py::arg("padding") = std::array<std::size_t, 2>{0, 0},
               py::call_guard<py::gil_scoped_release>());
    module.def("max_pool2d", &clnn::max_pool2d, py::arg("input"), py::arg("kernel_size"),
               py::arg("stride") = std::array<std::size_t, 2>{0, 0},
               py::arg("padding") = std::array<std::size_t, 2>{0, 0},
               py::call_guard<py::gil_scoped_release>());
    module.def("avg_pool2d", &clnn::avg_pool2d, py::arg("input"), py::arg("kernel_size"),
               py::arg("stride") = std::array<std::size_t, 2>{0, 0},
               py::arg("padding") = std::array<std::size_t, 2>{0, 0},
               py::call_guard<py::gil_scoped_release>());
    module.def("sum", py::overload_cast<const clnn::Tensor&>(&clnn::sum),
               py::call_guard<py::gil_scoped_release>());
    module.def("sum", py::overload_cast<const clnn::Tensor&, std::size_t, bool>(&clnn::sum),
               py::arg("input"), py::arg("dimension"), py::arg("keep_dimensions") = false,
               py::call_guard<py::gil_scoped_release>());
    module.def("mean", py::overload_cast<const clnn::Tensor&>(&clnn::mean),
               py::call_guard<py::gil_scoped_release>());
    module.def("mean", py::overload_cast<const clnn::Tensor&, std::size_t, bool>(&clnn::mean),
               py::arg("input"), py::arg("dimension"), py::arg("keep_dimensions") = false,
               py::call_guard<py::gil_scoped_release>());
    module.def("pow", &clnn::pow, py::arg("input"), py::arg("exponent"));
    module.def("exp", &clnn::exp);
    module.def("log", &clnn::log);
    module.def("relu", &clnn::relu);
    module.def("leaky_relu", &clnn::leaky_relu, py::arg("input"),
               py::arg("negative_slope") = 0.01F);
    module.def("sigmoid", &clnn::sigmoid);
    module.def("tanh", &clnn::tanh);
    module.def("softmax", &clnn::softmax, py::arg("input"), py::arg("dimension"));
    module.def("reshape", &clnn::reshape);
    module.def("transpose", &clnn::transpose);
    module.def("mse_loss", &clnn::mse_loss);
    module.def("binary_cross_entropy", &clnn::binary_cross_entropy, py::arg("prediction"),
               py::arg("target"), py::arg("epsilon") = 1.0e-7F);
    module.def("binary_cross_entropy_with_logits", &clnn::binary_cross_entropy_with_logits);
    module.def("cross_entropy", &clnn::cross_entropy);

    py::class_<clnn::TensorStatistics>(module, "TensorStatistics")
        .def_readonly("minimum", &clnn::TensorStatistics::minimum)
        .def_readonly("maximum", &clnn::TensorStatistics::maximum)
        .def_readonly("mean", &clnn::TensorStatistics::mean)
        .def_readonly("standard_deviation", &clnn::TensorStatistics::standard_deviation)
        .def_readonly("l2_norm", &clnn::TensorStatistics::l2_norm)
        .def_readonly("non_finite", &clnn::TensorStatistics::non_finite);
    module.def("statistics", &clnn::statistics);
    module.def("inspect", &clnn::inspect, py::arg("tensor"), py::arg("maximum_values") = 12);

    auto neural_network = module.def_submodule("nn");
    py::enum_<clnn::nn::ModuleType>(neural_network, "ModuleType")
        .value("CUSTOM", clnn::nn::ModuleType::custom)
        .value("LINEAR", clnn::nn::ModuleType::linear)
        .value("CONVOLUTION_2D", clnn::nn::ModuleType::convolution_2d)
        .value("RELU", clnn::nn::ModuleType::relu)
        .value("SIGMOID", clnn::nn::ModuleType::sigmoid)
        .value("TANH", clnn::nn::ModuleType::tanh)
        .value("FLATTEN", clnn::nn::ModuleType::flatten)
        .value("LEAKY_RELU", clnn::nn::ModuleType::leaky_relu)
        .value("SOFTMAX", clnn::nn::ModuleType::softmax)
        .value("MAX_POOL_2D", clnn::nn::ModuleType::max_pool_2d)
        .value("AVERAGE_POOL_2D", clnn::nn::ModuleType::average_pool_2d)
        .value("DROPOUT", clnn::nn::ModuleType::dropout)
        .value("BATCH_NORMALIZATION", clnn::nn::ModuleType::batch_normalization)
        .value("LAYER_NORMALIZATION", clnn::nn::ModuleType::layer_normalization)
        .value("GLOBAL_AVERAGE_POOL_2D", clnn::nn::ModuleType::global_average_pool_2d)
        .value("RESIDUAL", clnn::nn::ModuleType::residual);
    py::enum_<clnn::nn::PaddingMode>(neural_network, "PaddingMode")
        .value("VALID", clnn::nn::PaddingMode::valid)
        .value("SAME", clnn::nn::PaddingMode::same);
    py::class_<clnn::nn::Module, py::smart_holder>(neural_network, "Module")
        .def("forward", &clnn::nn::Module::forward, py::call_guard<py::gil_scoped_release>())
        .def("__call__", &clnn::nn::Module::operator(),
             py::call_guard<py::gil_scoped_release>())
        .def("parameters",
             [](clnn::nn::Module& self) {
                 std::vector<clnn::Tensor> parameters;
                 for (const auto parameter : self.parameters())
                     parameters.push_back(*parameter);
                 return parameters;
             })
        .def("named_parameters",
             [](clnn::nn::Module& self) {
                 py::dict parameters;
                 for (const auto& parameter : self.named_parameters()) {
                     parameters[py::str(parameter.name)] = py::cast(*parameter.tensor);
                 }
                 return parameters;
             })
        .def("named_buffers",
             [](clnn::nn::Module& self) {
                 py::dict buffers;
                 for (const auto& buffer : self.named_buffers()) {
                     buffers[py::str(buffer.name)] = py::cast(*buffer.tensor);
                 }
                 return buffers;
             })
        .def_property_readonly("type", &clnn::nn::Module::type)
        .def_property_readonly("training", &clnn::nn::Module::training)
        .def("to", &clnn::nn::Module::to, py::return_value_policy::reference_internal)
        .def("train", &clnn::nn::Module::train, py::arg("training") = true,
             py::return_value_policy::reference_internal)
        .def("eval", &clnn::nn::Module::eval, py::return_value_policy::reference_internal)
        .def(
            "register_forward_hook",
            [](clnn::nn::Module& self, py::function callback) {
                auto callback_holder =
                    std::make_shared<py::function>(std::move(callback));
                return self.register_forward_hook(
                    [callback_holder = std::move(callback_holder)](
                        clnn::nn::Module& source, const clnn::Tensor& input,
                        const clnn::Tensor& output) {
                        py::gil_scoped_acquire acquire;
                        (*callback_holder)(
                            py::cast(&source, py::return_value_policy::reference), input, output);
                    });
            },
            py::arg("hook"));
    py::class_<clnn::nn::ForwardHookHandle>(neural_network, "ForwardHookHandle")
        .def("remove", &clnn::nn::ForwardHookHandle::remove)
        .def_property_readonly("active", &clnn::nn::ForwardHookHandle::active);
    py::class_<clnn::nn::Linear, clnn::nn::Module, py::smart_holder>(neural_network, "Linear")
        .def(py::init(&make_linear), py::arg("input_features"), py::arg("output_features"),
             py::arg("bias") = true, py::arg("seed") = 0, py::arg("device") = clnn::Device::cpu())
        .def_property_readonly("weight", &clnn::nn::Linear::weight,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("bias", &clnn::nn::Linear::bias,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("input_features", &clnn::nn::Linear::input_features)
        .def_property_readonly("output_features", &clnn::nn::Linear::output_features)
        .def_property_readonly("has_bias", &clnn::nn::Linear::has_bias);
    py::class_<clnn::nn::Conv2d, clnn::nn::Module, py::smart_holder>(neural_network, "Conv2d")
        .def(py::init(&make_convolution), py::arg("input_channels"), py::arg("output_channels"),
             py::arg("kernel_size"), py::arg("stride") = std::array<std::size_t, 2>{1, 1},
             py::arg("padding") = py::str("valid"), py::arg("bias") = true, py::arg("seed") = 0,
             py::arg("device") = clnn::Device::cpu())
        .def_property_readonly("weight", &clnn::nn::Conv2d::weight,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("bias", &clnn::nn::Conv2d::bias,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("input_channels", &clnn::nn::Conv2d::input_channels)
        .def_property_readonly("output_channels", &clnn::nn::Conv2d::output_channels)
        .def_property_readonly("kernel_size", &clnn::nn::Conv2d::kernel_size)
        .def_property_readonly("stride", &clnn::nn::Conv2d::stride)
        .def_property_readonly("padding", &clnn::nn::Conv2d::padding)
        .def_property_readonly("has_bias", &clnn::nn::Conv2d::has_bias);
    py::class_<clnn::nn::ReLU, clnn::nn::Module, py::smart_holder>(neural_network, "ReLU")
        .def(py::init<>());
    py::class_<clnn::nn::Sigmoid, clnn::nn::Module, py::smart_holder>(neural_network, "Sigmoid")
        .def(py::init<>());
    py::class_<clnn::nn::Tanh, clnn::nn::Module, py::smart_holder>(neural_network, "Tanh")
        .def(py::init<>());
    py::class_<clnn::nn::LeakyReLU, clnn::nn::Module, py::smart_holder>(neural_network, "LeakyReLU")
        .def(py::init<float>(), py::arg("negative_slope") = 0.01F)
        .def_property_readonly("negative_slope", &clnn::nn::LeakyReLU::negative_slope);
    py::class_<clnn::nn::Softmax, clnn::nn::Module, py::smart_holder>(neural_network, "Softmax")
        .def(py::init<std::size_t>(), py::arg("dimension") = 1)
        .def_property_readonly("dimension", &clnn::nn::Softmax::dimension);
    py::class_<clnn::nn::Flatten, clnn::nn::Module, py::smart_holder>(neural_network, "Flatten")
        .def(py::init<>());
    py::class_<clnn::nn::MaxPool2d, clnn::nn::Module, py::smart_holder>(neural_network, "MaxPool2d")
        .def(py::init<std::array<std::size_t, 2>, std::array<std::size_t, 2>,
                      std::array<std::size_t, 2>>(),
             py::arg("kernel_size"), py::arg("stride") = std::array<std::size_t, 2>{0, 0},
             py::arg("padding") = std::array<std::size_t, 2>{0, 0})
        .def_property_readonly("kernel_size", &clnn::nn::MaxPool2d::kernel_size)
        .def_property_readonly("stride", &clnn::nn::MaxPool2d::stride)
        .def_property_readonly("padding", &clnn::nn::MaxPool2d::padding);
    py::class_<clnn::nn::AvgPool2d, clnn::nn::Module, py::smart_holder>(neural_network, "AvgPool2d")
        .def(py::init<std::array<std::size_t, 2>, std::array<std::size_t, 2>,
                      std::array<std::size_t, 2>>(),
             py::arg("kernel_size"), py::arg("stride") = std::array<std::size_t, 2>{0, 0},
             py::arg("padding") = std::array<std::size_t, 2>{0, 0})
        .def_property_readonly("kernel_size", &clnn::nn::AvgPool2d::kernel_size)
        .def_property_readonly("stride", &clnn::nn::AvgPool2d::stride)
        .def_property_readonly("padding", &clnn::nn::AvgPool2d::padding);
    py::class_<clnn::nn::Dropout, clnn::nn::Module, py::smart_holder>(neural_network, "Dropout")
        .def(py::init<float, std::uint32_t>(), py::arg("probability") = 0.5F, py::arg("seed") = 0)
        .def_property_readonly("probability", &clnn::nn::Dropout::probability);
    py::class_<clnn::nn::BatchNorm, clnn::nn::Module, py::smart_holder>(neural_network, "BatchNorm")
        .def(py::init<std::size_t, float, float, bool, clnn::Device>(), py::arg("features"),
             py::arg("epsilon") = 1.0e-5F, py::arg("momentum") = 0.1F, py::arg("affine") = true,
             py::arg("device") = clnn::Device::cpu())
        .def_property_readonly("features", &clnn::nn::BatchNorm::features)
        .def_property_readonly("epsilon", &clnn::nn::BatchNorm::epsilon)
        .def_property_readonly("momentum", &clnn::nn::BatchNorm::momentum)
        .def_property_readonly("affine", &clnn::nn::BatchNorm::affine)
        .def_property_readonly("weight", &clnn::nn::BatchNorm::weight,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("bias", &clnn::nn::BatchNorm::bias,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("running_mean", &clnn::nn::BatchNorm::running_mean,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("running_variance", &clnn::nn::BatchNorm::running_variance,
                               py::return_value_policy::reference_internal);
    py::class_<clnn::nn::LayerNorm, clnn::nn::Module, py::smart_holder>(neural_network, "LayerNorm")
        .def(py::init<clnn::Shape, float, bool, clnn::Device>(), py::arg("normalized_shape"),
             py::arg("epsilon") = 1.0e-5F, py::arg("affine") = true,
             py::arg("device") = clnn::Device::cpu())
        .def_property_readonly("normalized_shape", &clnn::nn::LayerNorm::normalized_shape)
        .def_property_readonly("epsilon", &clnn::nn::LayerNorm::epsilon)
        .def_property_readonly("affine", &clnn::nn::LayerNorm::affine)
        .def_property_readonly("weight", &clnn::nn::LayerNorm::weight,
                               py::return_value_policy::reference_internal)
        .def_property_readonly("bias", &clnn::nn::LayerNorm::bias,
                               py::return_value_policy::reference_internal);
    py::class_<clnn::nn::GlobalAvgPool2d, clnn::nn::Module, py::smart_holder>(neural_network,
                                                                              "GlobalAvgPool2d")
        .def(py::init<>());
    py::class_<clnn::nn::Residual, clnn::nn::Module, py::smart_holder>(neural_network, "Residual")
        .def(py::init<std::unique_ptr<clnn::nn::Module>>(), py::arg("module"))
        .def_property_readonly("module", &clnn::nn::Residual::module,
                               py::return_value_policy::reference_internal);
    py::class_<clnn::nn::Sequential, clnn::nn::Module, py::smart_holder>(neural_network,
                                                                         "Sequential")
        .def(py::init<>())
        .def("add", &clnn::nn::Sequential::add, py::arg("module"),
             py::return_value_policy::reference_internal)
        .def("__len__", &clnn::nn::Sequential::size)
        .def(
            "__getitem__",
            [](clnn::nn::Sequential& self, py::ssize_t index) -> clnn::nn::Module& {
                const auto size = static_cast<py::ssize_t>(self.size());
                if (index < 0)
                    index += size;
                if (index < 0 || index >= size)
                    throw py::index_error();
                return *self.modules()[static_cast<std::size_t>(index)];
            },
            py::return_value_policy::reference_internal);
    neural_network.def("save_state_dict", &clnn::nn::save_state_dict);
    neural_network.def("load_state_dict", &clnn::nn::load_state_dict);
    neural_network.def("save_checkpoint", &clnn::nn::save_checkpoint);
    neural_network.def("load_checkpoint", &clnn::nn::load_checkpoint, py::arg("path"),
                       py::arg("device") = clnn::Device::cpu(), py::arg("initialization_seed") = 0);

    auto explanation = module.def_submodule("explain");
    py::class_<clnn::explain::Target>(explanation, "Target")
        .def(py::init<std::size_t, std::optional<std::size_t>>(), py::arg("index"),
             py::arg("batch_index") = std::nullopt)
        .def_readonly("index", &clnn::explain::Target::index)
        .def_readonly("batch_index", &clnn::explain::Target::batch_index);
    explanation.def("saliency", &clnn::explain::saliency, py::arg("model"), py::arg("input"),
                    py::arg("target"), py::call_guard<py::gil_scoped_release>());
    explanation.def(
        "saliency",
        [](clnn::nn::Module& model, const clnn::Tensor& input, const std::size_t target,
           const std::optional<std::size_t> batch_index) {
            return clnn::explain::saliency(model, input, {target, batch_index});
        },
        py::arg("model"), py::arg("input"), py::arg("target"),
        py::arg("batch_index") = std::nullopt, py::call_guard<py::gil_scoped_release>());
    explanation.def("input_x_gradient", &clnn::explain::input_x_gradient, py::arg("model"),
                    py::arg("input"), py::arg("target"),
                    py::call_guard<py::gil_scoped_release>());
    explanation.def(
        "input_x_gradient",
        [](clnn::nn::Module& model, const clnn::Tensor& input, const std::size_t target,
           const std::optional<std::size_t> batch_index) {
            return clnn::explain::input_x_gradient(model, input, {target, batch_index});
        },
        py::arg("model"), py::arg("input"), py::arg("target"),
        py::arg("batch_index") = std::nullopt, py::call_guard<py::gil_scoped_release>());
    explanation.def("integrated_gradients", &clnn::explain::integrated_gradients,
                    py::arg("model"), py::arg("input"), py::arg("target"),
                    py::arg("baseline"), py::arg("steps") = 50,
                    py::call_guard<py::gil_scoped_release>());
    explanation.def(
        "integrated_gradients",
        [](clnn::nn::Module& model, const clnn::Tensor& input, const std::size_t target,
           const clnn::Tensor& baseline, const std::size_t steps,
           const std::optional<std::size_t> batch_index) {
            return clnn::explain::integrated_gradients(model, input, {target, batch_index},
                                                       baseline, steps);
        },
        py::arg("model"), py::arg("input"), py::arg("target"), py::arg("baseline"),
        py::arg("steps") = 50, py::arg("batch_index") = std::nullopt,
        py::call_guard<py::gil_scoped_release>());
    explanation.def("smoothgrad", &clnn::explain::smoothgrad, py::arg("model"),
                    py::arg("input"), py::arg("target"), py::arg("samples") = 50,
                    py::arg("noise_stddev") = 0.1F, py::arg("seed") = 0,
                    py::call_guard<py::gil_scoped_release>());
    explanation.def(
        "smoothgrad",
        [](clnn::nn::Module& model, const clnn::Tensor& input, const std::size_t target,
           const std::size_t samples, const float noise_stddev, const std::uint64_t seed,
           const std::optional<std::size_t> batch_index) {
            return clnn::explain::smoothgrad(model, input, {target, batch_index}, samples,
                                             noise_stddev, seed);
        },
        py::arg("model"), py::arg("input"), py::arg("target"), py::arg("samples") = 50,
        py::arg("noise_stddev") = 0.1F, py::arg("seed") = 0,
        py::arg("batch_index") = std::nullopt, py::call_guard<py::gil_scoped_release>());

    auto optimization = module.def_submodule("optim");
    py::class_<clnn::optim::ParameterGroup>(optimization, "ParameterGroup")
        .def(py::init(
                 [](clnn::nn::Module& model, const float learning_rate, const float weight_decay) {
                     return clnn::optim::ParameterGroup{model.parameters(), learning_rate,
                                                        weight_decay};
                 }),
             py::arg("model"), py::arg("learning_rate"), py::arg("weight_decay") = 0.0F,
             py::keep_alive<1, 2>())
        .def_readwrite("learning_rate", &clnn::optim::ParameterGroup::learning_rate)
        .def_readwrite("weight_decay", &clnn::optim::ParameterGroup::weight_decay);
    py::class_<clnn::optim::Optimizer, py::smart_holder>(optimization, "Optimizer")
        .def("step", &clnn::optim::Optimizer::step, py::call_guard<py::gil_scoped_release>())
        .def("zero_grad", &clnn::optim::Optimizer::zero_grad)
        .def_property_readonly("parameter_group_count",
                               &clnn::optim::Optimizer::parameter_group_count)
        .def("learning_rate", &clnn::optim::Optimizer::learning_rate, py::arg("group") = 0)
        .def("set_learning_rate", &clnn::optim::Optimizer::set_learning_rate,
             py::arg("learning_rate"), py::arg("group") = 0);
    py::class_<clnn::optim::SGD, clnn::optim::Optimizer, py::smart_holder>(optimization, "SGD")
        .def(py::init([](clnn::nn::Module& model, const float learning_rate, const float momentum,
                         const float weight_decay) {
                 return std::make_unique<clnn::optim::SGD>(model.parameters(), learning_rate,
                                                           momentum, weight_decay);
             }),
             py::arg("model"), py::arg("learning_rate"), py::arg("momentum") = 0.0F,
             py::arg("weight_decay") = 0.0F, py::keep_alive<1, 2>())
        .def(py::init([](std::vector<clnn::optim::ParameterGroup> groups, const float momentum) {
                 return std::make_unique<clnn::optim::SGD>(
                     clnn::optim::ParameterGroups{std::move(groups)}, momentum);
             }),
             py::arg("parameter_groups"), py::arg("momentum") = 0.0F, py::keep_alive<1, 2>());
    py::class_<clnn::optim::Adam, clnn::optim::Optimizer, py::smart_holder>(optimization, "Adam")
        .def(py::init([](clnn::nn::Module& model, const float learning_rate, const float beta1,
                         const float beta2, const float epsilon, const float weight_decay,
                         const bool decoupled_weight_decay) {
                 return std::make_unique<clnn::optim::Adam>(model.parameters(), learning_rate,
                                                            beta1, beta2, epsilon, weight_decay,
                                                            decoupled_weight_decay);
             }),
             py::arg("model"), py::arg("learning_rate") = 1.0e-3F, py::arg("beta1") = 0.9F,
             py::arg("beta2") = 0.999F, py::arg("epsilon") = 1.0e-8F,
             py::arg("weight_decay") = 0.0F, py::arg("decoupled_weight_decay") = false,
             py::keep_alive<1, 2>())
        .def(
            py::init([](std::vector<clnn::optim::ParameterGroup> groups, const float beta1,
                        const float beta2, const float epsilon, const bool decoupled_weight_decay) {
                return std::make_unique<clnn::optim::Adam>(
                    clnn::optim::ParameterGroups{std::move(groups)}, beta1, beta2, epsilon,
                    decoupled_weight_decay);
            }),
            py::arg("parameter_groups"), py::arg("beta1") = 0.9F, py::arg("beta2") = 0.999F,
            py::arg("epsilon") = 1.0e-8F, py::arg("decoupled_weight_decay") = false,
            py::keep_alive<1, 2>());
    py::class_<clnn::optim::AdamW, clnn::optim::Optimizer, py::smart_holder>(optimization, "AdamW")
        .def(py::init([](clnn::nn::Module& model, const float learning_rate, const float beta1,
                         const float beta2, const float epsilon, const float weight_decay) {
                 return std::make_unique<clnn::optim::AdamW>(model.parameters(), learning_rate,
                                                             beta1, beta2, epsilon, weight_decay);
             }),
             py::arg("model"), py::arg("learning_rate") = 1.0e-3F, py::arg("beta1") = 0.9F,
             py::arg("beta2") = 0.999F, py::arg("epsilon") = 1.0e-8F,
             py::arg("weight_decay") = 1.0e-2F, py::keep_alive<1, 2>())
        .def(py::init([](std::vector<clnn::optim::ParameterGroup> groups, const float beta1,
                         const float beta2, const float epsilon) {
                 return std::make_unique<clnn::optim::AdamW>(
                     clnn::optim::ParameterGroups{std::move(groups)}, beta1, beta2, epsilon);
             }),
             py::arg("parameter_groups"), py::arg("beta1") = 0.9F, py::arg("beta2") = 0.999F,
             py::arg("epsilon") = 1.0e-8F, py::keep_alive<1, 2>());
    py::class_<clnn::optim::LRScheduler, py::smart_holder>(optimization, "LRScheduler")
        .def("step", &clnn::optim::LRScheduler::step)
        .def_property_readonly("steps", &clnn::optim::LRScheduler::steps);
    py::class_<clnn::optim::ExponentialLR, clnn::optim::LRScheduler, py::smart_holder>(
        optimization, "ExponentialLR")
        .def(py::init<clnn::optim::Optimizer&, float>(), py::arg("optimizer"), py::arg("gamma"),
             py::keep_alive<1, 2>());
    py::class_<clnn::optim::StepLR, clnn::optim::LRScheduler, py::smart_holder>(optimization,
                                                                                "StepLR")
        .def(py::init<clnn::optim::Optimizer&, std::size_t, float>(), py::arg("optimizer"),
             py::arg("step_size"), py::arg("gamma") = 0.1F, py::keep_alive<1, 2>());
    py::class_<clnn::optim::CosineAnnealingLR, clnn::optim::LRScheduler, py::smart_holder>(
        optimization, "CosineAnnealingLR")
        .def(py::init<clnn::optim::Optimizer&, std::size_t, float>(), py::arg("optimizer"),
             py::arg("maximum_steps"), py::arg("minimum_learning_rate") = 0.0F,
             py::keep_alive<1, 2>());
    optimization.def("save_state_dict", &clnn::optim::save_state_dict);
    optimization.def("load_state_dict", &clnn::optim::load_state_dict);

    auto data = module.def_submodule("data");
    py::class_<clnn::data::TensorDataset>(data, "TensorDataset")
        .def(py::init(&dataset_from_arrays), py::arg("inputs"), py::arg("targets"),
             py::arg("class_labels") = std::nullopt)
        .def("__len__", &clnn::data::TensorDataset::size)
        .def_property_readonly("input_shape", &clnn::data::TensorDataset::input_shape)
        .def_property_readonly("target_shape", &clnn::data::TensorDataset::target_shape)
        .def_property_readonly("has_class_labels", &clnn::data::TensorDataset::has_class_labels)
        .def("subset", &clnn::data::TensorDataset::subset);
    py::class_<clnn::data::DatasetSplit>(data, "DatasetSplit")
        .def_readonly("training", &clnn::data::DatasetSplit::training)
        .def_readonly("validation", &clnn::data::DatasetSplit::validation)
        .def_readonly("test", &clnn::data::DatasetSplit::test);
    py::class_<clnn::data::Batch>(data, "Batch")
        .def_readonly("inputs", &clnn::data::Batch::inputs)
        .def_readonly("targets", &clnn::data::Batch::targets)
        .def_readonly("class_labels", &clnn::data::Batch::class_labels)
        .def_readonly("size", &clnn::data::Batch::size);
    py::class_<clnn::data::DataLoader>(data, "DataLoader")
        .def(py::init<const clnn::data::TensorDataset&, std::size_t, clnn::Device, bool,
                      std::uint32_t, bool, bool>(),
             py::arg("dataset"), py::arg("batch_size"), py::arg("device") = clnn::Device::cpu(),
             py::arg("shuffle") = false, py::arg("seed") = 0, py::arg("drop_last") = false,
             py::arg("prefetch") = false, py::keep_alive<1, 2>())
        .def("reset", &clnn::data::DataLoader::reset, py::arg("seed") = std::nullopt)
        .def_property_readonly("batch_count", &clnn::data::DataLoader::batch_count)
        .def_property_readonly("prefetch_enabled", &clnn::data::DataLoader::prefetch_enabled)
        .def(
            "__iter__",
            [](clnn::data::DataLoader& self) -> clnn::data::DataLoader& {
                self.reset();
                return self;
            },
            py::return_value_policy::reference_internal)
        .def("__next__", [](clnn::data::DataLoader& self) {
            auto batch = self.next();
            if (!batch)
                throw py::stop_iteration();
            return std::move(*batch);
        });
    data.def("split", &clnn::data::split, py::arg("dataset"), py::arg("training_fraction"),
             py::arg("validation_fraction"), py::arg("seed") = 0);
    data.def("load_csv_numerical", &clnn::data::load_csv_numerical, py::arg("path"),
             py::arg("input_columns"), py::arg("target_columns"), py::arg("delimiter") = ',');
    data.def("load_cifar10_batch", &clnn::data::load_cifar10_batch, py::arg("path"),
             py::arg("normalize") = true);
}
