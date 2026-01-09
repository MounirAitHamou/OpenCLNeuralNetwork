#pragma once
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>
#include <H5Cpp.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <filesystem>
#include <algorithm>
#include <vector>

#ifndef NEURAL_NETWORK_CONSTANTS_HPP
#define NEURAL_NETWORK_CONSTANTS_HPP

const cl_bool BLOCKING_READ = CL_TRUE;
const cl_bool NON_BLOCKING_READ = CL_FALSE;
const size_t NO_OFFSET = 0;
const float NO_SCALAR = 1.0f;
const float CLEAR_C = 0.0f;
template<typename>
inline constexpr bool always_false = false;
#endif

namespace Utils {
    struct SharedResources {
    public:

        SharedResources(cl::Context&& p_context, cl::Program&& p_program)
            : m_context(std::move(p_context)), m_program(std::move(p_program)) {}
        const cl::Context& getContext() const {
            return m_context;
        }

        const cl::Program& getProgram() const {
            return m_program;
        }

    private:
        cl::Context m_context;
        cl::Program m_program;
    };

    struct OpenCLResources {
    public:
        OpenCLResources(const OpenCLResources&) = delete;
        OpenCLResources& operator=(const OpenCLResources&) = delete;
        OpenCLResources(OpenCLResources&&) = default;
        OpenCLResources& operator=(OpenCLResources&&) = default;
        ~OpenCLResources() = default;

        std::shared_ptr<SharedResources> getSharedResources() const {
            return m_sharedResources;
        }

        const cl::Context& getContext() const {
            return m_sharedResources->getContext();
        }

        const cl::Program& getProgram() const {
            return m_sharedResources->getProgram();
        }

        const cl::CommandQueue& getForwardBackpropQueue() const {
            return m_forwardBackpropQueue;
        }
        const cl::CommandQueue& getDeltaToGradientQueue() const {
            return m_deltaToGradientQueue;
        }
        const cl::CommandQueue& getConcurrentQueue() const {
            return m_concurrentQueue;
        }

        static OpenCLResources createOpenCLResources(const std::string& p_kernelsPath = "kernels");

        static OpenCLResources createOpenCLResources(std::shared_ptr<SharedResources> p_sharedResources);

        void print() const;

    private:
        std::shared_ptr<SharedResources> m_sharedResources;
        cl::CommandQueue m_forwardBackpropQueue;
        cl::CommandQueue m_deltaToGradientQueue;
        cl::CommandQueue m_concurrentQueue;

        OpenCLResources(cl::Context&& p_context, cl::Program&& p_program, 
                        cl::CommandQueue&& p_forwardBackpropQueue, cl::CommandQueue&& p_deltaToGradientQueue, 
                        cl::CommandQueue&& p_concurrentQueue)
            : m_sharedResources(std::make_shared<SharedResources>(std::move(p_context), std::move(p_program))),
              m_forwardBackpropQueue(std::move(p_forwardBackpropQueue)),
              m_deltaToGradientQueue(std::move(p_deltaToGradientQueue)),
              m_concurrentQueue(std::move(p_concurrentQueue)) {}

        OpenCLResources(std::shared_ptr<SharedResources> p_sharedResources, 
                        cl::CommandQueue&& p_forwardBackpropQueue, cl::CommandQueue&& p_deltaToGradientQueue, 
                        cl::CommandQueue&& p_concurrentQueue)
            : m_sharedResources(std::move(p_sharedResources)),
              m_forwardBackpropQueue(std::move(p_forwardBackpropQueue)),
              m_deltaToGradientQueue(std::move(p_deltaToGradientQueue)),
              m_concurrentQueue(std::move(p_concurrentQueue)) {}

        static std::vector<std::string> getAllKernelFiles(const std::string& p_folderPath);
    };

    void saveBuffer(const cl::CommandQueue& p_queue, const cl::Buffer& p_buffer, H5::Group& p_group, const std::string& p_name, size_t p_size);

    template <typename T>
    std::vector<T> readVectorFromHDF5(const H5::Group& p_group, const std::string& p_attrName) {
        if (!p_group.attrExists(p_attrName)) {
            throw std::runtime_error("Attribute " + p_attrName + " does not exist");
        }

        H5::Attribute attr = p_group.openAttribute(p_attrName);
        H5::DataType type = attr.getDataType();
        H5::DataSpace space = attr.getSpace();

        hsize_t numElements = 1;
        int rank = space.getSimpleExtentNdims();
        if (rank != 1) {
            throw std::runtime_error("Attribute " + p_attrName + " is not 1D");
        }
        space.getSimpleExtentDims(&numElements, nullptr);

        std::vector<T> result(numElements);
        attr.read(type, result.data());

        return result;
    }


    template <typename T>
    void writeVectorToHDF5(H5::Group& group, const std::string& attrName, const std::vector<T>& data) {
        if (group.attrExists(attrName)) {
            group.removeAttr(attrName);
        }

        hsize_t dims[1] = { data.size() };
        H5::DataSpace space(1, dims);

        const H5::PredType& h5Type =
        []() -> const H5::PredType& {
            if constexpr (std::is_same_v<T, int>) {
                return H5::PredType::NATIVE_INT;
            } else if constexpr (std::is_same_v<T, size_t>) {
                return H5::PredType::NATIVE_HSIZE;
            } else if constexpr (std::is_same_v<T, float>) {
                return H5::PredType::NATIVE_FLOAT;
            } else if constexpr (std::is_same_v<T, double>) {
                return H5::PredType::NATIVE_DOUBLE;
            } else if constexpr (std::is_same_v<T, bool>) {
                return H5::PredType::NATIVE_HBOOL;
            } else if constexpr (std::is_same_v<T, char>) {
                return H5::PredType::NATIVE_CHAR;
            } else if constexpr (std::is_same_v<T, unsigned int>) {
                return H5::PredType::NATIVE_UINT;
            } else {
                static_assert(always_false<T>, "Unsupported type for HDF5 attribute");
            }
        }();

        H5::Attribute attr = group.createAttribute(attrName, h5Type, space);
        attr.write(h5Type, data.data());
    }


    template <typename T>
    T readValueFromHDF5(const H5::Group& group, const std::string& attrName) {
        if (!group.attrExists(attrName)) {
            throw std::runtime_error("Attribute " + attrName + " does not exist");
        }

        H5::Attribute attr = group.openAttribute(attrName);
        H5::DataType type = attr.getDataType();

        T value{};
        attr.read(type, &value);
        return value;
    }

    template <typename T>
    void writeValueToHDF5(H5::Group& group, const std::string& attrName, const T& value) {
        if (group.attrExists(attrName)) {
            group.removeAttr(attrName);
        }

        const H5::PredType& h5Type =
        []() -> const H5::PredType& {
            if constexpr (std::is_same_v<T, int>) {
                return H5::PredType::NATIVE_INT;
            } else if constexpr (std::is_same_v<T, size_t>) {
                return H5::PredType::NATIVE_HSIZE;
            } else if constexpr (std::is_same_v<T, float>) {
                return H5::PredType::NATIVE_FLOAT;
            } else if constexpr (std::is_same_v<T, double>) {
                return H5::PredType::NATIVE_DOUBLE;
            } else if constexpr (std::is_same_v<T, bool>) {
                return H5::PredType::NATIVE_HBOOL;
            } else if constexpr (std::is_same_v<T, char>) {
                return H5::PredType::NATIVE_CHAR;
            } else if constexpr (std::is_same_v<T, unsigned int>) {
                return H5::PredType::NATIVE_UINT;
            } else {
                static_assert(always_false<T>, "Unsupported type for HDF5 attribute");
            }
        }();

        H5::DataSpace space(H5S_SCALAR);
        H5::Attribute attr = group.createAttribute(attrName, h5Type, space);
        attr.write(h5Type, &value);
    }

    cl::Buffer loadBuffer(const cl::Context& p_context, 
                          const H5::Group& p_layerGroup, 
                          const std::string& p_bufferName, 
                          size_t p_size);

    void printCLBuffer(const cl::CommandQueue& p_queue, const cl::Buffer& p_buffer, size_t p_size, const std::string& p_label = "Buffer");

    std::vector<float> readCLBuffer(const cl::CommandQueue& p_queue, const cl::Buffer& p_buffer, size_t p_size);

    cl::Buffer createCLBuffer(const cl::Context& p_context, std::vector<float>& p_data);

    bool compareCLBuffers(const cl::CommandQueue& p_queue, const cl::Buffer& p_buffer1, const cl::Buffer& p_buffer2, size_t p_size, float p_epsilon = 1e-6f);

    void cpuGemm2D(const std::vector<std::vector<float>>& A,
                   const std::vector<std::vector<float>>& B,
                   std::vector<std::vector<float>>& C,
                   bool transposeA = false,
                   bool transposeB = false);
    
    void cpuGemv2D(const std::vector<std::vector<float>>& A,
                   const std::vector<float>& x,
                   std::vector<float>& y,
                   bool transposeA = false);

    std::vector<std::vector<float>> readBuffer2D(
        const cl::CommandQueue& queue,
        const cl::Buffer& buffer,
        size_t rows,
        size_t cols);

    std::vector<float> readBuffer1D(
        const cl::CommandQueue& queue,
        const cl::Buffer& buffer,
        size_t size);

    bool compare2D(const std::vector<std::vector<float>>& A,
                   const std::vector<std::vector<float>>& B,
                   float tol = 1e-4f);

    bool compare1D(const std::vector<float>& A,
                   const std::vector<float>& B,
                   float tol = 1e-4f);


}