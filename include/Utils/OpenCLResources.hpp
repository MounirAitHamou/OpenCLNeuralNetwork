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

        unsigned int getErrorCode() const {
            return m_errorCode;
        }
        
        static OpenCLResources createOpenCLResources(const std::string& p_kernelsPath = "kernels");

        static OpenCLResources createOpenCLResources(std::shared_ptr<SharedResources> p_sharedResources);

        void print() const;

    private:
        std::shared_ptr<SharedResources> m_sharedResources;
        cl::CommandQueue m_forwardBackpropQueue;
        cl::CommandQueue m_deltaToGradientQueue;
        cl::CommandQueue m_concurrentQueue;
        unsigned int m_errorCode;

        OpenCLResources() : m_errorCode(1) {}

        OpenCLResources(cl::Context&& p_context, cl::Program&& p_program, 
                        cl::CommandQueue&& p_forwardBackpropQueue, cl::CommandQueue&& p_deltaToGradientQueue, 
                        cl::CommandQueue&& p_concurrentQueue)
            : m_sharedResources(std::make_shared<SharedResources>(std::move(p_context), std::move(p_program))),
              m_forwardBackpropQueue(std::move(p_forwardBackpropQueue)),
              m_deltaToGradientQueue(std::move(p_deltaToGradientQueue)),
              m_concurrentQueue(std::move(p_concurrentQueue)),
              m_errorCode(0) {}

        OpenCLResources(std::shared_ptr<SharedResources> p_sharedResources, 
                        cl::CommandQueue&& p_forwardBackpropQueue, cl::CommandQueue&& p_deltaToGradientQueue, 
                        cl::CommandQueue&& p_concurrentQueue)
            : m_sharedResources(std::move(p_sharedResources)),
              m_forwardBackpropQueue(std::move(p_forwardBackpropQueue)),
              m_deltaToGradientQueue(std::move(p_deltaToGradientQueue)),
              m_concurrentQueue(std::move(p_concurrentQueue)),
              m_errorCode(0) {}

        static std::vector<std::string> getAllKernelFiles(const std::string& p_folderPath);
    };

    void saveBuffer(const cl::CommandQueue& p_queue, const cl::Buffer& p_buffer, H5::Group& p_group, const std::string& p_name, size_t p_size);

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