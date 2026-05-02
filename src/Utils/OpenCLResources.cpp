#include "Utils/OpenCLResources.hpp"
namespace Utils
{
    OpenCLResources OpenCLResources::createOpenCLResources(const std::string &p_kernelsPath, size_t p_platformIndex, size_t p_deviceIndex, bool p_verbose)
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty())
        {
            CLNN_FATAL("No OpenCL platforms found. Please ensure OpenCL drivers are installed.");
        }
        if (p_verbose)
        {
            std::cout << "Total platforms found: " << platforms.size() << "\n";
            for (size_t i = 0; i < platforms.size(); ++i)
            {
                std::cout << "Platform " << i << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << "\n";
            }
        }
        size_t platformIndex = p_platformIndex;
        if (p_platformIndex >= platforms.size())
        {
            std::cerr << "Invalid platform index, using 0 by default.\n";
            platformIndex = 0;
        }

        cl::Platform platform = platforms[platformIndex];
        if (p_verbose)
        {
            std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
        }
        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty())
        {
            platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
            if (devices.empty())
            {
                CLNN_FATAL("No OpenCL devices (GPU or CPU) found on the selected platform.");
            }
        }
        if (p_verbose)
        {

            std::cout << "Total devices found: " << devices.size() << "\n";
            for (size_t i = 0; i < devices.size(); ++i)
            {
                std::cout << "Device " << i << ": "
                          << ((devices[i].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU) ? "GPU" : "CPU")
                          << ", " << devices[i].getInfo<CL_DEVICE_VENDOR>()
                          << ", " << devices[i].getInfo<CL_DEVICE_VERSION>()
                          << ", " << devices[i].getInfo<CL_DEVICE_NAME>() << "\n";
            }
        }

        size_t deviceIndex = p_deviceIndex;
        if (p_deviceIndex >= devices.size())
        {
            std::cerr << "Invalid device index, using 0 by default.\n";
            deviceIndex = 0;
        }

        cl::Device device = devices[deviceIndex];
        if (p_verbose)
        {
            std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
        }
        cl::Context context(device);
        cl::CommandQueue forwardBackpropQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
        cl::CommandQueue deltaToGradientQueue;
        cl::CommandQueue concurrentQueue;

        cl_ulong props;
        device.getInfo(CL_DEVICE_QUEUE_PROPERTIES, &props);
        if ((props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0U)
        {
            deltaToGradientQueue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
            concurrentQueue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
        }
        else
        {
            deltaToGradientQueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
            concurrentQueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
        }

        // Handle kernels safely
        std::vector<std::string> kernelFiles = getAllKernelFiles(p_kernelsPath);
        cl::Program::Sources sources;

        if (kernelFiles.empty())
        {
            std::cout << "No kernel files found or folder does not exist. Continuing without kernels." << "\n";
            const char *emptyKernel = "__kernel void dummy() {}";
            sources.emplace_back(emptyKernel, strlen(emptyKernel));
        }
        else
        {
            if (p_verbose)
            {
                std::cout << "Found " << kernelFiles.size() << " kernel files:" << "\n";
            }
            for (const auto &filePath : kernelFiles)
            {
                if (p_verbose)
                {
                    std::cout << "- " << filePath << "\n";
                }
                std::ifstream file(filePath);
                if (!file)
                {
                    std::cerr << "Error: Could not open kernel file: " << filePath << "\n";
                    continue;
                }
                std::string sourceCode((std::istreambuf_iterator<char>(file)),
                                       std::istreambuf_iterator<char>());
                sources.emplace_back(sourceCode.c_str(), sourceCode.length());
            }
        }

        cl::Program program(context, sources);

        if (sources.empty())
        {
            return {std::move(context), std::move(program), std::move(forwardBackpropQueue), std::move(deltaToGradientQueue), std::move(concurrentQueue)};
        }

        std::string buildOptions = "-I " + p_kernelsPath + "/include -DCL_ENABLE_PRINTF";
        cl_int buildStatus = program.build({device}, buildOptions.c_str());
        std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        if (!buildLog.empty())
        {
            std::cerr << "Build log for device " << device.getInfo<CL_DEVICE_NAME>() << ":\n"
                      << buildLog << "\n";
        }
        if (buildStatus != CL_SUCCESS)
        {
            std::cerr << "Warning: OpenCL program build failed, continuing without kernels.\n";
        }

        return {
            std::move(context),
            std::move(program),
            std::move(forwardBackpropQueue),
            std::move(deltaToGradientQueue),
            std::move(concurrentQueue)};
    }

    OpenCLResources OpenCLResources::createOpenCLResources(std::shared_ptr<SharedResources> p_sharedResources)
    {
        if (!p_sharedResources)
        {
            std::cerr << "Error: SharedResources pointer is null." << "\n";
            CLNN_FATAL("SharedResources pointer is null.");
        }
        cl::Context context = p_sharedResources->getContext();
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        if (devices.empty())
        {
            std::cerr << "Error: No devices found in the provided context." << "\n";
            CLNN_FATAL("No devices found in the provided context.");
        }
        cl::Device &device = devices[0];

        cl::CommandQueue forwardBackpropQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
        cl::CommandQueue deltaToGradientQueue;
        cl::CommandQueue concurrentQueue;

        cl_ulong props;
        device.getInfo(CL_DEVICE_QUEUE_PROPERTIES, &props);
        if ((props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) != 0U)
        {
            deltaToGradientQueue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
            concurrentQueue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
        }
        else
        {
            deltaToGradientQueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
            concurrentQueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
        }
        return {std::move(p_sharedResources), std::move(forwardBackpropQueue), std::move(deltaToGradientQueue), std::move(concurrentQueue)};
    }

    void OpenCLResources::print() const
    {
        std::cout << "--- OpenCLResources Status ---" << "\n";

        if (m_sharedResources != nullptr)
        {
            std::cout << "SharedResources: Initialized" << "\n";

            if (m_sharedResources->getContext()() != nullptr)
            {
                std::cout << "  Context: Valid" << "\n";
            }
            else
            {
                std::cout << "  Context: Invalid" << "\n";
            }

            if (m_sharedResources->getProgram()() != nullptr)
            {
                std::cout << "  Program: Valid" << "\n";
            }
            else
            {
                std::cout << "  Program: Invalid" << "\n";
            }
        }
        else
        {
            std::cout << "SharedResources: Not Initialized" << "\n";
        }

        if (m_forwardBackpropQueue() != nullptr)
        {
            std::cout << "ForwardBackpropQueue: Valid" << "\n";
        }
        else
        {
            std::cout << "ForwardBackpropQueue: Invalid" << "\n";
        }

        if (m_deltaToGradientQueue() != nullptr)
        {
            std::cout << "DeltaToGradientQueue: Valid" << "\n";
        }
        else
        {
            std::cout << "DeltaToGradientQueue: Invalid" << "\n";
        }

        if (m_concurrentQueue() != nullptr)
        {
            std::cout << "ConcurrentQueue: Valid" << "\n";
        }
        else
        {
            std::cout << "ConcurrentQueue: Invalid" << "\n";
        }
        std::cout << "------------------------------" << "\n";
    }

    bool OpenCLResources::valid() const
    {
        return m_sharedResources != nullptr &&
               m_sharedResources->getContext()() != nullptr &&
               m_sharedResources->getProgram()() != nullptr &&
               m_forwardBackpropQueue() != nullptr &&
               m_deltaToGradientQueue() != nullptr &&
               m_concurrentQueue() != nullptr;
    }

    std::vector<std::string> OpenCLResources::getAllKernelFiles(const std::string &p_folderPath)
    {
        std::vector<std::string> filePaths;

        if (!std::filesystem::exists(p_folderPath) || !std::filesystem::is_directory(p_folderPath))
        {
            return {};
        }

        for (const auto &entry : std::filesystem::recursive_directory_iterator(p_folderPath))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".cl")
            {
                filePaths.push_back(entry.path().string());
            }
        }

        std::ranges::sort(filePaths);

        return filePaths;
    }

    void saveBuffer(const cl::CommandQueue &p_queue, const cl::Buffer &p_buffer, H5::Group &p_group, const std::string &p_name, size_t p_size)
    {
        if (H5Lexists(p_group.getId(), p_name.c_str(), H5P_DEFAULT) > 0)
        {
            std::cerr << "Warning: Dataset '" << p_name << "' already exists. Skipping write.\n";
            return;
        }
        std::vector<float> host_data(p_size);
        p_queue.enqueueReadBuffer(p_buffer, BLOCKING, NO_OFFSET, p_size * sizeof(float), host_data.data());

        H5::DataSpace dataspace(H5S_SIMPLE);
        std::array<hsize_t, 1> dims{p_size};
        dataspace.setExtentSimple(1, dims.data());

        H5::DataSet dataset = p_group.createDataSet(p_name, H5::PredType::NATIVE_FLOAT, dataspace);
        dataset.write(host_data.data(), H5::PredType::NATIVE_FLOAT);
    }

    cl::Buffer loadBuffer(const cl::Context &p_context,
                          const H5::Group &p_layerGroup,
                          const std::string &p_bufferName,
                          size_t p_size)
    {
        std::vector<float> data(p_size);
        p_layerGroup.openDataSet(p_bufferName).read(data.data(), H5::PredType::NATIVE_FLOAT);
        cl::Buffer buffer(p_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, p_size * sizeof(float), data.data());
        return buffer;
    }

    void printCLBuffer(const cl::CommandQueue &p_queue, const cl::Buffer &p_buffer, size_t p_size, const std::string &p_label)
    {
        std::vector<float> hostData(p_size);
        p_queue.enqueueReadBuffer(p_buffer, BLOCKING, NO_OFFSET, p_size * sizeof(float), hostData.data());
        std::cout << p_label << " Buffer Data: ";
        for (const auto &value : hostData)
        {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }

    std::vector<float> readCLBuffer(const cl::CommandQueue &p_queue, const cl::Buffer &p_buffer, size_t p_offset, size_t p_size)
    {
        std::vector<float> hostData(p_size);
        p_queue.enqueueReadBuffer(p_buffer, BLOCKING, p_offset * sizeof(float), p_size * sizeof(float), hostData.data());
        return hostData;
    }

    cl::Buffer createCLBuffer(const cl::Context &p_context, std::vector<float> &p_data)
    {
        cl::Buffer buffer(p_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, p_data.size() * sizeof(float), p_data.data());
        return buffer;
    }

    bool compareCLBuffers(const cl::CommandQueue &p_queue, const cl::Buffer &p_buffer1, const cl::Buffer &p_buffer2, size_t p_size, float p_epsilon)
    {
        std::vector<float> data1(p_size);
        std::vector<float> data2(p_size);
        p_queue.enqueueReadBuffer(p_buffer1, BLOCKING, NO_OFFSET, p_size * sizeof(float), data1.data());
        p_queue.enqueueReadBuffer(p_buffer2, BLOCKING, NO_OFFSET, p_size * sizeof(float), data2.data());

        for (size_t i = 0; i < p_size; ++i)
        {
            if (std::fabs(data1[i] - data2[i]) > p_epsilon)
            {
                return false;
            }
        }
        return true;
    }

    void cpuGemm2D(const std::vector<std::vector<float>> &p_A,
                   const std::vector<std::vector<float>> &p_B,
                   std::vector<std::vector<float>> &p_C,
                   bool p_transposeA,
                   bool p_transposeB)
    {
        size_t firstDimension = p_transposeA ? p_A[0].size() : p_A.size();
        size_t secondDimension = p_transposeA ? p_A.size() : p_A[0].size();
        size_t thirdDimension = p_transposeB ? p_B.size() : p_B[0].size();

        p_C.assign(firstDimension, std::vector<float>(secondDimension, 0.0F));

        for (size_t firstIndex = 0; firstIndex < firstDimension; ++firstIndex)
        {
            for (size_t secondIndex = 0; secondIndex < secondDimension; ++secondIndex)
            {
                float sum = 0.0F;
                for (size_t thirdIndex = 0; thirdIndex < thirdDimension; ++thirdIndex)
                {
                    float aEntry = p_transposeA ? p_A[thirdIndex][firstIndex] : p_A[firstIndex][thirdIndex];
                    float bEntry = p_transposeB ? p_B[secondIndex][thirdIndex] : p_B[thirdIndex][secondIndex];
                    sum += aEntry * bEntry;
                }
                p_C[firstIndex][secondIndex] = sum;
            }
        }
    }

    void cpuGemv2D(const std::vector<std::vector<float>> &p_A,
                   const std::vector<float> &p_x,
                   std::vector<float> &p_y,
                   bool p_transposeA)
    {
        size_t firstDimension = p_transposeA ? p_A[0].size() : p_A.size();
        size_t secondDimension = p_transposeA ? p_A.size() : p_A[0].size();

        p_y.assign(firstDimension, 0.0F);

        for (size_t firstIndex = 0; firstIndex < firstDimension; ++firstIndex)
        {
            float sum = 0.0F;
            for (size_t secondIndex = 0; secondIndex < secondDimension; ++secondIndex)
            {
                float aEntry = p_transposeA ? p_A[secondIndex][firstIndex] : p_A[firstIndex][secondIndex];
                sum += aEntry * p_x[secondIndex];
            }
            p_y[firstIndex] = sum;
        }
    }

    std::vector<std::vector<float>> readBuffer2D(
        const cl::CommandQueue &p_queue,
        const cl::Buffer &p_buffer,
        size_t p_rows,
        size_t p_cols)
    {
        std::vector<float> flat(p_rows * p_cols);
        cl_int err = p_queue.enqueueReadBuffer(p_buffer, BLOCKING, NO_OFFSET, sizeof(float) * flat.size(), flat.data());
        if (err != CL_SUCCESS)
        {
            CLNN_FATAL("Failed to read OpenCL buffer (error code: " + std::to_string(err) + ")");
        }

        std::vector<std::vector<float>> matrix(p_rows, std::vector<float>(p_cols));
        for (size_t i = 0; i < p_rows; ++i)
        {
            for (size_t j = 0; j < p_cols; ++j)
            {
                matrix[i][j] = flat[(i * p_cols) + j];
            }
        }

        return matrix;
    }

    bool compare2D(const std::vector<std::vector<float>> &p_A,
                   const std::vector<std::vector<float>> &p_B,
                   float p_tol)
    {
        for (size_t i = 0; i < p_A.size(); ++i)
        {
            for (size_t j = 0; j < p_A[0].size(); ++j)
            {
                if (std::fabs(p_A[i][j] - p_B[i][j]) > p_tol)
                {
                    std::cerr << "Mismatch at (" << i << "," << j << "): "
                              << "CPU=" << p_A[i][j] << ", GPU=" << p_B[i][j] << "\n";
                    return false;
                }
            }
        }
        return true;
    }

    std::vector<float> readBuffer1D(
        const cl::CommandQueue &p_queue,
        const cl::Buffer &p_buffer,
        size_t p_size)
    {
        std::vector<float> data(p_size);
        cl_int err = p_queue.enqueueReadBuffer(p_buffer, BLOCKING, NO_OFFSET, sizeof(float) * p_size, data.data());
        if (err != CL_SUCCESS)
        {
            CLNN_FATAL("Failed to read OpenCL buffer (error code: " + std::to_string(err) + ")");
        }
        return data;
    }

    bool compare1D(const std::vector<float> &p_A,
                   const std::vector<float> &p_B,
                   float p_tol)
    {
        for (size_t i = 0; i < p_A.size(); ++i)
        {
            if (std::fabs(p_A[i] - p_B[i]) > p_tol)
            {
                std::cerr << "Mismatch at index " << i << ": "
                          << "CPU=" << p_A[i] << ", GPU=" << p_B[i] << "\n";
                return false;
            }
        }
        return true;
    }
}