#include "Utils/OpenCLResources.hpp"
namespace Utils {
    OpenCLResources OpenCLResources::createOpenCLResources(const std::string& p_kernelsPath){
        std::cout << "OpenCL Neural Network Example (Batched Training)\n";

        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.empty()) {
            throw std::runtime_error("No OpenCL platforms found. Please ensure OpenCL drivers are installed.");
        }

        std::cout << "Total platforms found: " << platforms.size() << "\n";
        std::cout << "Available platforms:\n";
        for (int i = 0; i < platforms.size(); ++i) {
            std::cout << "Platform " << i << ": " << platforms[i].getInfo<CL_PLATFORM_NAME>() << std::endl;
        }

        int platformIndex = 0;
        /*if (platforms.size() == 1) {
            std::cout << "Only one platform found, using it by default.\n";
            platformIndex = 0;
        } else {
            std::cout << "Please select a platform by entering its index (0 to " << platforms.size() - 1 << "): ";
            std::cin >> platformIndex;
            if (std::cin.fail() || platformIndex < 0 || platformIndex >= platforms.size()) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                throw std::runtime_error("Invalid input for platform selection. Please enter a valid integer index within the range.");
            }
        }*/
        cl::Platform platform = platforms[platformIndex];
        std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;


        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
        if (devices.empty()) {
            platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
            if (devices.empty()) {
                throw std::runtime_error("No OpenCL devices (GPU or CPU) found on the selected platform.");
            }
        }

        std::cout << "Total devices found: " << devices.size() << "\n";
        std::cout << "Available devices:\n";
        for (int i = 0; i < devices.size(); ++i) {
            std::cout << "Device " << i << ": ";
            std::cout << "Type: " << (devices[i].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU") << ", ";
            std::cout << "Vendor: " << devices[i].getInfo<CL_DEVICE_VENDOR>() << ", ";
            std::cout << "Version: " << devices[i].getInfo<CL_DEVICE_VERSION>() << ", ";
            std::cout << devices[i].getInfo<CL_DEVICE_NAME>() << "\n";
        }

        int deviceIndex;
        if (devices.size() == 1) {
            std::cout << "Only one device found, using it by default.\n";
            deviceIndex = 0;
        } else {
            std::cout << "Please select a device by entering its index (0 to " << devices.size() - 1 << "): ";
            std::cin >> deviceIndex;
            if (std::cin.fail() || deviceIndex < 0 || deviceIndex >= devices.size()) {
                std::cin.clear();
                std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
                throw std::runtime_error("Invalid input for device selection. Please enter a valid integer index within the range.");
            }
        }
        cl::Device device = devices[deviceIndex];
        std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

        cl::Context context(device);
        cl::CommandQueue forwardBackpropQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
        cl::CommandQueue deltaToGradientQueue;
        cl::CommandQueue concurrentQueue;

        cl_ulong props;
        device.getInfo(CL_DEVICE_QUEUE_PROPERTIES, &props);

        if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
            deltaToGradientQueue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
            concurrentQueue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
        } else {
            deltaToGradientQueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
            concurrentQueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
        }

        std::vector<std::string> kernelFiles = getAllKernelFiles(p_kernelsPath);

        if (kernelFiles.empty()) {
            std::cout << "No kernel files found or folder does not exist. Proceeding without kernels." << std::endl;
            throw std::runtime_error("No valid kernel sources found. Ensure .cl files are in the specified path.");
        } else {
            std::cout << "Found " << kernelFiles.size() << " kernel files:" << std::endl;
            for (const auto& filePath : kernelFiles) {
                std::cout << "- " << filePath << std::endl;
            }
        }

        cl::Program::Sources sources;
        for (const auto& filePath : kernelFiles) {
            std::ifstream file(filePath);
            if (!file) {
                std::cerr << "Error: Could not open kernel file: " << filePath << std::endl;
                continue;
            }
            std::string sourceCode((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
            sources.emplace_back(sourceCode.c_str(), sourceCode.length());
        }

        if (sources.empty()) {
            throw std::runtime_error("No valid kernel sources were successfully loaded. Check file paths and permissions.");
        }

        cl::Program program(context, sources);
        std::string includeArg = "-I " + p_kernelsPath + "/include";
        std::string printfArg = "-DCL_ENABLE_PRINTF";
        std::string buildOptions = includeArg + " " + printfArg;

        cl_int buildStatus = program.build({ device }, buildOptions.c_str());

        if (buildStatus != CL_SUCCESS) {
            std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
            std::cerr << "Error building program: " << buildStatus << "\n";
            std::cerr << "Build log for device " << device.getInfo<CL_DEVICE_NAME>() << ":\n" << buildLog << std::endl;
            throw std::runtime_error("Failed to build OpenCL program.");
        }

        
        std::string buildLog = program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device);
        std::cerr << "Build log for device " << device.getInfo<CL_DEVICE_NAME>() << ":\n" << buildLog << std::endl;
        if (!buildLog.empty()) {
            return OpenCLResources();
        }

        return OpenCLResources(std::move(context), std::move(program), std::move(forwardBackpropQueue), std::move(deltaToGradientQueue), std::move(concurrentQueue));
    }

    OpenCLResources OpenCLResources::createOpenCLResources(std::shared_ptr<SharedResources> p_sharedResources) {
        if (!p_sharedResources) {
            std::cerr << "Error: SharedResources pointer is null." << std::endl;
            throw std::invalid_argument("SharedResources pointer is null.");
        }
        cl::Context context = p_sharedResources->getContext();
        std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
        if (devices.empty()) {
            std::cerr << "Error: No devices found in the provided context." << std::endl;
            throw std::runtime_error("No devices found in the provided context.");
        }
        cl::Device device = devices[0];

        cl::CommandQueue forwardBackpropQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
        cl::CommandQueue deltaToGradientQueue;
        cl::CommandQueue concurrentQueue;

        cl_ulong props;
        device.getInfo(CL_DEVICE_QUEUE_PROPERTIES, &props);
        if (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) {
            deltaToGradientQueue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
            concurrentQueue = cl::CommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE);
        } else {
            deltaToGradientQueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
            concurrentQueue = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE);
        }
        return OpenCLResources(std::move(p_sharedResources), std::move(forwardBackpropQueue), std::move(deltaToGradientQueue), std::move(concurrentQueue));
    }

    void OpenCLResources::print() const {
        std::cout << "--- OpenCLResources Status ---" << std::endl;
        
        if (m_sharedResources) {
            std::cout << "SharedResources: Initialized" << std::endl;
            
            if (m_sharedResources->getContext()()) {
                std::cout << "  Context: Valid" << std::endl;
            } else {
                std::cout << "  Context: Invalid" << std::endl;
            }
            
            if (m_sharedResources->getProgram()()) {
                std::cout << "  Program: Valid" << std::endl;
            } else {
                std::cout << "  Program: Invalid" << std::endl;
            }

        } else {
            std::cout << "SharedResources: Not Initialized" << std::endl;
        }
        
        if (m_forwardBackpropQueue()) {
            std::cout << "ForwardBackpropQueue: Valid" << std::endl;
        } else {
            std::cout << "ForwardBackpropQueue: Invalid" << std::endl;
        }
        
        if (m_deltaToGradientQueue()) {
            std::cout << "DeltaToGradientQueue: Valid" << std::endl;
        } else {
            std::cout << "DeltaToGradientQueue: Invalid" << std::endl;
        }

        if (m_concurrentQueue()) {
            std::cout << "ConcurrentQueue: Valid" << std::endl;
        } else {
            std::cout << "ConcurrentQueue: Invalid" << std::endl;
        }

        std::cout << "Error Code: " << m_errorCode << std::endl;
        std::cout << "------------------------------" << std::endl;
    }

    std::vector<std::string> OpenCLResources::getAllKernelFiles(const std::string& p_folderPath) {
        std::vector<std::string> filePaths;

        if (!std::filesystem::exists(p_folderPath)) {
            std::cerr << "Error: Folder '" << p_folderPath << "' does not exist." << std::endl;
            return filePaths;
        }

        if (!std::filesystem::is_directory(p_folderPath)) {
            std::cerr << "Error: Path '" << p_folderPath << "' is not a directory." << std::endl;
            return filePaths;
        }

        try {
            for (const auto& entry : std::filesystem::recursive_directory_iterator(p_folderPath)) {
                if (entry.is_regular_file() && (entry.path().extension() == ".cl")) {
                    filePaths.push_back(entry.path().string());
                }
            }
        } catch (const std::filesystem::filesystem_error& ex) {
            std::cerr << "Filesystem error during traversal: " << ex.what() << std::endl;
        }

        std::sort(filePaths.begin(), filePaths.end());
        return filePaths;
    }

    void saveBuffer(const cl::CommandQueue& p_queue, const cl::Buffer& p_buffer, H5::Group& p_group, const std::string& p_name, size_t p_size) {
        if (H5Lexists(p_group.getId(), p_name.c_str(), H5P_DEFAULT)) {
            std::cerr << "Warning: Dataset '" << p_name << "' already exists. Skipping write.\n";
            return;
        }
        std::vector<float> host_data(p_size);
        p_queue.enqueueReadBuffer(p_buffer, BLOCKING_READ, NO_OFFSET, p_size * sizeof(float), host_data.data());

        H5::DataSpace dataspace(H5S_SIMPLE);
        hsize_t dims[1] = {p_size};
        dataspace.setExtentSimple(1, dims);

        H5::DataSet dataset = p_group.createDataSet(p_name, H5::PredType::NATIVE_FLOAT, dataspace);
        dataset.write(host_data.data(), H5::PredType::NATIVE_FLOAT);
    }

    cl::Buffer loadBuffer(const cl::Context& p_context, 
                          const H5::Group& p_layerGroup, 
                          const std::string& p_bufferName, 
                          size_t p_size) {
        std::vector<float> data(p_size);
        p_layerGroup.openDataSet(p_bufferName).read(data.data(), H5::PredType::NATIVE_FLOAT);
        cl::Buffer buffer(p_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, p_size * sizeof(float), data.data());
        return buffer;
    }

    void printCLBuffer(const cl::CommandQueue& p_queue, const cl::Buffer& p_buffer, size_t p_size, const std::string& p_label) {
        std::vector<float> hostData(p_size);
        p_queue.enqueueReadBuffer(p_buffer, BLOCKING_READ, NO_OFFSET, p_size * sizeof(float), hostData.data());
        std::cout << p_label << " Buffer Data: ";
        for (const auto& value : hostData) {
            std::cout << value << " ";
        }
        std::cout << "\n";
    }

    std::vector<float> readCLBuffer(const cl::CommandQueue& p_queue, const cl::Buffer& p_buffer, size_t p_size) {
        std::vector<float> hostData(p_size);
        p_queue.enqueueReadBuffer(p_buffer, BLOCKING_READ, NO_OFFSET, p_size * sizeof(float), hostData.data());
        return hostData;
    }

    cl::Buffer createCLBuffer(const cl::Context& p_context, std::vector<float>& p_data) {
        cl::Buffer buffer(p_context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, p_data.size() * sizeof(float), p_data.data());
        return buffer;
    }

    bool compareCLBuffers(const cl::CommandQueue& p_queue, const cl::Buffer& p_buffer1, const cl::Buffer& p_buffer2, size_t p_size, float p_epsilon) {
        std::vector<float> data1(p_size);
        std::vector<float> data2(p_size);
        p_queue.enqueueReadBuffer(p_buffer1, BLOCKING_READ, NO_OFFSET, p_size * sizeof(float), data1.data());
        p_queue.enqueueReadBuffer(p_buffer2, BLOCKING_READ, NO_OFFSET, p_size * sizeof(float), data2.data());

        for (size_t i = 0; i < p_size; ++i) {
            if (std::fabs(data1[i] - data2[i]) > p_epsilon) {
                return false;
            }
        }
        return true;
    }

    void cpuGemm2D(const std::vector<std::vector<float>>& A,
                   const std::vector<std::vector<float>>& B,
                   std::vector<std::vector<float>>& C,
                   bool transposeA,
                   bool transposeB) {
        size_t M = transposeA ? A[0].size() : A.size();
        size_t K = transposeA ? A.size() : A[0].size();
        size_t N = transposeB ? B.size() : B[0].size();

        C.assign(M, std::vector<float>(N, 0.0f));

        for (size_t m = 0; m < M; ++m) {
            for (size_t n = 0; n < N; ++n) {
                float sum = 0.0f;
                for (size_t k = 0; k < K; ++k) {
                    float a = transposeA ? A[k][m] : A[m][k];
                    float b = transposeB ? B[n][k] : B[k][n];
                    sum += a * b;
                }
                C[m][n] = sum;
            }
        }
    }

    void cpuGemv2D(const std::vector<std::vector<float>>& A,
                   const std::vector<float>& x,
                   std::vector<float>& y,
                   bool transposeA) {
        size_t M = transposeA ? A[0].size() : A.size();
        size_t N = transposeA ? A.size() : A[0].size();

        y.assign(M, 0.0f);

        for (size_t m = 0; m < M; ++m) {
            float sum = 0.0f;
            for (size_t n = 0; n < N; ++n) {
                float a = transposeA ? A[n][m] : A[m][n];
                sum += a * x[n];
            }
            y[m] = sum;
        }
    }

    std::vector<std::vector<float>> readBuffer2D(
        const cl::CommandQueue& queue,
        const cl::Buffer& buffer,
        size_t rows,
        size_t cols)
    {
        std::vector<float> flat(rows * cols);
        cl_int err = queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(float) * flat.size(), flat.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read OpenCL buffer (error code: " + std::to_string(err) + ")");
        }

        std::vector<std::vector<float>> matrix(rows, std::vector<float>(cols));
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                matrix[i][j] = flat[i * cols + j];
            }
        }

        return matrix;
    }

    bool compare2D(const std::vector<std::vector<float>>& A,
                   const std::vector<std::vector<float>>& B,
                   float tol) {
        for (size_t i = 0; i < A.size(); ++i)
            for (size_t j = 0; j < A[0].size(); ++j)
                if (std::fabs(A[i][j] - B[i][j]) > tol) {
                    std::cerr << "Mismatch at (" << i << "," << j << "): "
                            << "CPU=" << A[i][j] << ", GPU=" << B[i][j] << std::endl;
                    return false;
                }
        return true;
    }

    std::vector<float> readBuffer1D(
        const cl::CommandQueue& queue,
        const cl::Buffer& buffer,
        size_t size) {
        std::vector<float> data(size);
        cl_int err = queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(float) * size, data.data());
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read OpenCL buffer (error code: " + std::to_string(err) + ")");
        }
        return data;
    }

    bool compare1D(const std::vector<float>& A,
                   const std::vector<float>& B,
                   float tol) {
        for (size_t i = 0; i < A.size(); ++i)
            if (std::fabs(A[i] - B[i]) > tol) {
                std::cerr << "Mismatch at index " << i << ": "
                          << "CPU=" << A[i] << ", GPU=" << B[i] << std::endl;
                return false;
            }
        return true;
    }
}