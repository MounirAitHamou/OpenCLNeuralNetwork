#include "Utils/OpenCLSetup.hpp"

OpenCLSetup::OpenCLSetup() : context(nullptr), queue(nullptr), program(nullptr) {}

OpenCLSetup::OpenCLSetup(cl::Context context, cl::CommandQueue queue, cl::Program program)
    : context(context), queue(queue), program(program) {}

std::vector<std::string> OpenCLSetup::getAllKernelFiles(const std::string& folderPath) {
    std::vector<std::string> filePaths;

    if (!std::filesystem::exists(folderPath)) {
        std::cerr << "Error: Folder '" << folderPath << "' does not exist." << std::endl;
        return filePaths;
    }

    if (!std::filesystem::is_directory(folderPath)) {
        std::cerr << "Error: Path '" << folderPath << "' is not a directory." << std::endl;
        return filePaths;
    }

    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(folderPath)) {
            if (entry.is_regular_file() && entry.path().extension() == ".cl") {
                filePaths.push_back(entry.path().string());
            }
        }
    } catch (const std::filesystem::filesystem_error& ex) {
        std::cerr << "Filesystem error during traversal: " << ex.what() << std::endl;
    }

    std::sort(filePaths.begin(), filePaths.end());
    return filePaths;
}

OpenCLSetup OpenCLSetup::createOpenCLSetup(const std::string& kernelsPath){
    std::cout << "OpenCL Neural Network Example (Batched Training)\n";

    // --- Platform Discovery ---
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found. Please ensure OpenCL drivers are installed.");
    }
    cl::Platform platform = platforms[0];
    std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    // --- Device Discovery ---
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

    // --- Device Selection ---
    int device_index;
    if (devices.size() == 1) {
        std::cout << "Only one device found, using it by default.\n";
        device_index = 0;
    } else {
        std::cout << "Please select a device by entering its index (0 to " << devices.size() - 1 << "): ";
        std::cin >> device_index;
        if (std::cin.fail() || device_index < 0 || device_index >= devices.size()) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            throw std::runtime_error("Invalid input for device selection. Please enter a valid integer index within the range.");
        }
    }
    cl::Device device = devices[device_index];
    std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    std::vector<std::string> kernelFiles = getAllKernelFiles(kernelsPath);
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

    program.build({ device });
    
    return OpenCLSetup(context, queue, program);
}