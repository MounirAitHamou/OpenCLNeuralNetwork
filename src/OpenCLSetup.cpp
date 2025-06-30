#include "OpenCLSetup.hpp"

OpenCLSetup::OpenCLSetup() : context(nullptr), queue(nullptr), program(nullptr) {}

OpenCLSetup::OpenCLSetup(cl::Context context, cl::CommandQueue queue, cl::Program program)
    : context(context), queue(queue), program(program) {}

OpenCLSetup OpenCLSetup::createOpenCLSetup(const std::string& kernel_file) {
    std::cout << "OpenCL Neural Network Example (Batched Training)\n";

    // OpenCL Setup
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        throw std::runtime_error("No OpenCL platforms found.");
    }

    cl::Platform platform = platforms[0];
    std::cout << "Using platform: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_GPU, &devices);
    if (devices.empty()) {
        platform.getDevices(CL_DEVICE_TYPE_CPU, &devices);
        if (devices.empty()) {
            throw std::runtime_error("No OpenCL devices found.");
        }
    }
    
    std::cout << "Total devices found: " << devices.size() << "\n";
    std::cout << "Available devices:\n";

    for (int i = 0; i < devices.size(); ++i) {
        std::cout << "Device " << i << ": ";
        std::cout << "Type: " << (devices[i].getInfo<CL_DEVICE_TYPE>() == CL_DEVICE_TYPE_GPU ? "GPU" : "Other") << ", ";
        std::cout << "Vendor: " << devices[i].getInfo<CL_DEVICE_VENDOR>() << ", ";
        std::cout << "Version: " << devices[i].getInfo<CL_DEVICE_VERSION>() << ", ";
        std::cout << devices[i].getInfo<CL_DEVICE_NAME>() << "\n";
    }

    std::cout << "Please select a device by entering its index (0 to " << devices.size() - 1 << "): ";
    int device_index;
    std::cin >> device_index;
    if (std::cin.fail() || device_index < 0 || device_index >= devices.size()) {
        throw std::runtime_error("Invalid input for device selection. Please enter a valid integer index within the range.");
    }

    cl::Device device = devices[device_index];
    std::cout << "Selected device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
    std::cout << "Using device: " << device.getInfo<CL_DEVICE_NAME>() << "\n";

    cl::Context context(device);
    cl::CommandQueue queue(context, device);

    std::ifstream kernelFile(kernel_file);
    if (!kernelFile.is_open()) {
        std::cerr << "Error: Could not open 'kernels.cl' kernel file. Make sure it's in the kernels directory." << std::endl;
        throw std::runtime_error("Kernel file not found.");
    }
    std::string src(std::istreambuf_iterator<char>(kernelFile), {});
    cl::Program program(context, cl::Program::Sources(1, {src.c_str(), src.length()}));


    program.build({device});
    

    return OpenCLSetup(context, queue, program);
}