#include <gtest/gtest.h>
#include "Utils/OpenCLResources.hpp"

TEST(OpenCLResourcesTest, InitializeContext) {
    Utils::OpenCLResources clRes = Utils::OpenCLResources::createOpenCLResources();
    bool ok = clRes.valid();
    EXPECT_TRUE(ok) << "OpenCL context should initialize successfully.";
}

TEST(OpenCLResourcesTest, GetDeviceCount) {
    Utils::OpenCLResources clRes = Utils::OpenCLResources::createOpenCLResources();
    cl::Context context = clRes.getContext();
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    size_t count = devices.size();
    EXPECT_GT(count, 0) << "Should detect at least one OpenCL device.";
}