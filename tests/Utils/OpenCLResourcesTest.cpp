#include <gtest/gtest.h>
#include "Utils/OpenCLResources.hpp"

TEST(OpenCLResourcesTest, InitializeContext)
{
    Utils::OpenCLResources clRes = Utils::OpenCLResources::createOpenCLResources();
    bool clResValid = clRes.valid();
    EXPECT_TRUE(clResValid) << "OpenCL context should initialize successfully.";
}

TEST(OpenCLResourcesTest, GetDeviceCount)
{
    Utils::OpenCLResources clRes = Utils::OpenCLResources::createOpenCLResources();
    const cl::Context &context = clRes.getContext();
    std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
    size_t count = devices.size();
    EXPECT_GT(count, 0) << "Should detect at least one OpenCL device.";
}