// Local test main — runs the first test case
// Usage: ./batch_matmul_max_sum_custom

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "acl/acl.h"

#ifndef TENSOR_GROUP_INFO_DEFINED
#define TENSOR_GROUP_INFO_DEFINED
// Dtype enum for TensorInfo.dtype:
// 0=float32 1=float16 2=bfloat16 3=int8 4=int16 5=int32 6=int64
// 7=uint8 8=uint16 9=uint32 10=uint64 11=bool
struct TensorInfo {
    const int64_t* shape;
    int64_t numDims;
    int32_t dtype;
};
struct TensorGroupInfo {
    const TensorInfo* tensors;
    int64_t numTensors;
};
#endif

#include <vector>
#include "data_utils.h"

#define ACL_CHECK(expr) do { aclError _ret = (expr); if (_ret != 0) { fprintf(stderr, "[ERROR] %s failed at %s:%d, ret=%d\n", #expr, __FILE__, __LINE__, _ret); return 1; } } while(0)
#define FILE_CHECK(expr) do { if (!(expr)) { fprintf(stderr, "[ERROR] File I/O failed at %s:%d\n", __FILE__, __LINE__); return 1; } } while(0)

#include "kernel.asc"

int main()
{
    int32_t deviceId = 0;
    int64_t availableCoreNum = 0;
    aclrtStream stream = nullptr;

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(deviceId));
    ACL_CHECK(aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_CUBE_CORE_NUM, &availableCoreNum));
    if (availableCoreNum <= 0) {
        return 1;
    }
    ACL_CHECK(aclrtCreateStream(&stream));

    int64_t x1_0_shape[] = {1, 1, 32};
    TensorInfo x1_tensors[] = {{x1_0_shape, 3, 1}};
    TensorGroupInfo info_x1 = {x1_tensors, 1};
    int64_t x2_0_shape[] = {1, 32, 1};
    TensorInfo x2_tensors[] = {{x2_0_shape, 3, 1}};
    TensorGroupInfo info_x2 = {x2_tensors, 1};
    int64_t y_0_shape[] = {1};
    TensorInfo y_tensors[] = {{y_0_shape, 1, 0}};
    TensorGroupInfo info_y = {y_tensors, 1};

    bool transposeX1 = false;
    bool transposeX2 = false;

    uint8_t *inputHost_x1 = nullptr, *inputDevice_x1 = nullptr;
    uint8_t *inputHost_x2 = nullptr, *inputDevice_x2 = nullptr;

    uint8_t *outputHost_y = nullptr, *outputDevice_y = nullptr;

    ACL_CHECK(aclrtMallocHost((void **)&inputHost_x1, 64));
    ACL_CHECK(aclrtMalloc((void **)&inputDevice_x1, 64, ACL_MEM_MALLOC_HUGE_FIRST));
    FILE_CHECK(ReadFile("./input/x1.bin", 64, inputHost_x1, 64));
    ACL_CHECK(aclrtMemcpy(inputDevice_x1, 64, inputHost_x1, 64, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMallocHost((void **)&inputHost_x2, 64));
    ACL_CHECK(aclrtMalloc((void **)&inputDevice_x2, 64, ACL_MEM_MALLOC_HUGE_FIRST));
    FILE_CHECK(ReadFile("./input/x2.bin", 64, inputHost_x2, 64));
    ACL_CHECK(aclrtMemcpy(inputDevice_x2, 64, inputHost_x2, 64, ACL_MEMCPY_HOST_TO_DEVICE));

    ACL_CHECK(aclrtMallocHost((void **)&outputHost_y, 4));
    ACL_CHECK(aclrtMalloc((void **)&outputDevice_y, 4, ACL_MEM_MALLOC_HUGE_FIRST));

    run_kernel((GM_ADDR)inputDevice_x1, info_x1, (GM_ADDR)inputDevice_x2, info_x2, (GM_ADDR)outputDevice_y, info_y, availableCoreNum, stream, transposeX1, transposeX2);
    auto sync_ret_ = aclrtSynchronizeStreamWithTimeout(stream, 3000);
    if (sync_ret_ != 0) { fprintf(stderr, "[ERROR] aclrtSynchronizeStreamWithTimeout failed, ret=%d\n", sync_ret_); return 1; }

    ACL_CHECK(aclrtMemcpy(outputHost_y, 4, outputDevice_y, 4, ACL_MEMCPY_DEVICE_TO_HOST));
    FILE_CHECK(WriteFile("./output/y.bin", outputHost_y, 4));

    if (inputDevice_x1) aclrtFree(inputDevice_x1);
    if (inputHost_x1) aclrtFreeHost(inputHost_x1);
    if (inputDevice_x2) aclrtFree(inputDevice_x2);
    if (inputHost_x2) aclrtFreeHost(inputHost_x2);
    if (outputDevice_y) aclrtFree(outputDevice_y);
    if (outputHost_y) aclrtFreeHost(outputHost_y);

    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return 0;
}
