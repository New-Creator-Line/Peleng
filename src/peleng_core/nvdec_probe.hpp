#pragma once

#include <string>

#if defined(PELENG_HAS_NVDEC_SDK) && PELENG_HAS_NVDEC_SDK
#include <cuda.h>
#include <cuviddec.h>
#include <nvcuvid.h>
#include <nvEncodeAPI.h>
#endif

namespace peleng {

struct NvdecCapabilities {
    bool sdk_linked{false};
    std::string sdk_version{"13.1.24"};
    bool h264_supported{false};
    bool hevc_supported{false};
    int num_nvdec_engines{0};
    unsigned int max_width{0};
    unsigned int max_height{0};
    unsigned short min_width{0};
    unsigned short min_height{0};
    unsigned short output_format_mask{0};
};

inline NvdecCapabilities query_nvdec_capabilities() {
    NvdecCapabilities out{};
#if defined(PELENG_HAS_NVDEC_SDK) && PELENG_HAS_NVDEC_SDK
    out.sdk_linked = true;
    if (cuInit(0) != CUDA_SUCCESS) {
        return out;
    }

    CUcontext current_ctx = nullptr;
    cuCtxGetCurrent(&current_ctx);
    CUcontext created_ctx = nullptr;
    if (current_ctx == nullptr) {
        CUdevice dev = 0;
        if (cuDeviceGet(&dev, 0) == CUDA_SUCCESS) {
            cuDevicePrimaryCtxRetain(&created_ctx, dev);
            cuCtxPushCurrent(created_ctx);
        }
    }

    CUVIDDECODECAPS h264_caps{};
    h264_caps.eCodecType = cudaVideoCodec_H264;
    h264_caps.eChromaFormat = cudaVideoChromaFormat_420;
    h264_caps.nBitDepthMinus8 = 0;

    if (cuvidGetDecoderCaps(&h264_caps) == CUDA_SUCCESS) {
        out.h264_supported = (h264_caps.bIsSupported != 0);
        out.num_nvdec_engines = static_cast<int>(h264_caps.nNumNVDECs);
        out.max_width = h264_caps.nMaxWidth;
        out.max_height = h264_caps.nMaxHeight;
        out.min_width = h264_caps.nMinWidth;
        out.min_height = h264_caps.nMinHeight;
        out.output_format_mask = h264_caps.nOutputFormatMask;
    }

    CUVIDDECODECAPS hevc_caps{};
    hevc_caps.eCodecType = cudaVideoCodec_HEVC;
    hevc_caps.eChromaFormat = cudaVideoChromaFormat_420;
    hevc_caps.nBitDepthMinus8 = 0;
    if (cuvidGetDecoderCaps(&hevc_caps) == CUDA_SUCCESS) {
        out.hevc_supported = (hevc_caps.bIsSupported != 0);
    }

    if (created_ctx != nullptr) {
        CUcontext popped = nullptr;
        cuCtxPopCurrent(&popped);
        CUdevice dev = 0;
        if (cuDeviceGet(&dev, 0) == CUDA_SUCCESS) {
            cuDevicePrimaryCtxRelease(dev);
        }
    }
#endif
    return out;
}

} // namespace peleng
