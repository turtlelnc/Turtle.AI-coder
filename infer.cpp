// infer.cpp - GGUF格式量化模型推理引擎（支持Q3_K, Q4_0, Q4_1, Q4_K_M, Q5_K_M, Q6_K, Q8_0, FP16, FP32）
// 依赖：nlohmann/json.hpp
// 编译：g++ -std=c++17 -O3 -march=native -pthread infer.cpp -o infer.exe

#define NOMINMAX
#include <queue>
#include <tuple>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include"omp.h"
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#undef min
#undef max
#endif

#include "json.hpp"
using json = nlohmann::json;

namespace infer {
    enum class DataType {
        FP32, FP16, INT8, INT4, Q4_0, Q4_1, Q8_0, Q4_K, Q3_K,
        Q5_K_M, Q6_K, Q4_K_M
    };

    struct Shape {
        std::vector<size_t> dims;
        Shape() = default;
        Shape(std::initializer_list<size_t> dims_) : dims(dims_) {}
        size_t elements() const {
            size_t n = 1;
            for (auto d : dims) n *= d;
            return n;
        }
        size_t operator[](size_t i) const { return dims[i]; }
        size_t& operator[](size_t i) { return dims[i]; }
    };

    class Tensor {
    public:
        Tensor() : dtype_(DataType::FP32), bytes_(0) {}
        Tensor(DataType dtype, const Shape& shape) : dtype_(dtype), shape_(shape) {
            bytes_ = shape_.elements() * dtype_size(dtype);
            data_ = std::unique_ptr<uint8_t[]>(new uint8_t[bytes_]);
        }
        Tensor(DataType dtype, const Shape& shape, const uint8_t* external, size_t sz)
            : dtype_(dtype), shape_(shape), bytes_(sz) {
            data_ = std::unique_ptr<uint8_t[]>(new uint8_t[sz]);
            std::memcpy(data_.get(), external, sz);
        }
        Tensor(Tensor&&) = default;
        Tensor& operator=(Tensor&&) = default;
        DataType dtype() const { return dtype_; }
        const Shape& shape() const { return shape_; }
        size_t bytes() const { return bytes_; }
        size_t elements() const { return shape_.elements(); }
        template<typename T> T* data() { return reinterpret_cast<T*>(data_.get()); }
        template<typename T> const T* data() const { return reinterpret_cast<const T*>(data_.get()); }
        uint8_t* raw_data() { return data_.get(); }
        const uint8_t* raw_data() const { return data_.get(); }
        static size_t dtype_size(DataType dtype) {
            switch (dtype) {
            case DataType::FP32: return 4;
            case DataType::FP16: return 2;
            case DataType::INT8: return 1;
            case DataType::INT4: return 1;
            default: return 0;
            }
        }
    private:
        DataType dtype_;
        Shape shape_;
        size_t bytes_;
        std::unique_ptr<uint8_t[]> data_;
    };

    // FP16 <-> FP32
    inline float fp16_to_fp32(uint16_t h) {
        const uint32_t sign = (h & 0x8000) << 16;
        const uint32_t exp = (h & 0x7C00) >> 10;
        const uint32_t mant = (h & 0x03FF) << 13;
        if (exp == 0) {
            if (mant == 0) return sign ? -0.0f : 0.0f;
            uint32_t m = mant, e = 0;
            while ((m & 0x00800000) == 0) { m <<= 1; e++; }
            m &= 0x007FFFFF;
            uint32_t new_exp = 127 - 15 - e;
            uint32_t bits = sign | (new_exp << 23) | m;
            float f; std::memcpy(&f, &bits, sizeof(f)); return f;
        }
        else if (exp == 0x1F) {
            uint32_t bits = sign | 0x7F800000 | mant;
            float f; std::memcpy(&f, &bits, sizeof(f)); return f;
        }
        else {
            uint32_t new_exp = exp - 15 + 127;
            uint32_t bits = sign | (new_exp << 23) | mant;
            float f; std::memcpy(&f, &bits, sizeof(f)); return f;
        }
    }

    void dequantize_q4_0_row(const uint8_t* packed, float* dst, size_t n) {
        size_t nb = (n + 31) / 32;
        for (size_t b = 0; b < nb; ++b) {
            uint16_t scale_f16; std::memcpy(&scale_f16, packed, 2);
            float scale = fp16_to_fp32(scale_f16);
            packed += 2;
            size_t blk = std::min<size_t>(32, n - b * 32);
            for (size_t i = 0; i < blk; ++i) {
                uint8_t byte = packed[i / 2];
                uint8_t val = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                dst[b * 32 + i] = (static_cast<float>(static_cast<int8_t>(val << 4) >> 4)) * scale;
            }
            packed += 16;
        }
    }

    void dequantize_q4_1_row(const uint8_t* packed, float* dst, size_t n) {
        size_t nb = (n + 31) / 32;
        for (size_t b = 0; b < nb; ++b) {
            uint16_t scale_f16, zp_f16;
            std::memcpy(&scale_f16, packed, 2);
            std::memcpy(&zp_f16, packed + 2, 2);
            float scale = fp16_to_fp32(scale_f16), zp = fp16_to_fp32(zp_f16);
            packed += 4;
            size_t blk = std::min<size_t>(32, n - b * 32);
            for (size_t i = 0; i < blk; ++i) {
                uint8_t byte = packed[i / 2];
                uint8_t val = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                dst[b * 32 + i] = (static_cast<float>(val) - zp) * scale;
            }
            packed += 16;
        }
    }

    namespace ops {

        void dequantize_q8_0_row(const uint8_t* packed, float* dst, size_t n) {
            size_t nb = (n + 31) / 32;
            for (size_t b = 0; b < nb; ++b) {
                uint16_t scale_f16;
                std::memcpy(&scale_f16, packed, 2);
                float scale = fp16_to_fp32(scale_f16);
                packed += 2;
                size_t blk = std::min<size_t>(32, n - b * 32);
                for (size_t i = 0; i < blk; ++i) {
                    int8_t val = static_cast<int8_t>(packed[i]);
                    dst[b * 32 + i] = static_cast<float>(val) * scale;
                }
                packed += 32;
            }
        }

        void dequantize_q3_K_row(const uint8_t* packed, float* dst, int64_t n) {
            const int nb = static_cast<int>((n + 255) / 256);
            for (int b = 0; b < nb; ++b) {
                const uint8_t* scales = packed;
                const uint8_t* q3_scale = packed + 12;
                packed += 16;

                float d_all = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(q3_scale));

                size_t blk_size = std::min<size_t>(256, static_cast<size_t>(n - b * 256));
                for (int i = 0; i < 16 && i * 16 < static_cast<int>(blk_size); ++i) {
                    uint8_t s6 = (scales[i / 2] >> (4 * (i % 2))) & 0x3F;
                    float d = d_all * (s6 - 32) / 64.0f;

                    size_t sub_blk = std::min<size_t>(16, blk_size - i * 16);
                    for (size_t j = 0; j < sub_blk; ++j) {
                        int bit_offset = static_cast<int>(j * 3);
                        int byte_idx = bit_offset / 8;
                        int bit_idx = bit_offset % 8;

                        uint8_t val = 0;
                        if (bit_idx + 3 <= 8) {
                            val = (packed[byte_idx] >> bit_idx) & 0x07;
                        }
                        else {
                            val = (packed[byte_idx] >> bit_idx) & ((1 << (8 - bit_idx)) - 1);
                            val |= (packed[byte_idx + 1] << (8 - bit_idx)) & 0x07;
                        }
                        dst[b * 256 + i * 16 + j] = (static_cast<float>(val) - 4) * d;
                    }
                    packed += 6;
                }
            }
        }

        void dequantize_q4_K_row(const uint8_t* packed, float* dst, int64_t n) {
            const int nb = static_cast<int>((n + 255) / 256);
            const uint8_t* const start = packed;
            
            
            // 计算预期的总字节数
            // Q4_K 格式：每个 256 元素的块需要 152 字节
            size_t expected_bytes = nb * 144;
            
            for (int b = 0; b < nb; ++b) {
                // 检查是否有足够的数据读取 scales (12 字节) 和 q3_scale 信息
                // 注意：packed 可能不是以 block 为单位对齐的，但我们需要确保有至少 24 字节
                
                const uint8_t* scales = packed;
                const uint8_t* mins = packed + 12;
                packed += 24;  // 跳过 scales 和 mins 区域（各12字节）
        
                size_t blk_size = std::min<size_t>(256, static_cast<size_t>(n - b * 256));
                
                // 每个 block 有最多 16 个子块（每个子块处理 16 个元素）
                for (int i = 0; i < 16 && i * 16 < static_cast<int>(blk_size); ++i) {
                    // 读取 scale 和 min (FP16)
                    uint16_t scale_f16, min_f16;
                    std::memcpy(&scale_f16, scales + i * 2, 2);
                    std::memcpy(&min_f16, mins + i * 2, 2);
        
                    float d = fp16_to_fp32(scale_f16);
                    float m = fp16_to_fp32(min_f16);
        
                    size_t sub_blk = std::min<size_t>(16, blk_size - i * 16);
                    
                    // 关键检查：确保有 8 字节的量化数据可读
                    // 每个子块包含 16 个 4-bit 值 = 8 字节
                    // 但我们无法在函数内检查边界，因为不知道 packed 的总大小
                    // 所以依赖调用者保证传入的数据足够大
                    
                    for (size_t j = 0; j < sub_blk; ++j) {
                        // 每个字节包含 2 个 4-bit 值
                        size_t byte_idx = j / 2;
                        int shift = (j % 2) * 4;
                        
                        // 注意：这里假设 packed 至少还有 byte_idx + 1 字节可读
                        uint8_t v = (packed[byte_idx] >> shift) & 0x0F;
                        dst[b * 256 + i * 16 + j] = d * (static_cast<float>(v) - m);
                    }
                    packed += 8;  // 每个子块消耗 8 字节
                }
                
            }
        }
        void dequantize_q5_K_M_row(const uint8_t* packed, float* dst, int64_t n) {
            const int nb = static_cast<int>((n + 255) / 256);
            for (int b = 0; b < nb; ++b) {
                const uint8_t* scales_ptr = packed;
                const uint8_t* mins_ptr = packed + 12;
                packed += 24;

                size_t blk_size = std::min<size_t>(256, static_cast<size_t>(n - b * 256));
                for (int i = 0; i < 8 && i * 32 < static_cast<int>(blk_size); ++i) {
                    uint16_t scale_f16, min_f16;
                    std::memcpy(&scale_f16, scales_ptr + i * 2, 2);
                    std::memcpy(&min_f16, mins_ptr + i * 2, 2);

                    float scale = fp16_to_fp32(scale_f16);
                    float min_val = fp16_to_fp32(min_f16);

                    size_t sub_blk = std::min<size_t>(32, blk_size - i * 32);
                    for (size_t j = 0; j < sub_blk; ++j) {
                        int bit_offset = static_cast<int>(j * 5);
                        int byte_idx = bit_offset / 8;
                        int bit_idx = bit_offset % 8;

                        uint16_t val = 0;
                        if (bit_idx + 5 <= 8) {
                            val = (packed[byte_idx] >> bit_idx) & 0x1F;
                        }
                        else {
                            val = (packed[byte_idx] >> bit_idx) & ((1 << (8 - bit_idx)) - 1);
                            val |= (packed[byte_idx + 1] << (8 - bit_idx)) & 0x1F;
                        }
                        dst[b * 256 + i * 32 + j] = (static_cast<float>(val) - min_val) * scale;
                    }
                    packed += 20;
                }
            }
        }

        void dequantize_q6_K_row(const uint8_t* packed, float* dst, int64_t n) {
            const int nb = static_cast<int>((n + 255) / 256);
            for (int b = 0; b < nb; ++b) {
                const uint8_t* scales_ptr = packed;
                const uint8_t* mins_ptr = packed + 12;
                packed += 24;

                size_t blk_size = std::min<size_t>(256, static_cast<size_t>(n - b * 256));
                for (int i = 0; i < 16 && i * 16 < static_cast<int>(blk_size); ++i) {
                    uint16_t scale_f16, min_f16;
                    std::memcpy(&scale_f16, scales_ptr + i * 2, 2);
                    std::memcpy(&min_f16, mins_ptr + i * 2, 2);

                    float scale = fp16_to_fp32(scale_f16);
                    float min_val = fp16_to_fp32(min_f16);

                    size_t sub_blk = std::min<size_t>(16, blk_size - i * 16);
                    for (size_t j = 0; j < sub_blk; ++j) {
                        int bit_offset = static_cast<int>(j * 6);
                        int byte_idx = bit_offset / 8;
                        int bit_idx = bit_offset % 8;

                        uint16_t val = 0;
                        if (bit_idx + 6 <= 8) {
                            val = (packed[byte_idx] >> bit_idx) & 0x3F;
                        }
                        else {
                            val = (packed[byte_idx] >> bit_idx) & ((1 << (8 - bit_idx)) - 1);
                            val |= (packed[byte_idx + 1] << (8 - bit_idx)) & 0x3F;
                        }
                        dst[b * 256 + i * 16 + j] = (static_cast<float>(val) - min_val) * scale;
                    }
                    packed += 12;
                }
            }
        }

        void gemm_fp32(bool transA, bool transB, int M, int N, int K, float alpha,
            const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
            #pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0;
                    for (int k = 0; k < K; ++k) {
                        float a = transA ? A[k * lda + i] : A[i * lda + k];
                        float b = transB ? B[j * ldb + k] : B[k * ldb + j];
                        sum += a * b;
                    }
                    if (beta == 0.0f)
                        C[i * ldc + j] = alpha * sum;
                    else
                        C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                }
            }
        }

        void gemm_fp16_weight(bool transB, int M, int N, int K, float alpha, const float* A, int lda,
            const uint16_t* B, int ldb, float beta, float* C, int ldc) {
            for (int i = 0; i < M; ++i) {
                for (int j = 0; j < N; ++j) {
                    float sum = 0;
                    for (int k = 0; k < K; ++k) {
                        float b_val = transB ? fp16_to_fp32(B[j * ldb + k]) : fp16_to_fp32(B[k * ldb + j]);
                        sum += A[i * lda + k] * b_val;
                    }
                    if (beta == 0.0f)
                        C[i * ldc + j] = alpha * sum;
                    else
                        C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                }
            }
        }

        void gemm_q4_0_weight(bool transB, int M, int N, int K, float alpha, const float* A, int lda,
            const uint8_t* B, float beta, float* C, int ldc) {
            if (transB) {
                size_t stride = ((K + 31) / 32) * 18;
                std::vector<float> B_row(K);
                for (int j = 0; j < N; ++j) {
                    dequantize_q4_0_row(B + j * stride, B_row.data(), K);
                    for (int i = 0; i < M; ++i) {
                        float sum = 0;
                        for (int k = 0; k < K; ++k) sum += A[i * lda + k] * B_row[k];
                        if (beta == 0.0f)
                            C[i * ldc + j] = alpha * sum;
                        else
                            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    }
                }
            }
            else {
                size_t stride = ((N + 31) / 32) * 18;
                std::vector<float> B_row(N);
                if (beta == 0.0f) {
                    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;
                }
                else {
                    for (int i = 0; i < M * N; ++i) C[i] *= beta;
                }
                for (int k = 0; k < K; ++k) {
                    dequantize_q4_0_row(B + k * stride, B_row.data(), N);
                    for (int i = 0; i < M; ++i) {
                        float a = alpha * A[i * lda + k];
                        for (int j = 0; j < N; ++j) C[i * ldc + j] += a * B_row[j];
                    }
                }
            }
        }

        void gemm_q4_1_weight(bool transB, int M, int N, int K, float alpha, const float* A, int lda,
            const uint8_t* B, float beta, float* C, int ldc) {
            if (transB) {
                size_t stride = ((K + 31) / 32) * 20;
                std::vector<float> B_row(K);
                for (int j = 0; j < N; ++j) {
                    dequantize_q4_1_row(B + j * stride, B_row.data(), K);
                    for (int i = 0; i < M; ++i) {
                        float sum = 0;
                        for (int k = 0; k < K; ++k) sum += A[i * lda + k] * B_row[k];
                        if (beta == 0.0f)
                            C[i * ldc + j] = alpha * sum;
                        else
                            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    }
                }
            }
            else {
                size_t stride = ((N + 31) / 32) * 20;
                std::vector<float> B_row(N);
                if (beta == 0.0f) {
                    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;
                }
                else {
                    for (int i = 0; i < M * N; ++i) C[i] *= beta;
                }
                for (int k = 0; k < K; ++k) {
                    dequantize_q4_1_row(B + k * stride, B_row.data(), N);
                    for (int i = 0; i < M; ++i) {
                        float a = alpha * A[i * lda + k];
                        for (int j = 0; j < N; ++j) C[i * ldc + j] += a * B_row[j];
                    }
                }
            }
        }

        void gemm_q8_0_weight(bool transB, int M, int N, int K, float alpha, const float* A, int lda,
            const uint8_t* B, float beta, float* C, int ldc) {
            if (transB) {
                size_t stride = ((K + 31) / 32) * 34;
                std::vector<float> B_row(K);
                for (int j = 0; j < N; ++j) {
                    dequantize_q8_0_row(B + j * stride, B_row.data(), K);
                    for (int i = 0; i < M; ++i) {
                        float sum = 0;
                        for (int k = 0; k < K; ++k) sum += A[i * lda + k] * B_row[k];
                        if (beta == 0.0f)
                            C[i * ldc + j] = alpha * sum;
                        else
                            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    }
                }
            }
            else {
                size_t stride = ((N + 31) / 32) * 34;
                std::vector<float> B_row(N);
                if (beta == 0.0f) {
                    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;
                }
                else {
                    for (int i = 0; i < M * N; ++i) C[i] *= beta;
                }
                for (int k = 0; k < K; ++k) {
                    dequantize_q8_0_row(B + k * stride, B_row.data(), N);
                    for (int i = 0; i < M; ++i) {
                        float a = alpha * A[i * lda + k];
                        for (int j = 0; j < N; ++j) C[i * ldc + j] += a * B_row[j];
                    }
                }
            }
        }

        void gemm_q3_K_weight(bool transB, int M, int N, int K, float alpha, const float* A, int lda,
            const uint8_t* B, float beta, float* C, int ldc) {
            if (transB) {
                size_t stride = ((K + 255) / 256) * 170;
                std::vector<float> B_row(K);
                for (int j = 0; j < N; ++j) {
                    dequantize_q3_K_row(B + j * stride, B_row.data(), K);
                    for (int i = 0; i < M; ++i) {
                        float sum = 0;
                        for (int k = 0; k < K; ++k) sum += A[i * lda + k] * B_row[k];
                        if (beta == 0.0f)
                            C[i * ldc + j] = alpha * sum;
                        else
                            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    }
                }
            }
            else {
                size_t stride = ((N + 255) / 256) * 170;
                std::vector<float> B_row(N);
                if (beta == 0.0f) {
                    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;
                }
                else {
                    for (int i = 0; i < M * N; ++i) C[i] *= beta;
                }
                for (int k = 0; k < K; ++k) {
                    dequantize_q3_K_row(B + k * stride, B_row.data(), N);
                    for (int i = 0; i < M; ++i) {
                        float a = alpha * A[i * lda + k];
                        for (int j = 0; j < N; ++j) C[i * ldc + j] += a * B_row[j];
                    }
                }
            }
        }

        void gemm_q4_K_weight(bool transB, int M, int N, int K, float alpha, const float* A, int lda,
            const uint8_t* B, float beta, float* C, int ldc) {
            if (transB) {
                size_t stride = ((K + 255) / 256) * 144;
                std::vector<float> B_row(K);
                for (int j = 0; j < N; ++j) {
                    dequantize_q4_K_row(B + j * stride, B_row.data(), K);
                    for (int i = 0; i < M; ++i) {
                        float sum = 0;
                        for (int k = 0; k < K; ++k) sum += A[i * lda + k] * B_row[k];
                        if (beta == 0.0f)
                            C[i * ldc + j] = alpha * sum;
                        else
                            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    }
                }
            }
            else {
                size_t stride = ((N + 255) / 256) * 144;
                std::vector<float> B_row(N);
                if (beta == 0.0f) {
                    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;
                }
                else {
                    for (int i = 0; i < M * N; ++i) C[i] *= beta;
                }
                for (int k = 0; k < K; ++k) {
                    dequantize_q4_K_row(B + k * stride, B_row.data(), N);
                    for (int i = 0; i < M; ++i) {
                        float a = alpha * A[i * lda + k];
                        for (int j = 0; j < N; ++j) C[i * ldc + j] += a * B_row[j];
                    }
                }
            }
        }

        void gemm_q5_K_M_weight(bool transB, int M, int N, int K, float alpha, const float* A, int lda,
            const uint8_t* B, float beta, float* C, int ldc) {
            if (transB) {
                size_t stride = ((K + 255) / 256) * 184;
                std::vector<float> B_row(K);
                for (int j = 0; j < N; ++j) {
                    dequantize_q5_K_M_row(B + j * stride, B_row.data(), K);
                    for (int i = 0; i < M; ++i) {
                        float sum = 0;
                        for (int k = 0; k < K; ++k) sum += A[i * lda + k] * B_row[k];
                        if (beta == 0.0f)
                            C[i * ldc + j] = alpha * sum;
                        else
                            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    }
                }
            }
            else {
                size_t stride = ((N + 255) / 256) * 184;
                std::vector<float> B_row(N);
                if (beta == 0.0f) {
                    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;
                }
                else {
                    for (int i = 0; i < M * N; ++i) C[i] *= beta;
                }
                for (int k = 0; k < K; ++k) {
                    dequantize_q5_K_M_row(B + k * stride, B_row.data(), N);
                    for (int i = 0; i < M; ++i) {
                        float a = alpha * A[i * lda + k];
                        for (int j = 0; j < N; ++j) C[i * ldc + j] += a * B_row[j];
                    }
                }
            }
        }

        void gemm_q6_K_weight(bool transB, int M, int N, int K, float alpha, const float* A, int lda,
            const uint8_t* B, float beta, float* C, int ldc) {
            if (transB) {
                size_t stride = ((K + 255) / 256) * 216;
                std::vector<float> B_row(K);
                for (int j = 0; j < N; ++j) {
                    dequantize_q6_K_row(B + j * stride, B_row.data(), K);
                    for (int i = 0; i < M; ++i) {
                        float sum = 0;
                        for (int k = 0; k < K; ++k) sum += A[i * lda + k] * B_row[k];
                        if (beta == 0.0f)
                            C[i * ldc + j] = alpha * sum;
                        else
                            C[i * ldc + j] = alpha * sum + beta * C[i * ldc + j];
                    }
                }
            }
            else {
                size_t stride = ((N + 255) / 256) * 216;
                std::vector<float> B_row(N);
                if (beta == 0.0f) {
                    for (int i = 0; i < M * N; ++i) C[i] = 0.0f;
                }
                else {
                    for (int i = 0; i < M * N; ++i) C[i] *= beta;
                }
                for (int k = 0; k < K; ++k) {
                    dequantize_q6_K_row(B + k * stride, B_row.data(), N);
                    for (int i = 0; i < M; ++i) {
                        float a = alpha * A[i * lda + k];
                        for (int j = 0; j < N; ++j) C[i * ldc + j] += a * B_row[j];
                    }
                }
            }
        }

        void axpy(int n, float a, const float* x, float* y) {
            for (int i = 0;i < n;++i) y[i] += a * x[i];
        }

        void rms_norm(int n, const float* x, const float* w, float eps, float* y) {
            float ss = 0;
            for (int i = 0;i < n;++i) ss += x[i] * x[i];
            float ir = 1.0f / std::sqrt(ss / n + eps);
            for (int i = 0;i < n;++i) y[i] = x[i] * ir * w[i];
        }

        void softmax(float* x, int n) {
            float m = *std::max_element(x, x + n), sum = 0;
            for (int i = 0;i < n;++i) { x[i] = std::exp(x[i] - m); sum += x[i]; }
            for (int i = 0;i < n;++i) x[i] /= sum;
        }

        void apply_rope(int n_heads, int n_kv_heads, int head_dim, int pos, float* q, float* k) {
            for (int h = 0; h < n_heads; ++h) {
                float* qh = q + h * head_dim;
                for (int i = 0; i < head_dim; i += 2) {
                    float theta = 1.0f / std::pow(10000.0f, static_cast<float>(i) / head_dim);
                    float c = std::cos(pos * theta), s = std::sin(pos * theta);
                    float q0 = qh[i], q1 = qh[i + 1];
                    qh[i] = q0 * c - q1 * s;
                    qh[i + 1] = q1 * c + q0 * s;
                }
            }
            for (int h = 0; h < n_kv_heads; ++h) {
                float* kh = k + h * head_dim;
                for (int i = 0; i < head_dim; i += 2) {
                    float theta = 1.0f / std::pow(10000.0f, static_cast<float>(i) / head_dim);
                    float c = std::cos(pos * theta), s = std::sin(pos * theta);
                    float k0 = kh[i], k1 = kh[i + 1];
                    kh[i] = k0 * c - k1 * s;
                    kh[i + 1] = k1 * c + k0 * s;
                }
            }
        }

    } // namespace ops

    struct ModelConfig {
        int dim = 0, n_layers = 32, n_heads = 32, n_kv_heads = 32, hidden_dim = 11008, vocab_size = 32000, max_seq_len = 2048;
        float norm_eps = 1e-5f;
        int head_dim() const { return dim / n_heads; }
        int kv_dim() const { return (dim * n_kv_heads) / n_heads; }
    };

    class GGUFLoader {
    public:
        struct GGUFTensorInfo {
            std::string name;
            uint32_t type;
            std::vector<uint64_t> dimensions;
            uint64_t offset;
            uint64_t size;
            uint64_t num_weights;
        };
        // 在 GGUFLoader 类的 public 区域添加以下方法

        // 获取词汇表
        std::vector<std::string> get_vocab() const {
            auto it = metadata_.find("tokenizer.ggml.tokens");
            if (it == metadata_.end() || it->second.type != GGUFValue::ARR) return {};
            std::vector<std::string> vocab;
            for (const auto& v : it->second.arr) {
                if (v.type == GGUFValue::STR) vocab.push_back(v.str);
            }
            return vocab;
        }

        // 获取 BPE 合并规则（merges）
        std::vector<std::pair<std::string, std::string>> get_merges() const {
            std::vector<std::pair<std::string, std::string>> merges;
            auto it = metadata_.find("tokenizer.ggml.merges");
            if (it == metadata_.end() || it->second.type != GGUFValue::ARR) return merges;
            
            for (const auto& v : it->second.arr) {
                if (v.type == GGUFValue::STR) {
                    std::string rule = v.str;
                    size_t space_pos = rule.find(' ');
                    if (space_pos != std::string::npos) {
                        std::string a = rule.substr(0, space_pos);
                        std::string b = rule.substr(space_pos + 1);
                        merges.emplace_back(a, b);
                    }
                }
            }
            return merges;
        }

        // 获取特殊 Token ID
        int get_eos_token_id() const {
            auto it = metadata_.find("tokenizer.ggml.eos_token_id");
            if (it != metadata_.end() && it->second.type == GGUFValue::U32) return it->second.u32;
            return -1;
        }
        int get_bos_token_id() const {
            auto it = metadata_.find("tokenizer.ggml.bos_token_id");
            if (it != metadata_.end() && it->second.type == GGUFValue::U32) return it->second.u32;
            return -1;
        }
        bool load(const std::string& path) {
            #ifdef _WIN32
                int wlen = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
                std::wstring wpath(wlen, 0);
                MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &wpath[0], wlen);
                file_.open(wpath.c_str(), std::ios::binary);
                if (!file_) return false;
            #else
                file_.open(path, std::ios::binary);
                if (!file_) return false;
            #endif
            
            if (!read_header()) return false;
            if (!read_metadata()) return false;
            if (!read_tensor_infos()) return false;

            if (!tensor_infos_.empty()) {
                tensor_data_start_offset_ = tensor_infos_[0].offset;
            }
            else {
                tensor_data_start_offset_ = 0;
            }
            file_.seekg(tensor_data_start_offset_);

            file_.seekg(0, std::ios::end);
            size_t file_size = file_.tellg();
            size_t data_size = file_size - tensor_data_start_offset_;
            file_.seekg(tensor_data_start_offset_);
            tensor_data_.resize(data_size);
            file_.read(reinterpret_cast<char*>(tensor_data_.data()), data_size);

            for (size_t i = 0; i < tensor_infos_.size(); ++i) {
                auto& t = tensor_infos_[i];
                if (i + 1 < tensor_infos_.size()) {
                    t.size = tensor_infos_[i + 1].offset - t.offset;
                }
                else {
                    t.size = file_size - t.offset;
                }
            }
            return true;
        }

        const std::vector<GGUFTensorInfo>& tensor_infos() const { return tensor_infos_; }

        const GGUFTensorInfo* find_tensor(const std::string& name) const {
            for (const auto& t : tensor_infos_) {
                if (t.name == name) return &t;
            }
            return nullptr;
        }
        int get_unknown_token_id() const {
            auto it = metadata_.find("tokenizer.ggml.unknown_token_id");
            if (it != metadata_.end() && it->second.type == GGUFValue::U32) return it->second.u32;
            return -1;
        }
        std::unique_ptr<Tensor> load_tensor(const GGUFTensorInfo& info) {
            Shape sh;
            for (auto d : info.dimensions) sh.dims.push_back(static_cast<size_t>(d));
            DataType dt = gguf_to_dtype(info.type);

            size_t offset_in_data = static_cast<size_t>(info.offset - tensor_data_start_offset_);
            size_t required_size = static_cast<size_t>(info.size);

            if (offset_in_data + required_size > tensor_data_.size()) {
                std::cerr << "ERROR: Tensor '" << info.name << "' out of bounds" << std::endl;
                return nullptr;
            }

            const uint8_t* src = tensor_data_.data() + offset_in_data;
            return std::make_unique<Tensor>(dt, sh, src, required_size);
        }

        template<typename T> T get_metadata(const std::string& key, T def) const {
            auto it = metadata_.find(key);
            if (it == metadata_.end()) return def;
            return extract_value<T>(it->second);
        }

    private:
        struct GGUFValue {
            enum Type { U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64 } type;
            union { uint8_t u8; int8_t i8; uint16_t u16; int16_t i16; uint32_t u32; int32_t i32; float f32; bool b; uint64_t u64; int64_t i64; double f64; };
            std::string str;
            std::vector<GGUFValue> arr;
        };

        std::ifstream file_;
        uint32_t version_;
        uint64_t tensor_count_, metadata_kv_count_;
        std::unordered_map<std::string, GGUFValue> metadata_;
        std::vector<GGUFTensorInfo> tensor_infos_;
        std::vector<uint8_t> tensor_data_;
        uint64_t tensor_data_start_offset_ = 0;

        DataType gguf_to_dtype(uint32_t t) const {
            DataType result;
            switch (t) {
            case 0:  result = DataType::FP32; break;
            case 1:  result = DataType::FP16; break;
            case 2:  result = DataType::Q4_0; break;
            case 3:  result = DataType::Q4_1; break;
            case 6:  result = DataType::Q5_K_M; break;
            case 7:  result = DataType::Q6_K; break;
            case 8:  result = DataType::Q8_0; break;
            case 10: result = DataType::Q4_K; break;   // GGUF 10 = Q4_K
            case 11: result = DataType::Q3_K; break;   // GGUF 11 = Q3_K
            case 12: result = DataType::Q4_K_M; break; // GGUF 12 = Q4_K_M  <-- 添加这行！
            case 14: result = DataType::Q4_K_M; break;
            default:
                std::cerr << "Unknown GGUF type: " << t << std::endl;
                result = DataType::FP32;
            }
            return result;
        }

        bool read_header() {
            uint32_t magic;
            file_.read(reinterpret_cast<char*>(&magic), 4);
            if (magic != 0x46554747) return false;
            file_.read(reinterpret_cast<char*>(&version_), 4);
            file_.read(reinterpret_cast<char*>(&tensor_count_), 8);
            file_.read(reinterpret_cast<char*>(&metadata_kv_count_), 8);
            return true;
        }

        bool read_metadata() {
            for (uint64_t i = 0; i < metadata_kv_count_; ++i) {
                std::string key = read_string();
                uint32_t vt;
                file_.read(reinterpret_cast<char*>(&vt), 4);
                GGUFValue val;
                val.type = static_cast<typename GGUFValue::Type>(vt);
                read_value(val);
                metadata_[key] = val;
            }
            return true;
        }

        bool read_tensor_infos() {
            for (uint64_t i = 0; i < tensor_count_; ++i) {
                GGUFTensorInfo info;
                info.name = read_string();
                uint32_t nd;
                file_.read(reinterpret_cast<char*>(&nd), 4);
                info.dimensions.resize(nd);
                file_.read(reinterpret_cast<char*>(info.dimensions.data()), nd * 8);
                file_.read(reinterpret_cast<char*>(&info.type), 4);
                file_.read(reinterpret_cast<char*>(&info.offset), 8);
                info.num_weights = 1;
                for (auto d : info.dimensions) info.num_weights *= d;
                info.size = info.num_weights;
                tensor_infos_.push_back(info);
            }
            return true;
        }

        std::string read_string() {
            uint64_t len;
            file_.read(reinterpret_cast<char*>(&len), 8);
            std::string s(static_cast<size_t>(len), 0);
            file_.read(&s[0], len);
            return s;
        }

        void read_value(GGUFValue& v) {
            switch (v.type) {
            case GGUFValue::U8: file_.read(reinterpret_cast<char*>(&v.u8), 1); break;
            case GGUFValue::I8: file_.read(reinterpret_cast<char*>(&v.i8), 1); break;
            case GGUFValue::U16: file_.read(reinterpret_cast<char*>(&v.u16), 2); break;
            case GGUFValue::I16: file_.read(reinterpret_cast<char*>(&v.i16), 2); break;
            case GGUFValue::U32: file_.read(reinterpret_cast<char*>(&v.u32), 4); break;
            case GGUFValue::I32: file_.read(reinterpret_cast<char*>(&v.i32), 4); break;
            case GGUFValue::F32: file_.read(reinterpret_cast<char*>(&v.f32), 4); break;
            case GGUFValue::BOOL: file_.read(reinterpret_cast<char*>(&v.b), 1); break;
            case GGUFValue::STR: v.str = read_string(); break;
            case GGUFValue::U64: file_.read(reinterpret_cast<char*>(&v.u64), 8); break;
            case GGUFValue::I64: file_.read(reinterpret_cast<char*>(&v.i64), 8); break;
            case GGUFValue::F64: file_.read(reinterpret_cast<char*>(&v.f64), 8); break;
            case GGUFValue::ARR: {
                uint32_t at; uint64_t al;
                file_.read(reinterpret_cast<char*>(&at), 4);
                file_.read(reinterpret_cast<char*>(&al), 8);
                v.arr.resize(static_cast<size_t>(al));
                for (auto& x : v.arr) {
                    x.type = static_cast<typename GGUFValue::Type>(at);
                    read_value(x);
                }
            } break;
            }
        }

        template<typename T> T extract_value(const GGUFValue& v) const;
    };

    template<> inline uint8_t  GGUFLoader::extract_value(const GGUFValue& v) const { return v.u8; }
    template<> inline int32_t  GGUFLoader::extract_value(const GGUFValue& v) const { return v.i32; }
    template<> inline float    GGUFLoader::extract_value(const GGUFValue& v) const { return v.f32; }
    template<> inline bool     GGUFLoader::extract_value(const GGUFValue& v) const { return v.b; }
    template<> inline std::string GGUFLoader::extract_value(const GGUFValue& v) const { return v.str; }
    class GGUFBPETokenizer {
        public:
        void load_from_gguf(GGUFLoader& loader) {
            // 加载词汇表
            vocab_ = loader.get_vocab();
            for (size_t i = 0; i < vocab_.size(); ++i) {
                token_to_id_[vocab_[i]] = static_cast<int>(i);
            }
    
            // 加载 BPE 合并规则
            merges_ = loader.get_merges();
            for (size_t i = 0; i < merges_.size(); ++i) {
                merge_rank_[merges_[i]] = static_cast<int>(i);
            }
    
            // 加载特殊 token ID
            bos_token_id_ = loader.get_bos_token_id();
            eos_token_id_ = loader.get_eos_token_id();
            unk_token_id_ = loader.get_unknown_token_id();
    
            std::cout << "✓ BPE Tokenizer loaded: vocab=" << vocab_.size() 
                      << ", merges=" << merges_.size() 
                      << ", bos=" << bos_token_id_ 
                      << ", eos=" << eos_token_id_ << std::endl;
        }
        
        std::vector<int> encode(const std::string& text) {
            // --- 第一步：将文本转换为 UTF-8 字节序列，每个字节作为独立符号 ---
            std::vector<std::string> symbols;
            for (unsigned char c : text) {
                symbols.push_back(std::string(1, c));
            }
        
            if (symbols.empty()) return {};
        
            // --- 第二步：构建初始相邻对及其优先级 ---
            // 优先队列元素: (优先级(越小越优先), 左索引, 右索引)
            // 使用 greater 使得小值在堆顶（BPE 优先合并出现最早的规则，即 rank 最小的）
            using HeapItem = std::tuple<int, size_t, size_t>;
            std::priority_queue<HeapItem, std::vector<HeapItem>, std::greater<HeapItem>> heap;
        
            // 记录每个位置当前有效的左邻居和右邻居（用于处理合并后的更新）
            std::vector<size_t> left(symbols.size());
            std::vector<size_t> right(symbols.size());
            for (size_t i = 0; i < symbols.size(); ++i) {
                left[i] = i - 1;
                right[i] = i + 1;
            }
        
            // 用于快速查找某个 pair 的 rank
            auto get_rank = [&](const std::string& a, const std::string& b) -> int {
                auto it = merge_rank_.find({a, b});
                return (it != merge_rank_.end()) ? it->second : -1;
            };
        
            // 初始化堆
            for (size_t i = 0; i + 1 < symbols.size(); ++i) {
                int rank = get_rank(symbols[i], symbols[i + 1]);
                if (rank != -1) {
                    heap.emplace(rank, i, i + 1);
                }
            }
        
            // 用于标记哪些节点已被合并（失效）
            std::vector<bool> deleted(symbols.size(), false);
        
            // --- 第三步：BPE 合并循环 ---
            // 为避免过于频繁的输出，每处理 1000 次合并才打印一次进度
            int merge_count = 0;
            std::cout << "  BPE merging... (total initial symbols: " << symbols.size() << ")" << std::endl;
        
            while (!heap.empty()) {
                auto [rank, l, r] = heap.top();
                heap.pop();
        
                // 如果任一节点已被标记删除，跳过
                if (deleted[l] || deleted[r]) continue;
                // 如果它们不再是相邻的（中间有被合并过），跳过
                if (right[l] != r) continue;
        
                // 执行合并：将 symbols[l] 和 symbols[r] 合并为新的符号
                std::string merged = symbols[l] + symbols[r];
                
                // 更新符号列表：保留 l，标记 r 为删除，并让 l 指向 r 的右邻居
                symbols[l] = merged;
                deleted[r] = true;
                right[l] = right[r];
                if (right[r] < symbols.size()) {
                    left[right[r]] = l;
                }
        
                // 将 l 与其新的右邻居组成对，尝试加入堆
                if (right[l] < symbols.size()) {
                    int new_rank = get_rank(symbols[l], symbols[right[l]]);
                    if (new_rank != -1) {
                        heap.emplace(new_rank, l, right[l]);
                    }
                }
        
                // 将 l 的左邻居与 l 组成对，尝试加入堆
                if (left[l] < symbols.size()) {
                    int new_rank = get_rank(symbols[left[l]], symbols[l]);
                    if (new_rank != -1) {
                        heap.emplace(new_rank, left[l], l);
                    }
                }
        
                // 进度提示（可选）
                if (++merge_count % 1000 == 0) {
                    std::cout << "    merged " << merge_count << " pairs..." << std::endl;
                }
            }
            std::cout << "  BPE merging done. Total merges: " << merge_count 
                      << ", final symbols: " << (symbols.size() - std::count(deleted.begin(), deleted.end(), true)) << std::endl;
        
            // --- 第四步：将最终符号映射为 Token ID ---
            std::vector<int> tokens;
            for (size_t i = 0; i < symbols.size(); ++i) {
                if (!deleted[i]) {
                    auto it = token_to_id_.find(symbols[i]);
                    if (it != token_to_id_.end()) {
                        tokens.push_back(it->second);
                    } else {
                        // 理论上不应该发生，但若遇到未知符号，可用 unk_token_id
                        if (unk_token_id_ != -1) tokens.push_back(unk_token_id_);
                    }
                }
            }
            return tokens;
        }
        
            std::string decode(int id) const {
                if (id >= 0 && id < static_cast<int>(vocab_.size())) {
                    return vocab_[id];
                }
                return "";
            }
        
            int bos_token_id() const { return bos_token_id_; }
            int eos_token_id() const { return eos_token_id_; }
            int unk_token_id() const { return unk_token_id_; }
        
        private:
            std::vector<std::string> vocab_;
            std::unordered_map<std::string, int> token_to_id_;
            std::vector<std::pair<std::string, std::string>> merges_;
            std::map<std::pair<std::string, std::string>, int> merge_rank_;
            int bos_token_id_ = -1;
            int eos_token_id_ = -1;
            int unk_token_id_ = -1;
        };
    class TensorNameMapper {
    public:
        void add_pattern(const std::string& canon, std::vector<std::string> pats) { patterns_[canon] = pats; }
        void set_default_llama() {
            add_pattern("tok_embeddings", { "token_embd","tok_embeddings","model.embed_tokens" });
            add_pattern("norm", { "output_norm","model.norm","norm","final_norm" });
            add_pattern("output", { "output","lm_head","lm_head.weight","model.output" });
            add_pattern("blk.{bid}.attn_norm", { "blk.{bid}.attn_norm","layers.{bid}.attention_norm","model.layers.{bid}.input_layernorm" });
            add_pattern("blk.{bid}.ffn_norm", { "blk.{bid}.ffn_norm","layers.{bid}.ffn_norm","model.layers.{bid}.post_attention_layernorm" });
            add_pattern("blk.{bid}.attn_q", { "blk.{bid}.attn_q","layers.{bid}.attention.wq","model.layers.{bid}.self_attn.q_proj" });
            add_pattern("blk.{bid}.attn_k", { "blk.{bid}.attn_k","layers.{bid}.attention.wk","model.layers.{bid}.self_attn.k_proj" });
            add_pattern("blk.{bid}.attn_v", { "blk.{bid}.attn_v","layers.{bid}.attention.wv","model.layers.{bid}.self_attn.v_proj" });
            add_pattern("blk.{bid}.attn_output", { "blk.{bid}.attn_output","layers.{bid}.attention.wo","model.layers.{bid}.self_attn.o_proj" });
            add_pattern("blk.{bid}.ffn_gate", { "blk.{bid}.ffn_gate","layers.{bid}.feed_forward.w1","model.layers.{bid}.mlp.gate_proj" });
            add_pattern("blk.{bid}.ffn_down", { "blk.{bid}.ffn_down","layers.{bid}.feed_forward.w2","model.layers.{bid}.mlp.down_proj" });
            add_pattern("blk.{bid}.ffn_up", { "blk.{bid}.ffn_up","layers.{bid}.feed_forward.w3","model.layers.{bid}.mlp.up_proj" });
        }
        std::vector<std::string> get_variants(const std::string& canon, int lid) const {
            auto it = patterns_.find(canon); if (it == patterns_.end()) return {};
            std::vector<std::string> res;
            for (auto& p : it->second) {
                std::string s = p;
                if (lid >= 0) { size_t pos = s.find("{bid}"); if (pos != std::string::npos) s.replace(pos, 5, std::to_string(lid)); }
                res.push_back(s);
            } return res;
        }
    private:
        std::unordered_map<std::string, std::vector<std::string>> patterns_;
    };

    // 辅助函数：将一行量化数据反量化为float
    // 辅助函数：将一行量化数据反量化为float
    void dequantize_row(const uint8_t* src, float* dst, size_t n, DataType dtype) {
        switch (dtype) {
        case DataType::Q4_0:
            dequantize_q4_0_row(src, dst, n);
            break;
        case DataType::Q4_1:
            dequantize_q4_1_row(src, dst, n);
            break;
        case DataType::Q8_0:
            ops::dequantize_q8_0_row(src, dst, n);
            break;
        case DataType::Q3_K:
            ops::dequantize_q3_K_row(src, dst, static_cast<int64_t>(n));
            break;
        case DataType::Q4_K:
        case DataType::Q4_K_M:  // 确保 Q4_K_M 使用正确的反量化函数
            ops::dequantize_q4_K_row(src, dst, static_cast<int64_t>(n));
            break;
        case DataType::Q5_K_M:
            ops::dequantize_q5_K_M_row(src, dst, static_cast<int64_t>(n));
            break;
        case DataType::Q6_K:
            ops::dequantize_q6_K_row(src, dst, static_cast<int64_t>(n));
            break;
        default:
            std::cerr << "Unsupported quantized embedding type: " << static_cast<int>(dtype) << std::endl;
            throw std::runtime_error("Unsupported quantized embedding type");
        }
    }

    class Transformer {
    public:
        Transformer(const ModelConfig& cfg) : config_(cfg) {
            mapper_.set_default_llama();
        }

        bool load_from_gguf(const std::string& path) {
            GGUFLoader loader;
            if (!loader.load(path)) {
                std::cerr << "✗ Failed to load GGUF" << std::endl;
                return false;
            }

            config_.max_seq_len = loader.get_metadata<int32_t>("llama.context_length", 2048);
            config_.norm_eps = loader.get_metadata<float>("llama.attention.layer_norm_rms_epsilon", 1e-5f);

            embeddings_ = load_tensor(loader, "tok_embeddings", -1);
            if (!embeddings_) {
                std::cerr << "✗ Failed to load embeddings" << std::endl;
                return false;
            }
            const auto& emb_shape = embeddings_->shape();
            if (emb_shape.dims.size() < 2) {
                std::cerr << "✗ Invalid embeddings shape" << std::endl;
                return false;
            }
            size_t d0 = emb_shape.dims[0], d1 = emb_shape.dims[1];
            if (d0 < d1) {
                config_.dim = static_cast<int>(d0);
                config_.vocab_size = static_cast<int>(d1);
            }
            else {
                config_.dim = static_cast<int>(d1);
                config_.vocab_size = static_cast<int>(d0);
            }
            std::cout << "✓ dim=" << config_.dim << " vocab=" << config_.vocab_size << std::endl;

            int max_layer = -1;
            for (const auto& info : loader.tensor_infos()) {
                const std::string& name = info.name;
                if (name.find("blk.") == 0) {
                    size_t dot_pos = name.find('.', 4);
                    if (dot_pos != std::string::npos) {
                        int layer = std::stoi(name.substr(4, dot_pos - 4));
                        if (layer > max_layer) max_layer = layer;
                    }
                }
            }
            config_.n_layers = max_layer + 1;
            std::cout << "✓ layers=" << config_.n_layers << std::endl;

            LayerWeights l0;
            l0.attention_norm = load_tensor(loader, "blk.{bid}.attn_norm", 0);
            l0.ffn_norm = load_tensor(loader, "blk.{bid}.ffn_norm", 0);
            l0.wq = load_tensor(loader, "blk.{bid}.attn_q", 0);
            l0.wk = load_tensor(loader, "blk.{bid}.attn_k", 0);
            l0.wv = load_tensor(loader, "blk.{bid}.attn_v", 0);
            l0.wo = load_tensor(loader, "blk.{bid}.attn_output", 0);
            l0.w1 = load_tensor(loader, "blk.{bid}.ffn_gate", 0);
            l0.w2 = load_tensor(loader, "blk.{bid}.ffn_down", 0);
            l0.w3 = load_tensor(loader, "blk.{bid}.ffn_up", 0);

            if (!l0.wq || !l0.wk || !l0.wv || !l0.w1) {
                std::cerr << "✗ Failed to load layer 0" << std::endl;
                return false;
            }

            int kv_dim = static_cast<int>(l0.wk->shape().dims[1]);
            int head_dim_guess = 128;
            config_.n_kv_heads = kv_dim / head_dim_guess;
            if (config_.n_kv_heads == 0) config_.n_kv_heads = 1;
            int head_dim = kv_dim / config_.n_kv_heads;
            config_.n_heads = config_.dim / head_dim;
            if (config_.n_heads == 0) config_.n_heads = 1;

            config_.hidden_dim = static_cast<int>(l0.w1->shape().dims[1]);

            std::cout << "✓ n_heads=" << config_.n_heads
                << " kv_heads=" << config_.n_kv_heads
                << " hidden=" << config_.hidden_dim << std::endl;

            layers_.push_back(std::move(l0));

            for (int l = 1; l < config_.n_layers; ++l) {
                LayerWeights lw;
                lw.attention_norm = load_tensor(loader, "blk.{bid}.attn_norm", l);
                lw.ffn_norm = load_tensor(loader, "blk.{bid}.ffn_norm", l);
                lw.wq = load_tensor(loader, "blk.{bid}.attn_q", l);
                lw.wk = load_tensor(loader, "blk.{bid}.attn_k", l);
                lw.wv = load_tensor(loader, "blk.{bid}.attn_v", l);
                lw.wo = load_tensor(loader, "blk.{bid}.attn_output", l);
                lw.w1 = load_tensor(loader, "blk.{bid}.ffn_gate", l);
                lw.w2 = load_tensor(loader, "blk.{bid}.ffn_down", l);
                lw.w3 = load_tensor(loader, "blk.{bid}.ffn_up", l);

                if (!lw.attention_norm || !lw.ffn_norm || !lw.wq || !lw.wk || !lw.wv || !lw.wo ||
                    !lw.w1 || !lw.w2 || !lw.w3) {
                    std::cerr << "✗ Failed to load layer " << l << std::endl;
                    return false;
                }
                layers_.push_back(std::move(lw));
            }

            final_norm_ = load_tensor(loader, "norm", -1);
            if (!final_norm_) {
                std::cerr << "✗ Failed to load final norm" << std::endl;
                return false;
            }

            lm_head_ = load_tensor(loader, "output", -1);
            if (!lm_head_) {
                std::cout << "⚠ Output not found, using weight tying" << std::endl;
                lm_head_ = std::make_unique<Tensor>(
                    embeddings_->dtype(),
                    embeddings_->shape(),
                    embeddings_->raw_data(),
                    embeddings_->bytes()
                );
            }

            std::cout << "✓ Model loaded! Embedding dtype: " << static_cast<int>(embeddings_->dtype())
                      << ", LM head dtype: " << static_cast<int>(lm_head_->dtype()) << std::endl;
            return true;
        }

        std::vector<float> forward(int token, int pos, std::vector<std::unique_ptr<Tensor>>& kv_cache) {
            int dim = config_.dim;
            int kv_dim = config_.kv_dim();
            int head_dim = config_.head_dim();
            int n_heads = config_.n_heads;
            int n_kv_heads = config_.n_kv_heads;

            auto x = std::make_unique<Tensor>(DataType::FP32, Shape{ (size_t)dim });
            
            // 修复4：安全处理量化 Embedding
            DataType emb_dtype = embeddings_->dtype();
            if (emb_dtype == DataType::FP16) {
                const uint16_t* emb = embeddings_->data<uint16_t>() + token * dim;
                for (int i = 0; i < dim; ++i) x->data<float>()[i] = fp16_to_fp32(emb[i]);
            }
            else if (emb_dtype == DataType::FP32) {
                const float* emb = embeddings_->data<float>() + token * dim;
                std::memcpy(x->data<float>(), emb, dim * sizeof(float));
            }
            else {
                
                // 不使用硬编码的行大小，直接从张量信息计算
                // 对于 embedding 张量，shape 通常是 [vocab_size, dim]
                // 所以每行的字节数 = total_bytes / vocab_size
                size_t row_size = embeddings_->bytes() / config_.vocab_size;
                
                const uint8_t* src = embeddings_->raw_data() + token * row_size;
                
                // 直接调用反量化，传入正确的 n=dim
                dequantize_row(src, x->data<float>(), dim, emb_dtype);
            }
            for (int l = 0; l < config_.n_layers; ++l) {
                auto& lw = layers_[l];

                auto normed = std::make_unique<Tensor>(DataType::FP32, Shape{ (size_t)dim });
                ops::rms_norm(dim, x->data<float>(), lw.attention_norm->data<float>(),
                    config_.norm_eps, normed->data<float>());

                auto q = std::make_unique<Tensor>(DataType::FP32, Shape{ (size_t)dim });
                auto k = std::make_unique<Tensor>(DataType::FP32, Shape{ (size_t)kv_dim });
                auto v = std::make_unique<Tensor>(DataType::FP32, Shape{ (size_t)kv_dim });
                gemm_weight(1, dim, dim, normed->data<float>(), lw.wq.get(), q->data<float>());
                gemm_weight(1, kv_dim, dim, normed->data<float>(), lw.wk.get(), k->data<float>());
                gemm_weight(1, kv_dim, dim, normed->data<float>(), lw.wv.get(), v->data<float>());

                // 修复1：正确应用RoPE到所有heads
                ops::apply_rope(n_heads, n_kv_heads, head_dim, pos, q->data<float>(), k->data<float>());

                if (kv_cache.size() <= (size_t)l * 2 + 1)
                    kv_cache.resize(l * 2 + 2);
                auto& kc = kv_cache[l * 2];
                auto& vc = kv_cache[l * 2 + 1];
                if (!kc) {
                    kc = std::make_unique<Tensor>(DataType::FP32, Shape{ (size_t)config_.max_seq_len, (size_t)kv_dim });
                    vc = std::make_unique<Tensor>(DataType::FP32, Shape{ (size_t)config_.max_seq_len, (size_t)kv_dim });
                }
                std::memcpy(kc->data<float>() + pos * kv_dim, k->data<float>(), kv_dim * sizeof(float));
                std::memcpy(vc->data<float>() + pos * kv_dim, v->data<float>(), kv_dim * sizeof(float));

                auto attn_out = std::make_unique<Tensor>(DataType::FP32, Shape{ (size_t)dim });
                std::memset(attn_out->data<float>(), 0, dim * sizeof(float));

                for (int h = 0; h < n_heads; ++h) {
                    int kv_h = (h * n_kv_heads) / n_heads;
                    float* qh = q->data<float>() + h * head_dim;
                    std::vector<float> scores(pos + 1);
                    for (int t = 0; t <= pos; ++t) {
                        float* kh = kc->data<float>() + t * kv_dim + kv_h * head_dim;
                        float s = 0;
                        for (int d = 0; d < head_dim; ++d) s += qh[d] * kh[d];
                        scores[t] = s / std::sqrt(static_cast<float>(head_dim));
                    }
                    ops::softmax(scores.data(), pos + 1);
                    float* outh = attn_out->data<float>() + h * head_dim;
                    for (int t = 0; t <= pos; ++t) {
                        float* vh = vc->data<float>() + t * kv_dim + kv_h * head_dim;
                        float w = scores[t];
                        for (int d = 0; d < head_dim; ++d) outh[d] += w * vh[d];
                    }
                }

                auto attn_proj = std::make_unique<Tensor>(DataType::FP32, Shape{ (size_t)dim });
                gemm_weight(1, dim, dim, attn_out->data<float>(), lw.wo.get(), attn_proj->data<float>());
                ops::axpy(dim, 1.0f, attn_proj->data<float>(), x->data<float>());

                ops::rms_norm(dim, x->data<float>(), lw.ffn_norm->data<float>(),
                    config_.norm_eps, normed->data<float>());
                auto ffn_hidden = std::make_unique<Tensor>(DataType::FP32, Shape{ (size_t)config_.hidden_dim });
                auto ffn_gate = std::make_unique<Tensor>(DataType::FP32, Shape{ (size_t)config_.hidden_dim });
                gemm_weight(1, config_.hidden_dim, dim, normed->data<float>(), lw.w1.get(), ffn_hidden->data<float>());
                gemm_weight(1, config_.hidden_dim, dim, normed->data<float>(), lw.w3.get(), ffn_gate->data<float>());
                
                // 修复2：正确应用SwiGLU激活函数
                for (int i = 0; i < config_.hidden_dim; ++i) {
                    float h = ffn_hidden->data<float>()[i];  // gate_proj
                    float g = ffn_gate->data<float>()[i];    // up_proj
                    // silu(gate) * up
                    ffn_hidden->data<float>()[i] = (h / (1.0f + std::exp(-h))) * g;
                }
                
                auto ffn_out = std::make_unique<Tensor>(DataType::FP32, Shape{ (size_t)dim });
                gemm_weight(1, dim, config_.hidden_dim, ffn_hidden->data<float>(), lw.w2.get(), ffn_out->data<float>());
                ops::axpy(dim, 1.0f, ffn_out->data<float>(), x->data<float>());
            }

            auto final_hidden = std::make_unique<Tensor>(DataType::FP32, Shape{ (size_t)dim });
            ops::rms_norm(dim, x->data<float>(), final_norm_->data<float>(),
                config_.norm_eps, final_hidden->data<float>());

            std::vector<float> logits(config_.vocab_size);
            
            // 修复3：统一使用gemm_weight处理lm_head（支持量化）
            gemm_weight(1, config_.vocab_size, dim, final_hidden->data<float>(), lm_head_.get(), logits.data());
            
            return logits;
        }

        void reset_kv_cache() { kv_cache_.clear(); }
        const ModelConfig& config() const { return config_; }

    private:
        struct LayerWeights {
            std::unique_ptr<Tensor> wq, wk, wv, wo;
            std::unique_ptr<Tensor> w1, w2, w3;
            std::unique_ptr<Tensor> attention_norm, ffn_norm;

            LayerWeights() = default;
            LayerWeights(LayerWeights&&) = default;
            LayerWeights& operator=(LayerWeights&&) = default;
        };

        std::unique_ptr<Tensor> load_tensor(GGUFLoader& loader, const std::string& canon, int lid) {
            auto vars = mapper_.get_variants(canon, lid);
            for (auto& name : vars) {
                for (auto& suffix : { ".weight", "", ".bias" }) {
                    std::string full = name + suffix;
                    auto* info = loader.find_tensor(full);
                    if (info) {
                        return loader.load_tensor(*info);
                    }
                }
            }
            return nullptr;
        }

        void gemm_weight(int M, int N, int K, const float* A, const Tensor* B, float* C) {
            bool transB = false;
            int ldb = N;
            if (B->shape().dims.size() == 2) {
                if (B->shape().dims[0] == (size_t)K && B->shape().dims[1] == (size_t)N) {
                    transB = true;
                    ldb = K;
                }
                else if (B->shape().dims[0] == (size_t)N && B->shape().dims[1] == (size_t)K) {
                    transB = false;
                    ldb = N;
                }
            }
        
            DataType dtype = B->dtype();
        
            switch (dtype) {
            case DataType::FP32:
                ops::gemm_fp32(false, transB, M, N, K, 1.0f, A, K, B->data<float>(), ldb, 0.0f, C, N);
                break;
            case DataType::FP16:
                ops::gemm_fp16_weight(transB, M, N, K, 1.0f, A, K, B->data<uint16_t>(), ldb, 0.0f, C, N);
                break;
            case DataType::Q4_0:
                ops::gemm_q4_0_weight(transB, M, N, K, 1.0f, A, K, B->raw_data(), 0.0f, C, N);
                break;
            case DataType::Q4_1:
                ops::gemm_q4_1_weight(transB, M, N, K, 1.0f, A, K, B->raw_data(), 0.0f, C, N);
                break;
            case DataType::Q8_0:
                ops::gemm_q8_0_weight(transB, M, N, K, 1.0f, A, K, B->raw_data(), 0.0f, C, N);
                break;
            case DataType::Q3_K:
                ops::gemm_q3_K_weight(transB, M, N, K, 1.0f, A, K, B->raw_data(), 0.0f, C, N);
                break;
            case DataType::Q4_K:
                ops::gemm_q4_K_weight(transB, M, N, K, 1.0f, A, K, B->raw_data(), 0.0f, C, N);
                break;
            case DataType::Q4_K_M:
                ops::gemm_q4_K_weight(transB, M, N, K, 1.0f, A, K, B->raw_data(), 0.0f, C, N);
                break;
            case DataType::Q5_K_M:
                ops::gemm_q5_K_M_weight(transB, M, N, K, 1.0f, A, K, B->raw_data(), 0.0f, C, N);
                break;
            case DataType::Q6_K:
                ops::gemm_q6_K_weight(transB, M, N, K, 1.0f, A, K, B->raw_data(), 0.0f, C, N);
                break;
            default:
                std::cerr << "✗ Unsupported dtype: " << static_cast<int>(dtype) << std::endl;
                throw std::runtime_error("Unsupported GEMM dtype");
            }
        }

        ModelConfig config_;
        TensorNameMapper mapper_;
        std::unique_ptr<Tensor> embeddings_;
        std::unique_ptr<Tensor> final_norm_;
        std::unique_ptr<Tensor> lm_head_;
        std::vector<LayerWeights> layers_;
        std::vector<std::unique_ptr<Tensor>> kv_cache_;
    };

    class GGUFTokenizer {
        public:
            void load_from_gguf(GGUFLoader& loader) {
                vocab_ = loader.get_vocab();
                bos_id_ = loader.get_bos_token_id();
                eos_id_ = loader.get_eos_token_id();
                
                // 建立字符串到ID的映射
                for (size_t i = 0; i < vocab_.size(); ++i) {
                    str_to_id_[vocab_[i]] = static_cast<int>(i);
                }
                
                std::cout << "✓ Tokenizer loaded: vocab_size=" << vocab_.size() 
                          << ", BOS=" << bos_id_ << ", EOS=" << eos_id_ << std::endl;
            }
            
            std::vector<int> encode(const std::string& text) {
                std::vector<int> tokens;
                // 简单的按空格分割匹配，适用于英文
                std::string word;
                for (char c : text) {
                    if (c == ' ') {
                        if (!word.empty()) {
                            tokens.push_back(get_token_id(word));
                            word.clear();
                        }
                        // 空格本身可能也是一个token
                        tokens.push_back(get_token_id(" "));
                    } else {
                        word += c;
                    }
                }
                if (!word.empty()) {
                    tokens.push_back(get_token_id(word));
                }
                // 移除连续的-1（未找到的token）
                tokens.erase(std::remove(tokens.begin(), tokens.end(), -1), tokens.end());
                return tokens;
            }
            
            std::string decode(int id) {
                if (id >= 0 && id < static_cast<int>(vocab_.size())) {
                    return vocab_[id];
                }
                return "";
            }
            
            int get_token_id(const std::string& token) {
                auto it = str_to_id_.find(token);
                if (it != str_to_id_.end()) return it->second;
                return -1; // 未知token
            }
            
            int bos_id() const { return bos_id_; }
            int eos_id() const { return eos_id_; }
            
        private:
            std::vector<std::string> vocab_;
            std::unordered_map<std::string, int> str_to_id_;
            int bos_id_ = -1;
            int eos_id_ = -1;
        };

} // namespace infer

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model.gguf>\n";
        return 1;
    }
    try {
        infer::ModelConfig cfg;
        infer::Transformer model(cfg);
        if (!model.load_from_gguf(argv[1])) {
            std::cerr << "✗ Failed to load model\n";
            return 1;
        }

        // 加载分词器
        infer::GGUFBPETokenizer tok;
        {
            infer::GGUFLoader loader;
            if (!loader.load(argv[1])) {
                std::cerr << "✗ Failed to load GGUF for tokenizer\n";
                return 1;
            }
            tok.load_from_gguf(loader);
        }

        std::cout << "Prompt: ";
        std::string prompt;
        std::getline(std::cin, prompt);

        // ---- 构建对话模板 ----
        // Qwen2.5-Coder 的对话格式
        std::string formatted_prompt = 
            "<|im_start|>system\n"
            "You are Qwen, a helpful coding assistant created by Alibaba Cloud.<|im_end|>\n"
            "<|im_start|>user\n" +
            prompt + "<|im_end|>\n"
            "<|im_start|>assistant\n";

        std::cout << "Formatted prompt:\n" << formatted_prompt << std::endl;

        // 编码完整对话
        auto input_tokens = tok.encode(formatted_prompt);
        std::cout << "Encoded tokens (" << input_tokens.size() << "): ";
        for (int t : input_tokens) std::cout << t << " ";
        std::cout << std::endl;

        // 构建输入序列：BOS + 对话tokens
        std::vector<int> tokens;
        int bos_id = tok.bos_token_id();
        int eos_id = tok.eos_token_id();
        
        if (bos_id != -1) {
            tokens.push_back(bos_id);
        }
        tokens.insert(tokens.end(), input_tokens.begin(), input_tokens.end());

        if (tokens.empty()) {
            std::cerr << "Failed to encode prompt\n";
            return 1;
        }

        std::vector<std::unique_ptr<infer::Tensor>> kvcache;
        int pos = 0;

        std::cout << "Processing prompt... ";
        for (size_t i = 0; i < tokens.size(); ++i) {
            model.forward(tokens[i], pos++, kvcache);
            std::cout << ".";
        }
        std::cout << " done!" << std::endl;

        int next = tokens.back();
        std::cout << "Generating: " << std::flush;
        
        for (int step = 0; step < 20; ++step) {
            auto logits = model.forward(next, pos++, kvcache);
            
            // 调试：检查 logits 是否有 NaN 或 Inf
            for (size_t i = 0; i < logits.size(); ++i) {
                if (std::isnan(logits[i]) || std::isinf(logits[i])) {
                    std::cerr << "ERROR: NaN/Inf in logits at index " << i << std::endl;
                    exit(1);
                }
            }
            
            // 打印 Top-5 预测
            std::vector<std::pair<float, int>> top5;
            for (int i = 0; i < std::min(5, (int)logits.size()); ++i) {
                top5.push_back({logits[i], i});
            }
            std::partial_sort(top5.begin(), top5.begin() + 5, top5.end(), std::greater<>());
            
            std::cout << "Step " << step << " Top-5: ";
            for (auto& p : top5) {
                std::cout << p.second << "(" << tok.decode(p.second) << "):" << p.first << " ";
            }
            std::cout << std::endl;
            
            next = std::max_element(logits.begin(), logits.end()) - logits.begin();
            // ...
        }
        std::cout << std::endl;
    }
    catch (const std::exception& e) {
        std::cerr << "✗ Error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
