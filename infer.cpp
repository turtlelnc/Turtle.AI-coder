// infer.cpp - GGUF格式 INT4/FP16 量化模型 CPU 推理引擎
// 依赖：nlohmann/json.hpp (需放置于同目录)
// 编译：g++ -std=c++17 -O3 -march=native -pthread infer.cpp -o infer
// 或 Visual Studio 中设置为 C++17 并包含 json.hpp 路径

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

#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <fcntl.h>
#endif

#include "json.hpp"
using json = nlohmann::json;

namespace infer {
    enum class DataType {
        FP32, FP16, INT8, INT4, Q4_0, Q4_1, Q8_0, Q4_K, Q3_K
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
    const uint32_t exp  = (h & 0x7C00) >> 10;
    const uint32_t mant = (h & 0x03FF) << 13;
    if (exp == 0) {
        if (mant == 0) return sign ? -0.0f : 0.0f;
        uint32_t m = mant, e = 0;
        while ((m & 0x00800000) == 0) { m <<= 1; e++; }
        m &= 0x007FFFFF;
        uint32_t new_exp = 127 - 15 - e;
        uint32_t bits = sign | (new_exp << 23) | m;
        float f; std::memcpy(&f, &bits, sizeof(f)); return f;
    } else if (exp == 0x1F) {
        uint32_t bits = sign | 0x7F800000 | mant;
        float f; std::memcpy(&f, &bits, sizeof(f)); return f;
    } else {
        uint32_t new_exp = exp - 15 + 127;
        uint32_t bits = sign | (new_exp << 23) | mant;
        float f; std::memcpy(&f, &bits, sizeof(f)); return f;
    }
}

// Q4_0 反量化（简化版，实际需更精确）
void dequantize_q4_0_row(const uint8_t* packed, float* dst, size_t n) {
    size_t nb = (n + 31) / 32;
    for (size_t b = 0; b < nb; ++b) {
        uint16_t scale_f16; std::memcpy(&scale_f16, packed, 2);
        float scale = fp16_to_fp32(scale_f16);
        packed += 2;
        size_t blk = std::min((size_t)32, n - b*32);
        for (size_t i = 0; i < blk; ++i) {
            uint8_t byte = packed[i/2];
            uint8_t val = (i%2==0) ? (byte & 0x0F) : (byte >> 4);
            dst[b*32 + i] = (static_cast<float>(static_cast<int8_t>(val << 4) >> 4)) * scale;
        }
        packed += 16;
    }
}
void dequantize_q4_1_row(const uint8_t* packed, float* dst, size_t n) {
    size_t nb = (n + 31) / 32;
    for (size_t b = 0; b < nb; ++b) {
        uint16_t scale_f16, zp_f16; std::memcpy(&scale_f16, packed, 2); std::memcpy(&zp_f16, packed+2, 2);
        float scale = fp16_to_fp32(scale_f16), zp = fp16_to_fp32(zp_f16);
        packed += 4;
        size_t blk = std::min((size_t)32, n - b*32);
        for (size_t i = 0; i < blk; ++i) {
            uint8_t byte = packed[i/2];
            uint8_t val = (i%2==0) ? (byte & 0x0F) : (byte >> 4);
            dst[b*32 + i] = (static_cast<float>(val) - zp) * scale;
        }
        packed += 16;
    }
}

namespace ops {
// Q3_K 反量化（适配 llama.cpp 格式）
void dequantize_q3_K_row(const uint8_t* x, float* y, int64_t n) {
    const int nb = n / 256;
    const size_t bs = 16;
    
    for (int b = 0; b < nb; ++b) {
        // Q3_K 块结构：
        // - 12 字节 scales（每个子块一个 6-bit scale）
        // - 4 字节 q3_scale（用于高精度部分）
        // - 256 个 3-bit 值 + 一些辅助位
        const uint8_t* scales = x;
        const uint8_t* q3_scale = x + 12;
        x += 16;
        
        float d_all = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(q3_scale));
        
        for (int i = 0; i < 16; ++i) {
            // 获取当前子块的 scale（6-bit 值）
            uint8_t s6 = (scales[i/2] >> (4*(i%2))) & 0x3F;
            float d = d_all * (s6 - 32) / 64.0f;
            
            // 读取 16 个 3-bit 值（需要复杂的位操作，此处简化但正确）
            for (int j = 0; j < 16; ++j) {
                // 这里省略了完整实现，因为 Q3_K 的位布局较复杂
                // 实际实现请参考 llama.cpp 的 dequantize_row_q3_K
                y[j] = 0.0f; // 占位，需替换为真实代码
            }
            y += 16;
            x += 16 * 3 / 8; // 16 个 3-bit 值占 6 字节
        }
    }
}
void dequantize_q4_K_row(const uint8_t* x, float* y, int64_t n) {
    const int nb = n / 256;  // 每 256 个元素一个块
    const size_t bs = 16;    // 每个子块 16 个元素
    
    for (int b = 0; b < nb; ++b) {
        // 读取 scale 和 min（各 12 字节，包含多个子块的量化参数）
        const uint8_t* scales = x;
        const uint8_t* mins  = x + 12;
        x += 24;
        
        // 256 个权重分为 16 个子块，每子块 16 个元素
        for (int i = 0; i < 16; ++i) {
            float d = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(scales + i*2));
            float m = fp16_to_fp32(*reinterpret_cast<const uint16_t*>(mins + i*2));
            
            // 每个子块有 16 个 4-bit 值（8 字节）
            for (int j = 0; j < 16; ++j) {
                uint8_t v = (x[j/2] >> ((j%2)*4)) & 0x0F;
                y[j] = d * (static_cast<float>(v) - m);
            }
            x += 8;
            y += 16;
        }
    }
}void gemm_q4_K_weight(bool transB, int M, int N, int K, float alpha, const float* A, int lda,
                      const uint8_t* B, float beta, float* C, int ldc) {
    if (transB) {
        // B 以 N 行 K 列存储
        size_t stride = ((K + 255) / 256) * (24 + 16*8); // 每 256 个元素需要 24 + 16*8 = 152 字节
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
    } else {
        // B 以 K 行 N 列存储（常见）
        size_t stride = ((N + 255) / 256) * 152;
        std::vector<float> B_row(N);
        if (beta == 0.0f) {
            for (int i = 0; i < M * N; ++i) C[i] = 0.0f;
        } else {
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
    void gemm_fp32(bool transA, bool transB, int M, int N, int K, float alpha,
        const float* A, int lda, const float* B, int ldb, float beta, float* C, int ldc) {
for (int i = 0; i < M; ++i) {
 for (int j = 0; j < N; ++j) {
     float sum = 0;
     for (int k = 0; k < K; ++k) {
         float a = transA ? A[k*lda+i] : A[i*lda+k];
         float b = transB ? B[j*ldb+k] : B[k*ldb+j];
         sum += a * b;
     }
     if (beta == 0.0f)
         C[i*ldc+j] = alpha * sum;
     else
         C[i*ldc+j] = alpha * sum + beta * C[i*ldc+j];
 }
}
}

void gemm_fp16_weight(bool transB, int M, int N, int K, float alpha, const float* A, int lda,
               const uint16_t* B, int ldb, float beta, float* C, int ldc) {
for (int i = 0; i < M; ++i) {
 for (int j = 0; j < N; ++j) {
     float sum = 0;
     for (int k = 0; k < K; ++k) {
         float b_val = transB ? fp16_to_fp32(B[j*ldb+k]) : fp16_to_fp32(B[k*ldb+j]);
         sum += A[i*lda+k] * b_val;
     }
     if (beta == 0.0f)
         C[i*ldc+j] = alpha * sum;
     else
         C[i*ldc+j] = alpha * sum + beta * C[i*ldc+j];
 }
}
}

void gemm_q4_0_weight(bool transB, int M, int N, int K, float alpha, const float* A, int lda,
               const uint8_t* B, float beta, float* C, int ldc) {
if (transB) {
 // B 以 N 行 K 列存储
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
} else {
 // B 以 K 行 N 列存储（常见）
 size_t stride = ((N + 31) / 32) * 18;
 std::vector<float> B_row(N);
 // 初始化 C
 if (beta == 0.0f) {
     for (int i = 0; i < M * N; ++i) C[i] = 0.0f;
 } else {
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
} else {
 size_t stride = ((N + 31) / 32) * 20;
 std::vector<float> B_row(N);
 if (beta == 0.0f) {
     for (int i = 0; i < M * N; ++i) C[i] = 0.0f;
 } else {
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
void axpy(int n, float a, const float* x, float* y) { for(int i=0;i<n;++i) y[i]+=a*x[i]; }
void rms_norm(int n, const float* x, const float* w, float eps, float* y) {
    float ss = 0; for(int i=0;i<n;++i) ss += x[i]*x[i];
    float ir = 1.0f / std::sqrt(ss/n + eps);
    for(int i=0;i<n;++i) y[i] = x[i] * ir * w[i];
}
void softmax(float* x, int n) {
    float m = *std::max_element(x, x+n), sum = 0;
    for(int i=0;i<n;++i) { x[i] = std::exp(x[i]-m); sum += x[i]; }
    for(int i=0;i<n;++i) x[i] /= sum;
}
void apply_rope(int /*seq_len*/, int head_dim, int pos, float* q, float* k) {
    for(int i=0;i<head_dim;i+=2) {
        float theta = 1.0f / std::pow(10000.0f, (float)i/head_dim);
        float c = std::cos(pos*theta), s = std::sin(pos*theta);
        float q0 = q[i], q1 = q[i+1]; q[i]=q0*c-q1*s; q[i+1]=q1*c+q0*s;
        float k0 = k[i], k1 = k[i+1]; k[i]=k0*c-k1*s; k[i+1]=k1*c+k0*s;
    }
}
}

struct ModelConfig {
    int dim=0, n_layers=32, n_heads=32, n_kv_heads=32, hidden_dim=11008, vocab_size=32000, max_seq_len=2048;
    float norm_eps=1e-5f;
    int head_dim() const { return dim/n_heads; }
    int kv_dim() const { return (dim*n_kv_heads)/n_heads; }
};

class GGUFLoader {
    public:
        // 将结构体定义移至 public，以便外部访问
        struct GGUFTensorInfo {
            std::string name;
            uint32_t type;
            std::vector<uint64_t> dimensions;
            uint64_t offset;
            uint64_t size;
            uint64_t num_weights;
        };
    
        bool load(const std::string& path) {
            #ifdef _WIN32
                int wlen = MultiByteToWideChar(CP_UTF8,0,path.c_str(),-1,nullptr,0);
                std::wstring wpath(wlen,0);
                MultiByteToWideChar(CP_UTF8,0,path.c_str(),-1,&wpath[0],wlen);
                HANDLE h = CreateFileW(wpath.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                if(h==INVALID_HANDLE_VALUE) return false;
                int fd = _open_osfhandle((intptr_t)h, _O_RDONLY|_O_BINARY);
                if(fd==-1) { CloseHandle(h); return false; }
                FILE* fp = _fdopen(fd, "rb");
                if(!fp) { _close(fd); return false; }
                file_ = std::ifstream(fp);
            #else
                file_.open(path, std::ios::binary);
                if(!file_) return false;
            #endif
            
                if(!read_header()) return false;
                if(!read_metadata()) return false;
                if(!read_tensor_infos()) return false;
            
                // ===== 关键修正：使用第一个张量的偏移量作为数据区起始 =====
                if (!tensor_infos_.empty()) {
                    tensor_data_start_offset_ = tensor_infos_[0].offset;
                } else {
                    tensor_data_start_offset_ = 0;
                }
                // 将文件指针移动到数据区起始位置
                file_.seekg(tensor_data_start_offset_);
                
                // 读取剩余所有数据到 tensor_data_
                file_.seekg(0, std::ios::end);
                size_t file_size = file_.tellg();
                size_t data_size = file_size - tensor_data_start_offset_;
                file_.seekg(tensor_data_start_offset_);
                tensor_data_.resize(data_size);
                file_.read(reinterpret_cast<char*>(tensor_data_.data()), data_size);
            
                // 按张量在文件中的实际大小修正 info.size
                for (size_t i = 0; i < tensor_infos_.size(); ++i) {
                    auto& t = tensor_infos_[i];
                    if (i + 1 < tensor_infos_.size()) {
                        t.size = tensor_infos_[i+1].offset - t.offset;
                    } else {
                        t.size = file_size - t.offset;
                    }
                }
            
                std::cout << "Data section start offset: " << tensor_data_start_offset_ 
                          << ", data size: " << data_size << " bytes" << std::endl;
                return true;
            }
    
        const std::vector<GGUFTensorInfo>& tensor_infos() const { return tensor_infos_; }
    
        const GGUFTensorInfo* find_tensor(const std::string& name) const {
            for(const auto& t : tensor_infos_) {
                if(t.name == name) return &t;
            }
            return nullptr;
        }
    
        std::unique_ptr<Tensor> load_tensor(const GGUFTensorInfo& info) {
            Shape sh;
            for (auto d : info.dimensions) sh.dims.push_back(static_cast<size_t>(d));
            DataType dt = gguf_to_dtype(info.type);
        
            // 计算在 tensor_data_ 中的偏移
            size_t offset_in_data = static_cast<size_t>(info.offset - tensor_data_start_offset_);
            size_t required_size = static_cast<size_t>(info.size);
        
            if (offset_in_data + required_size > tensor_data_.size()) {
                std::cerr << "ERROR: Tensor '" << info.name << "' out of bounds. "
                          << "offset_in_data=" << offset_in_data
                          << ", required_size=" << required_size
                          << ", buffer_size=" << tensor_data_.size() << std::endl;
                return nullptr;
            }
        
            const uint8_t* src = tensor_data_.data() + offset_in_data;
            return std::make_unique<Tensor>(dt, sh, src, required_size);
        }
    
        template<typename T> T get_metadata(const std::string& key, T def) const {
            auto it = metadata_.find(key); 
            if(it==metadata_.end()) return def;
            return extract_value<T>(it->second);
        }
    
    private:
    struct GGUFValue {
        enum Type { U8,I8,U16,I16,U32,I32,F32,BOOL,STR,ARR,U64,I64,F64 } type;
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
            switch(t) { 
                case 0:  return DataType::FP32; 
                case 1:  return DataType::FP16; 
                case 2:  return DataType::Q4_0; 
                case 3:  return DataType::Q4_1; 
                case 8:  return DataType::Q8_0; 
                case 12: return DataType::Q3_K;   // Q3_K
                case 14: return DataType::Q4_K;   // Q4_K_M
                default: 
                    throw std::runtime_error("Unsupported GGUF type ID: " + std::to_string(t)); 
            }
        }
    
        bool read_header() {
            uint32_t magic; 
            file_.read(reinterpret_cast<char*>(&magic), 4);
            if(magic != 0x46554747) return false;
            file_.read(reinterpret_cast<char*>(&version_), 4); 
            file_.read(reinterpret_cast<char*>(&tensor_count_), 8); 
            file_.read(reinterpret_cast<char*>(&metadata_kv_count_), 8);
            return true;
        }
    
        bool read_metadata() {
            for(uint64_t i=0; i<metadata_kv_count_; ++i) {
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
            for(uint64_t i=0; i<tensor_count_; ++i) {
                GGUFTensorInfo info; 
                info.name = read_string(); 
                uint32_t nd; 
                file_.read(reinterpret_cast<char*>(&nd), 4);
                info.dimensions.resize(nd); 
                file_.read(reinterpret_cast<char*>(info.dimensions.data()), nd*8);
                file_.read(reinterpret_cast<char*>(&info.type), 4); 
                file_.read(reinterpret_cast<char*>(&info.offset), 8);
                info.num_weights = 1; 
                for(auto d : info.dimensions) info.num_weights *= d;
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
            switch(v.type) {
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
                    for(auto& x : v.arr) { 
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
    

class TensorNameMapper {
public:
    void add_pattern(const std::string& canon, std::vector<std::string> pats) { patterns_[canon] = pats; }
    void set_default_llama() {
        add_pattern("tok_embeddings", {"token_embd","tok_embeddings","model.embed_tokens"});
        add_pattern("norm", {"output_norm","model.norm","norm","final_norm"});
        add_pattern("output", {"output","lm_head","lm_head.weight","model.output"});
        add_pattern("blk.{bid}.attn_norm", {"blk.{bid}.attn_norm","layers.{bid}.attention_norm","model.layers.{bid}.input_layernorm"});
        add_pattern("blk.{bid}.ffn_norm", {"blk.{bid}.ffn_norm","layers.{bid}.ffn_norm","model.layers.{bid}.post_attention_layernorm"});
        add_pattern("blk.{bid}.attn_q", {"blk.{bid}.attn_q","layers.{bid}.attention.wq","model.layers.{bid}.self_attn.q_proj"});
        add_pattern("blk.{bid}.attn_k", {"blk.{bid}.attn_k","layers.{bid}.attention.wk","model.layers.{bid}.self_attn.k_proj"});
        add_pattern("blk.{bid}.attn_v", {"blk.{bid}.attn_v","layers.{bid}.attention.wv","model.layers.{bid}.self_attn.v_proj"});
        add_pattern("blk.{bid}.attn_output", {"blk.{bid}.attn_output","layers.{bid}.attention.wo","model.layers.{bid}.self_attn.o_proj"});
        add_pattern("blk.{bid}.ffn_gate", {"blk.{bid}.ffn_gate","layers.{bid}.feed_forward.w1","model.layers.{bid}.mlp.gate_proj"});
        add_pattern("blk.{bid}.ffn_down", {"blk.{bid}.ffn_down","layers.{bid}.feed_forward.w2","model.layers.{bid}.mlp.down_proj"});
        add_pattern("blk.{bid}.ffn_up", {"blk.{bid}.ffn_up","layers.{bid}.feed_forward.w3","model.layers.{bid}.mlp.up_proj"});
    }
    std::vector<std::string> get_variants(const std::string& canon, int lid) const {
        auto it = patterns_.find(canon); if(it==patterns_.end()) return {};
        std::vector<std::string> res;
        for(auto& p : it->second) {
            std::string s = p;
            if(lid>=0) { size_t pos = s.find("{bid}"); if(pos!=std::string::npos) s.replace(pos,5,std::to_string(lid)); }
            res.push_back(s);
        } return res;
    }
private:
    std::unordered_map<std::string, std::vector<std::string>> patterns_;
};

class Transformer {
public:
    Transformer(const ModelConfig& cfg) : config_(cfg) {
        mapper_.set_default_llama();
    }

    bool load_from_gguf(const std::string& path) {
        GGUFLoader loader;
        if (!loader.load(path)) {
            std::cerr << "Failed to load GGUF file: " << path << std::endl;
            return false;
        }

        // 仅读取与架构无关的元数据
        config_.max_seq_len = loader.get_metadata<int32_t>("llama.context_length", 2048);
        config_.norm_eps    = loader.get_metadata<float>("llama.attention.layer_norm_rms_epsilon", 1e-5f);

        // 加载嵌入层，推断 vocab_size 和 dim
        embeddings_ = load_tensor(loader, "tok_embeddings", -1);
        if (!embeddings_) {
            std::cerr << "Failed to load token embeddings" << std::endl;
            return false;
        }
        const auto& emb_shape = embeddings_->shape();
        if (emb_shape.dims.size() < 2) {
            std::cerr << "Invalid embeddings shape" << std::endl;
            return false;
        }
        size_t d0 = emb_shape.dims[0], d1 = emb_shape.dims[1];
        if (d0 < d1) {
            config_.dim        = static_cast<int>(d0);
            config_.vocab_size = static_cast<int>(d1);
        } else {
            config_.dim        = static_cast<int>(d1);
            config_.vocab_size = static_cast<int>(d0);
        }
        std::cout << "Inferred: dim=" << config_.dim << ", vocab_size=" << config_.vocab_size << std::endl;

        // 自动检测层数
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
        std::cout << "Auto-detected layers: " << config_.n_layers << std::endl;

        // 加载第一层以推断头数、KV头数和FFN维度
        LayerWeights l0;
        l0.attention_norm = load_tensor(loader, "blk.{bid}.attn_norm", 0);
        l0.ffn_norm       = load_tensor(loader, "blk.{bid}.ffn_norm", 0);
        l0.wq             = load_tensor(loader, "blk.{bid}.attn_q", 0);
        l0.wk             = load_tensor(loader, "blk.{bid}.attn_k", 0);
        l0.wv             = load_tensor(loader, "blk.{bid}.attn_v", 0);
        l0.wo             = load_tensor(loader, "blk.{bid}.attn_output", 0);
        l0.w1             = load_tensor(loader, "blk.{bid}.ffn_gate", 0);
        l0.w2             = load_tensor(loader, "blk.{bid}.ffn_down", 0);
        l0.w3             = load_tensor(loader, "blk.{bid}.ffn_up", 0);

        if (!l0.wq || !l0.wk || !l0.wv || !l0.w1) {
            std::cerr << "Failed to load first layer tensors" << std::endl;
            return false;
        }

        // 从 wk 形状推断 kv_dim 和头数
        int kv_dim = static_cast<int>(l0.wk->shape().dims[1]);
        int head_dim_guess = 128;
        config_.n_kv_heads = kv_dim / head_dim_guess;
        if (config_.n_kv_heads == 0) config_.n_kv_heads = 1;
        int head_dim = kv_dim / config_.n_kv_heads;
        config_.n_heads = config_.dim / head_dim;
        if (config_.n_heads == 0) config_.n_heads = 1;

        // 从 w1 形状推断 hidden_dim
        config_.hidden_dim = static_cast<int>(l0.w1->shape().dims[1]);

        std::cout << "Inferred: n_heads=" << config_.n_heads 
                  << ", n_kv_heads=" << config_.n_kv_heads 
                  << ", head_dim=" << config_.head_dim() 
                  << ", hidden_dim=" << config_.hidden_dim << std::endl;

        layers_.push_back(std::move(l0));

        // 加载剩余层
        for (int l = 1; l < config_.n_layers; ++l) {
            std::cout << "Loading layer " << l << "..." << std::endl;
            LayerWeights lw;
            lw.attention_norm = load_tensor(loader, "blk.{bid}.attn_norm", l);
            lw.ffn_norm       = load_tensor(loader, "blk.{bid}.ffn_norm", l);
            lw.wq             = load_tensor(loader, "blk.{bid}.attn_q", l);
            lw.wk             = load_tensor(loader, "blk.{bid}.attn_k", l);
            lw.wv             = load_tensor(loader, "blk.{bid}.attn_v", l);
            lw.wo             = load_tensor(loader, "blk.{bid}.attn_output", l);
            lw.w1             = load_tensor(loader, "blk.{bid}.ffn_gate", l);
            lw.w2             = load_tensor(loader, "blk.{bid}.ffn_down", l);
            lw.w3             = load_tensor(loader, "blk.{bid}.ffn_up", l);

            if (!lw.attention_norm || !lw.ffn_norm || !lw.wq || !lw.wk || !lw.wv || !lw.wo ||
                !lw.w1 || !lw.w2 || !lw.w3) {
                std::cerr << "Failed to load layer " << l << std::endl;
                return false;
            }
            layers_.push_back(std::move(lw));
        }

        // 最终归一化
        final_norm_ = load_tensor(loader, "norm", -1);
        if (!final_norm_) {
            std::cerr << "Failed to load final norm" << std::endl;
            return false;
        }

        // 输出层（复用嵌入层）
        lm_head_ = load_tensor(loader, "output", -1);
        if (!lm_head_) {
            std::cout << "Output tensor not found, reusing token embeddings (weight tying)." << std::endl;
            lm_head_ = std::make_unique<Tensor>(
                embeddings_->dtype(),
                embeddings_->shape(),
                embeddings_->raw_data(),
                embeddings_->bytes()
            );
        }

        std::cout << "Model loaded successfully!" << std::endl;
        return true;
    }

    std::vector<float> forward(int token, int pos, std::vector<std::unique_ptr<Tensor>>& kv_cache) {
        // 从第一层的 wq 获取真实 dim
        int dim = static_cast<int>(layers_[0].wq->shape().dims[0]);
        int kv_dim = config_.kv_dim();
        int head_dim = config_.head_dim();
        int n_heads = config_.n_heads;
        int n_kv_heads = config_.n_kv_heads;

        if (config_.dim != dim) {
            std::cout << "Correcting config_.dim from " << config_.dim << " to " << dim << std::endl;
            config_.dim = dim;
            head_dim = config_.head_dim();
            kv_dim = config_.kv_dim();
        }

        std::cout << "forward start: token=" << token << ", pos=" << pos << ", dim=" << dim << std::endl;

        // Token embedding
        auto x = std::make_unique<Tensor>(DataType::FP32, Shape{(size_t)dim});
        if (embeddings_->dtype() == DataType::FP16) {
            const uint16_t* emb = embeddings_->data<uint16_t>() + token * dim;
            for (int i = 0; i < dim; ++i) x->data<float>()[i] = fp16_to_fp32(emb[i]);
        } else {
            const float* emb = embeddings_->data<float>() + token * dim;
            std::memcpy(x->data<float>(), emb, dim * sizeof(float));
        }

        // 遍历所有层
        for (int l = 0; l < config_.n_layers; ++l) {
            auto& lw = layers_[l];
            std::cout << "  layer " << l << " start" << std::endl;

            // RMSNorm (attention)
            auto normed = std::make_unique<Tensor>(DataType::FP32, Shape{(size_t)dim});
            ops::rms_norm(dim, x->data<float>(), lw.attention_norm->data<float>(),
                          config_.norm_eps, normed->data<float>());

            // Attention Q/K/V
            auto q = std::make_unique<Tensor>(DataType::FP32, Shape{(size_t)dim});
            auto k = std::make_unique<Tensor>(DataType::FP32, Shape{(size_t)kv_dim});
            auto v = std::make_unique<Tensor>(DataType::FP32, Shape{(size_t)kv_dim});
            gemm_weight(1, dim, dim, normed->data<float>(), lw.wq.get(), q->data<float>());
            gemm_weight(1, kv_dim, dim, normed->data<float>(), lw.wk.get(), k->data<float>());
            gemm_weight(1, kv_dim, dim, normed->data<float>(), lw.wv.get(), v->data<float>());

            // RoPE
            ops::apply_rope(1, head_dim, pos, q->data<float>(), k->data<float>());

            // KV Cache
            if (kv_cache.size() <= (size_t)l * 2 + 1)
                kv_cache.resize(l * 2 + 2);
            auto& kc = kv_cache[l * 2];
            auto& vc = kv_cache[l * 2 + 1];
            if (!kc) {
                kc = std::make_unique<Tensor>(DataType::FP32, Shape{(size_t)config_.max_seq_len, (size_t)kv_dim});
                vc = std::make_unique<Tensor>(DataType::FP32, Shape{(size_t)config_.max_seq_len, (size_t)kv_dim});
            }
            std::memcpy(kc->data<float>() + pos * kv_dim, k->data<float>(), kv_dim * sizeof(float));
            std::memcpy(vc->data<float>() + pos * kv_dim, v->data<float>(), kv_dim * sizeof(float));

            // 多头注意力
            auto attn_out = std::make_unique<Tensor>(DataType::FP32, Shape{(size_t)dim});
            std::memset(attn_out->data<float>(), 0, dim * sizeof(float));
            for (int h = 0; h < n_heads; ++h) {
                int kv_h = h * n_kv_heads / n_heads;
                float* qh = q->data<float>() + h * head_dim;
                std::vector<float> scores(pos + 1);
                for (int t = 0; t <= pos; ++t) {
                    float* kh = kc->data<float>() + t * kv_dim + kv_h * head_dim;
                    float s = 0;
                    for (int d = 0; d < head_dim; ++d) s += qh[d] * kh[d];
                    scores[t] = s / std::sqrt((float)head_dim);
                }
                ops::softmax(scores.data(), pos + 1);
                float* outh = attn_out->data<float>() + h * head_dim;
                for (int t = 0; t <= pos; ++t) {
                    float* vh = vc->data<float>() + t * kv_dim + kv_h * head_dim;
                    float w = scores[t];
                    for (int d = 0; d < head_dim; ++d) outh[d] += w * vh[d];
                }
            }

            // 输出投影
            auto attn_proj = std::make_unique<Tensor>(DataType::FP32, Shape{(size_t)dim});
            gemm_weight(1, dim, dim, attn_out->data<float>(), lw.wo.get(), attn_proj->data<float>());
            ops::axpy(dim, 1.0f, attn_proj->data<float>(), x->data<float>());

            // FFN (RMSNorm)
            ops::rms_norm(dim, x->data<float>(), lw.ffn_norm->data<float>(),
                          config_.norm_eps, normed->data<float>());
            auto ffn_hidden = std::make_unique<Tensor>(DataType::FP32, Shape{(size_t)config_.hidden_dim});
            auto ffn_gate   = std::make_unique<Tensor>(DataType::FP32, Shape{(size_t)config_.hidden_dim});
            gemm_weight(1, config_.hidden_dim, dim, normed->data<float>(), lw.w1.get(), ffn_hidden->data<float>());
            gemm_weight(1, config_.hidden_dim, dim, normed->data<float>(), lw.w3.get(), ffn_gate->data<float>());
            // SiLU
            for (int i = 0; i < config_.hidden_dim; ++i) {
                float g = ffn_gate->data<float>()[i];
                ffn_hidden->data<float>()[i] *= g / (1.0f + std::exp(-g));
            }
            auto ffn_out = std::make_unique<Tensor>(DataType::FP32, Shape{(size_t)dim});
            gemm_weight(1, dim, config_.hidden_dim, ffn_hidden->data<float>(), lw.w2.get(), ffn_out->data<float>());
            ops::axpy(dim, 1.0f, ffn_out->data<float>(), x->data<float>());
        }

        // 最终 RMSNorm
        auto final_hidden = std::make_unique<Tensor>(DataType::FP32, Shape{(size_t)dim});
        ops::rms_norm(dim, x->data<float>(), final_norm_->data<float>(),
                      config_.norm_eps, final_hidden->data<float>());

        // 输出 logits
        std::vector<float> logits(config_.vocab_size);
        if (lm_head_->dtype() == DataType::FP16) {
            const uint16_t* head = lm_head_->data<uint16_t>();
            for (int i = 0; i < config_.vocab_size; ++i) {
                float sum = 0.0f;
                for (int j = 0; j < dim; ++j)
                    sum += final_hidden->data<float>()[j] * fp16_to_fp32(head[i * dim + j]);
                logits[i] = sum;
            }
        } else {
            ops::gemm_fp32(false, true, 1, config_.vocab_size, dim, 1.0f,
                           final_hidden->data<float>(), dim,
                           lm_head_->data<float>(), dim,
                           0.0f, logits.data(), config_.vocab_size);
        }
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
            for (auto& suffix : {".weight", "", ".bias"}) {
                std::string full = name + suffix;
                auto* info = loader.find_tensor(full);
                if (info) {
                    std::cout << "  Loaded " << canon << " -> " << full << std::endl;
                    return loader.load_tensor(*info);
                }
            }
        }
        std::cerr << "  Failed to find tensor for " << canon << std::endl;
        return nullptr;
    }

    void gemm_weight(int M, int N, int K, const float* A, const Tensor* B, float* C) {
        bool transB = false;
        int ldb = N;
        if (B->shape().dims.size() == 2) {
            if (B->shape().dims[0] == (size_t)K && B->shape().dims[1] == (size_t)N) {
                transB = true;
                ldb = K;
            } else if (B->shape().dims[0] == (size_t)N && B->shape().dims[1] == (size_t)K) {
                transB = false;
                ldb = N;
            }
        }

        if (B->dtype() == DataType::FP16) {
            ops::gemm_fp16_weight(transB, M, N, K, 1.0f, A, K, B->data<uint16_t>(), ldb, 0.0f, C, N);
        } else if (B->dtype() == DataType::Q4_0) {
            ops::gemm_q4_0_weight(transB, M, N, K, 1.0f, A, K, B->raw_data(), 0.0f, C, N);
        } else if (B->dtype() == DataType::Q4_1) {
            ops::gemm_q4_1_weight(transB, M, N, K, 1.0f, A, K, B->raw_data(), 0.0f, C, N);
        } else if (B->dtype() == DataType::FP32) {
            ops::gemm_fp32(false, transB, M, N, K, 1.0f, A, K, B->data<float>(), ldb, 0.0f, C, N);
        }  else if (B->dtype() == DataType::Q4_K) {
            ops::gemm_q4_K_weight(transB, M, N, K, 1.0f, A, K, B->raw_data(), 0.0f, C, N);
        }else {
            throw std::runtime_error("Unsupported GEMM tensor dtype!");
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

class SimpleTokenizer {
public:
    void load(const std::string& path) {
        std::ifstream f(path); if(!f) { use_char_=true; return; }
        json j; f >> j; for(auto& [k,v] : j.items()) { vocab_[k]=v; id2tok_[v]=k; }
    }
    std::vector<int> encode(const std::string& s) {
        if(use_char_) { std::vector<int> t; for(char c:s) t.push_back((unsigned char)c); return t; }
        // 简单空格分词，实际需完善
        std::vector<int> t; std::string w;
        for(char c:s) { if(c==' ') { if(!w.empty()){ auto it=vocab_.find(w); t.push_back(it!=vocab_.end()?it->second:0); w.clear(); } } else w+=c; }
        if(!w.empty()) { auto it=vocab_.find(w); t.push_back(it!=vocab_.end()?it->second:0); } return t;
    }
    std::string decode(int id) {
        if(use_char_) return std::string(1,(char)id);
        auto it=id2tok_.find(id); return it!=id2tok_.end()?it->second:"<unk>";
    }
private:
    std::unordered_map<std::string,int> vocab_;
    std::unordered_map<int,std::string> id2tok_;
    bool use_char_=false;
};

} // namespace infer

int main(int argc, char* argv[]) {
    if(argc<2) { std::cerr << "Usage: " << argv[0] << " <model.gguf> [vocab.json]\n"; return 1; }
    try {
        infer::ModelConfig cfg;
        infer::Transformer model(cfg);
        if(!model.load_from_gguf(argv[1])) { std::cerr << "Failed to load model\n"; return 1; }
        infer::SimpleTokenizer tok;
        if(argc>=3) tok.load(argv[2]); else tok.load("");
        std::cout << "Model loaded. Enter prompt:\n";
        std::string prompt; std::getline(std::cin, prompt);
        auto toks = tok.encode(prompt); if(toks.empty()) return 1;
        std::vector<std::unique_ptr<infer::Tensor>> kvcache;
        int pos=0; for(int t : toks) model.forward(t, pos++, kvcache);
        int next = toks.back();
        for(int step=0; step<50; ++step) {
            auto logits = model.forward(next, pos++, kvcache);
            next = std::max_element(logits.begin(), logits.end()) - logits.begin();
            std::cout << tok.decode(next);
            if(next==0) break;
        }
        std::cout << std::endl;
    } catch(const std::exception& e) { std::cerr << "Error: " << e.what() << "\n"; return 1; }
    return 0;
}
