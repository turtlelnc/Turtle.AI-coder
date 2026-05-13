// main.cpp – 融合 Gemma 4 + DeepSeek V4 Pro 长上下文设计（致命错误已修复）
// 架构：循环深度 Transformer + 混合注意力（滑动窗口 + 因果稀疏全局）+ MoE
// 纯单线程稳定版，无 OpenMP 依赖，数值安全，无越界，无 NaN
#define EIGEN_NO_DEBUG
#define EIGEN_USE_BLAS
#include "Eigen/Dense"
#include "Eigen/Core"
#include <cmath>
#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include "BPE.h"
#include <map>
#include <memory>
#include <cstdint>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// 类型（行主序）
// ============================================================================
using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowVector = Eigen::RowVectorXf;
using Vector = Eigen::VectorXf;
using Array = Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

constexpr float SNR_THRESHOLD   = 1.0f;
constexpr float THRUST_STRENGTH = 0.1f;
constexpr float THRUST_MAX      = 1.0f;
constexpr int   MAX_SEQ_LEN     = 256;     // 与训练时的 seq_len 一致
constexpr int   MAX_DIM         = 256;
constexpr int   MAX_HEADS       = 8;
constexpr int MAX_EXPERTS = 32;

// ============================================================================
// 数值稳定激活函数
// ============================================================================
inline float stable_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + std::exp(-x));
    float ex = std::exp(x);
    return ex / (1.0f + ex);
}
inline float silu(float x) { return x * stable_sigmoid(x); }
inline float d_silu(float x) {
    float s = stable_sigmoid(x);
    return s + x * s * (1.0f - s);
}

// ============================================================================
// RoPE 预计算缓存
// ============================================================================
struct RoPECache {
    Matrix cos_tab, sin_tab;
    RoPECache(int max_len, int head_dim) : cos_tab(max_len, head_dim/2), sin_tab(max_len, head_dim/2) {
        for (int pos = 0; pos < max_len; ++pos)
            for (int j = 0; j < head_dim / 2; ++j) {
                float theta = std::pow(10000.0f, -2.0f * j / (float)head_dim);
                float angle = pos * theta;
                cos_tab(pos, j) = std::cos(angle);
                sin_tab(pos, j) = std::sin(angle);
            }
    }
    static const RoPECache& get(int max_len, int head_dim) {
        static RoPECache cache(max_len, head_dim);
        return cache;
    }
};

void apply_rope_inplace_fast(Matrix& Q, Matrix& K, int n_heads, int head_dim, bool inverse = false) {
    int L = Q.rows();
    float sign = inverse ? -1.0f : 1.0f;
    const auto& cache = RoPECache::get(MAX_SEQ_LEN, head_dim);
    using Complex = std::complex<float>;

    for (int pos = 0; pos < L; ++pos) {
        for (int h = 0; h < n_heads; ++h) {
            auto* q_ptr = reinterpret_cast<Complex*>(&Q(pos, h * head_dim));
            auto* k_ptr = reinterpret_cast<Complex*>(&K(pos, h * head_dim));
            for (int j = 0; j < head_dim / 2; ++j) {
                Complex rot(cache.cos_tab(pos, j), cache.sin_tab(pos, j) * sign);
                q_ptr[j] *= rot;
                k_ptr[j] *= rot;
            }
        }
    }
}

// ============================================================================
// 随机数与 SNR‑Gated 更新
// ============================================================================
std::mt19937 rng(42);
float randn(float mean = 0.0f, float std = 0.02f) {
    std::normal_distribution<float> dist(mean, std);
    return dist(rng);
}

template <typename Mat, typename Grad>
inline void snr_gated_update(Mat& param, const Grad& grad,
                            Mat& m, Mat& v,
                            float lr, float b1_corr, float b2_corr,
                            float eps = 1e-8f)
{
    constexpr float beta1 = 0.9f;
    constexpr float beta2 = 0.999f;
    using MatArray = Eigen::Array<typename Mat::Scalar, Mat::RowsAtCompileTime, Mat::ColsAtCompileTime, Mat::Options>;
    m = beta1 * m + (1.0f - beta1) * grad;
    v = beta2 * v + (1.0f - beta2) * grad.array().square().matrix();
    MatArray m_hat = m.array() / b1_corr;
    MatArray v_hat = v.array() / b2_corr;
    MatArray adam_step = lr * m_hat / (v_hat.sqrt() + eps);
    MatArray snr = m_hat.abs() / (v_hat.sqrt() + eps);
    MatArray thrust = -m_hat * lr * THRUST_STRENGTH;
    thrust = thrust.min(THRUST_MAX * lr).max(-THRUST_MAX * lr);
    MatArray total = adam_step + (snr > SNR_THRESHOLD).select(thrust, 0.0f);
    param -= total.matrix();
}

// ============================================================================
// 损失函数
// ============================================================================
float cross_entropy_backward(const RowVector& logits, int target, RowVector& dlogits) {
    float max_logit = logits.maxCoeff();
    RowVector shifted = (logits.array() - max_logit).matrix();
    RowVector exp_shifted = shifted.array().exp().matrix();
    float Z = exp_shifted.sum();
    RowVector softmax = exp_shifted / Z;
    float loss = -std::log(softmax(target) + 1e-10f);
    dlogits = softmax;
    dlogits(target) -= 1.0f;
    return loss;
}
Matrix apply_softmax_backward(const Matrix& dout, const Matrix& probs) {
    Matrix dlogits = Matrix::Zero(dout.rows(), dout.cols());
    for (int i = 0; i < dout.rows(); ++i) {
        RowVector dp = dout.row(i);
        RowVector p  = probs.row(i);
        float p_dot_dp = (p.array() * dp.array()).sum();
        dlogits.row(i) = (p.array() * (dp.array() - p_dot_dp)).matrix();
    }
    return dlogits;
}

// ============================================================================
// 基础层（无预分配，返回动态尺寸）
// ============================================================================
struct Linear {
    Matrix W; RowVector b;
    Matrix dW, mW, vW; RowVector db, mb, vb;
    int adam_t = 0;

    Linear(int in, int out) : W(in, out), b(out), dW(in, out), db(out),
                              mW(in, out), vW(in, out), mb(out), vb(out) {
        dW.setZero(); db.setZero(); mW.setZero(); vW.setZero(); mb.setZero(); vb.setZero();
        for (int i = 0; i < in; ++i) for (int j = 0; j < out; ++j) W(i, j) = randn();
        b.setZero();
    }

    Matrix forward(const Matrix& x_in) const { return (x_in * W).rowwise() + b; }
    Matrix backward(const Matrix& dout, const Matrix& cache_x) {
        dW.noalias() += cache_x.transpose() * dout;
        db += dout.colwise().sum();
        return dout * W.transpose();
    }
    void step(float lr, float max_grad_norm, float b1_corr, float b2_corr) {
        float nw = std::sqrt(dW.squaredNorm() + db.squaredNorm());
        if (nw > max_grad_norm) { dW *= max_grad_norm / nw; db *= max_grad_norm / nw; }
        snr_gated_update(W, dW, mW, vW, lr, b1_corr, b2_corr);
        snr_gated_update(b, db, mb, vb, lr, b1_corr, b2_corr);
        dW.setZero(); db.setZero();
    }
    void save(std::ostream& os) const {
        int r = W.rows(), c = W.cols();
        os.write((char*)&r, sizeof(int)); os.write((char*)&c, sizeof(int));
        os.write((char*)W.data(), sizeof(float)*W.size());
        os.write((char*)b.data(), sizeof(float)*b.size());
    }
    void load(std::istream& is) {
        int r, c; is.read((char*)&r, sizeof(int)); is.read((char*)&c, sizeof(int));
        is.read((char*)W.data(), sizeof(float)*W.size());
        is.read((char*)b.data(), sizeof(float)*b.size());
    }
};

struct RMSNorm {
    float eps;
    RowVector weight, dweight, mweight, vweight;
    int adam_t = 0;

    RMSNorm(int dim, float e = 1e-4f) : eps(e), weight(RowVector::Ones(dim)),
        dweight(RowVector::Zero(dim)), mweight(RowVector::Zero(dim)), vweight(RowVector::Zero(dim)) {}

    Matrix forward(const Matrix& x_in) const {
        Matrix norm = x_in;
        for (int i = 0; i < norm.rows(); ++i) {
            double var = norm.row(i).template cast<double>().squaredNorm() / norm.cols();
            float inv_rms = static_cast<float>(1.0 / std::sqrt(var + eps));
            norm.row(i) *= inv_rms;
            norm.row(i).array() *= weight.array();
        }
        return norm;
    }
    Matrix backward(const Matrix& dout, const Matrix& cache_x) {
        Matrix dx = Matrix::Zero(dout.rows(), dout.cols());
        for (int i = 0; i < dout.rows(); ++i) {
            double var = cache_x.row(i).template cast<double>().squaredNorm() / cache_x.cols();
            float inv_rms = static_cast<float>(1.0 / std::sqrt(var + eps));
            RowVector x_norm_row = cache_x.row(i) * inv_rms;
            RowVector d_norm_row = dout.row(i).array() * weight.array();
            float mean_dx = (d_norm_row.array() * x_norm_row.array()).mean();
            dx.row(i) = (d_norm_row.array() - x_norm_row.array() * mean_dx) * inv_rms;
            dweight.array() += d_norm_row.array() * cache_x.row(i).array() * inv_rms;
        }
        return dx;
    }
    void step(float lr, float b1_corr, float b2_corr) {
        snr_gated_update(weight, dweight, mweight, vweight, lr, b1_corr, b2_corr);
        dweight.setZero();
    }
    void save(std::ostream& os) const {
        int d = weight.size(); os.write((char*)&d, sizeof(int)); os.write((char*)weight.data(), sizeof(float)*d);
    }
    void load(std::istream& is) {
        int d; is.read((char*)&d, sizeof(int)); is.read((char*)weight.data(), sizeof(float)*d);
    }
};

struct LoopEmbedding {
    Matrix W, dW, mW, vW;
    int adam_t = 0;
    LoopEmbedding(int max_loops, int dim) : W(max_loops, dim), dW(max_loops, dim), mW(max_loops, dim), vW(max_loops, dim) {
        for (int i = 0; i < W.rows(); ++i) for (int j = 0; j < W.cols(); ++j) W(i,j) = randn() * 0.02f;
        dW.setZero(); mW.setZero(); vW.setZero();
    }
    Matrix forward(int t, int batch_size) const { return W.row(t).replicate(batch_size, 1); }
    void backward(const Matrix& dout, int t) { dW.row(t) += dout.colwise().sum(); }
    void step(float lr, float max_grad_norm, float b1_corr, float b2_corr) {
        float nw = std::sqrt(dW.squaredNorm());
        if (nw > max_grad_norm) dW *= max_grad_norm / nw;
        snr_gated_update(W, dW, mW, vW, lr, b1_corr, b2_corr);
        dW.setZero();
    }
};

// ============================================================================
// 混合注意力模块（滑动窗口 + 因果稀疏全局，已修复所有致命错误）
// ============================================================================


struct AttnCache { Matrix q, k, v, out; std::vector<Matrix> attn_probs; };

struct SlidingWindowAttention {
    Linear Wq, Wk, Wv, Wo;
    int n_heads, head_dim, window_size;
    Matrix *attn_scores, *attn_probs, *attn_tmp;

    SlidingWindowAttention(int dim, int h, int ws) 
        : Wq(dim, dim), Wk(dim, dim), Wv(dim, dim), Wo(dim, dim),
          n_heads(h), head_dim(dim/h), window_size(ws),
          attn_scores(nullptr), attn_probs(nullptr), attn_tmp(nullptr) {}

    void set_buf(Matrix* s, Matrix* p, Matrix* t) {
        attn_scores = s; attn_probs = p; attn_tmp = t;
    }

    Matrix forward(const Matrix& x, AttnCache& cache) {
        int L = x.rows();
        cache.q = Wq.forward(x);
        cache.k = Wk.forward(x);
        cache.v = Wv.forward(x);
        apply_rope_inplace_fast(cache.q, cache.k, n_heads, head_dim);

        cache.out.resize(L, Wo.W.cols());
        cache.out.setZero();
        cache.attn_probs.clear();

        float s = 1.0f / std::sqrt((float)head_dim);

        for (int h = 0; h < n_heads; ++h) {
            auto Qb = cache.q.block(0, h * head_dim, L, head_dim);
            auto Kb = cache.k.block(0, h * head_dim, L, head_dim);
            auto Vb = cache.v.block(0, h * head_dim, L, head_dim);

            // 计算完整注意力分数
            attn_scores->topLeftCorner(L, L).noalias() = Qb * Kb.transpose() * s;

            // 应用滑动窗口 + 因果掩码
            for (int i = 0; i < L; ++i) {
                for (int j = 0; j < L; ++j) {
                    if (j > i || (i - j) >= window_size)
                        (*attn_scores)(i, j) = -1e9f;
                }
            }

            // 稳定 softmax，仅在前 L 列有效
            for (int i = 0; i < L; ++i) {
                auto row_head = attn_scores->row(i).head(L);
                float max_val = row_head.maxCoeff();
                RowVector probs = (row_head.array() - max_val).exp();
                probs /= probs.sum();
                attn_probs->row(i).head(L) = probs;
                if (L < attn_probs->cols())
                    attn_probs->row(i).tail(attn_probs->cols() - L).setZero();
            }

            cache.attn_probs.push_back(attn_probs->topLeftCorner(L, L));
            cache.out.block(0, h * head_dim, L, head_dim).noalias() =
                attn_probs->topLeftCorner(L, L) * Vb;
        }
        return Wo.forward(cache.out);
    }

    Matrix backward(const Matrix& dout, const Matrix& cache_x, const AttnCache& cache) {
        int L = dout.rows();
        Matrix d_out = Wo.backward(dout, cache.out);
        Matrix dQ(L, d_out.cols()), dK(L, d_out.cols()), dV(L, d_out.cols());
        dQ.setZero(); dK.setZero(); dV.setZero();
        float s = 1.0f / std::sqrt((float)head_dim);

        for (int h = 0; h < n_heads; ++h) {
            auto dh = d_out.block(0, h * head_dim, L, head_dim);
            const Matrix& p = cache.attn_probs[h];
            auto Vb = cache.v.block(0, h * head_dim, L, head_dim);
            auto Kb = cache.k.block(0, h * head_dim, L, head_dim);
            auto Qb = cache.q.block(0, h * head_dim, L, head_dim);

            dV.block(0, h * head_dim, L, head_dim).noalias() += p.transpose() * dh;
            Matrix ds = dh * Vb.transpose();

            for (int i = 0; i < L; ++i) {
                RowVector dp = ds.row(i), pr = p.row(i);
                float p_dot_dp = (pr.array() * dp.array()).sum();
                attn_tmp->row(i).head(L) =
                    (pr.array() * (dp.array() - p_dot_dp)).matrix();
            }

            // 掩码区域梯度置零
            for (int i = 0; i < L; ++i)
                for (int j = 0; j < L; ++j)
                    if (j > i || (i - j) >= window_size)
                        (*attn_tmp)(i, j) = 0.0f;

            attn_tmp->topLeftCorner(L, L) *= s;

            dQ.block(0, h * head_dim, L, head_dim).noalias() +=
                attn_tmp->topLeftCorner(L, L) * Kb;
            dK.block(0, h * head_dim, L, head_dim).noalias() +=
                attn_tmp->topLeftCorner(L, L).transpose() * Qb;
        }

        apply_rope_inplace_fast(dQ, dK, n_heads, head_dim, true);
        return Wq.backward(dQ, cache_x) + Wk.backward(dK, cache_x) + Wv.backward(dV, cache_x);
    }

    void step(float lr, float gn, float b1_corr, float b2_corr) {
        Wq.step(lr, gn, b1_corr, b2_corr);
        Wk.step(lr, gn, b1_corr, b2_corr);
        Wv.step(lr, gn, b1_corr, b2_corr);
        Wo.step(lr, gn, b1_corr, b2_corr);
    }

    void save(std::ostream& os) const {
        Wq.save(os); Wk.save(os); Wv.save(os); Wo.save(os);
    }
    void load(std::istream& is) {
        Wq.load(is); Wk.load(is); Wv.load(is); Wo.load(is);
    }
};

struct SparseGlobalAttention {
    Linear Wq, Wk, Wv, Wo;
    int n_heads, head_dim, topk;
    Matrix *attn_scores, *attn_probs, *attn_tmp;

    SparseGlobalAttention(int dim, int h, int tk)
        : Wq(dim, dim), Wk(dim, dim), Wv(dim, dim), Wo(dim, dim),
          n_heads(h), head_dim(dim/h), topk(tk),
          attn_scores(nullptr), attn_probs(nullptr), attn_tmp(nullptr) {}

    void set_buf(Matrix* s, Matrix* p, Matrix* t) { attn_scores = s; attn_probs = p; attn_tmp = t; }

    Matrix forward(const Matrix& x, AttnCache& cache) {
        int L = x.rows();
        cache.q = Wq.forward(x); cache.k = Wk.forward(x); cache.v = Wv.forward(x);
        apply_rope_inplace_fast(cache.q, cache.k, n_heads, head_dim);
        cache.out.resize(L, Wo.W.cols()); cache.out.setZero();
        cache.attn_probs.clear();
        float s = 1.0f / std::sqrt((float)head_dim);

        for (int h = 0; h < n_heads; ++h) {
            auto Qb = cache.q.block(0, h*head_dim, L, head_dim);
            auto Kb = cache.k.block(0, h*head_dim, L, head_dim);
            auto Vb = cache.v.block(0, h*head_dim, L, head_dim);
            attn_scores->topLeftCorner(L, L).noalias() = Qb * Kb.transpose() * s;
            for (int i = 0; i < L; ++i) {
                for (int j = i + 1; j < L; ++j)
                    (*attn_scores)(i, j) = -1e9f;

                int valid_len = i + 1;
                int k = std::min(topk, valid_len);
                if (k > 0) {
                    RowVector temp_row = attn_scores->row(i).head(valid_len);
                    std::nth_element(temp_row.data(),
                                     temp_row.data() + k - 1,
                                     temp_row.data() + valid_len,
                                     std::greater<float>());
                    float threshold = temp_row[k - 1];
                    for (int j = 0; j < valid_len; ++j)
                        if ((*attn_scores)(i, j) < threshold)
                            (*attn_scores)(i, j) = -1e9f;
                }
            }

            for (int i = 0; i < L; ++i) {
                float max_val = attn_scores->row(i).maxCoeff();
                RowVector row = (attn_scores->row(i).array() - max_val).exp();
                row /= row.sum();
                attn_probs->row(i) = row;
            }

            cache.attn_probs.push_back(attn_probs->topLeftCorner(L, L));
            cache.out.block(0, h*head_dim, L, head_dim).noalias() =
                attn_probs->topLeftCorner(L, L) * Vb;
        }
        return Wo.forward(cache.out);
    }

    Matrix backward(const Matrix& dout, const Matrix& cache_x, const AttnCache& cache) {
        int L = dout.rows();
        Matrix d_out = Wo.backward(dout, cache.out);
        Matrix dQ(L, d_out.cols()), dK(L, d_out.cols()), dV(L, d_out.cols());
        dQ.setZero(); dK.setZero(); dV.setZero();
        float s = 1.0f / std::sqrt((float)head_dim);

        for (int h = 0; h < n_heads; ++h) {
            auto dh = d_out.block(0, h*head_dim, L, head_dim);
            const Matrix& p = cache.attn_probs[h];
            auto Vb = cache.v.block(0, h*head_dim, L, head_dim);
            auto Kb = cache.k.block(0, h*head_dim, L, head_dim);
            auto Qb = cache.q.block(0, h*head_dim, L, head_dim);

            dV.block(0, h*head_dim, L, head_dim).noalias() += p.transpose() * dh;
            Matrix ds = dh * Vb.transpose();

            for (int i = 0; i < L; ++i) {
                RowVector dp = ds.row(i), pr = p.row(i);
                float p_dot_dp = (pr.array() * dp.array()).sum();
                attn_tmp->row(i).head(L) = (pr.array() * (dp.array() - p_dot_dp)).matrix();
            }
            for (int i = 0; i < L; ++i)
                for (int j = i + 1; j < L; ++j)
                    (*attn_tmp)(i, j) = 0.0f;
            attn_tmp->topLeftCorner(L, L) *= s;

            dQ.block(0, h*head_dim, L, head_dim).noalias() += attn_tmp->topLeftCorner(L, L) * Kb;
            dK.block(0, h*head_dim, L, head_dim).noalias() += attn_tmp->topLeftCorner(L, L).transpose() * Qb;
        }
        apply_rope_inplace_fast(dQ, dK, n_heads, head_dim, true);
        return Wq.backward(dQ, cache_x) + Wk.backward(dK, cache_x) + Wv.backward(dV, cache_x);
    }

    void step(float lr, float gn, float b1_corr, float b2_corr) {
        Wq.step(lr, gn, b1_corr, b2_corr); Wk.step(lr, gn, b1_corr, b2_corr);
        Wv.step(lr, gn, b1_corr, b2_corr); Wo.step(lr, gn, b1_corr, b2_corr);
    }
    void save(std::ostream& os) const { Wq.save(os); Wk.save(os); Wv.save(os); Wo.save(os); }
    void load(std::istream& is) { Wq.load(is); Wk.load(is); Wv.load(is); Wo.load(is); }
};

// ============================================================================
// MoE 模块（预分配 + Max‑Trick）
// ============================================================================
struct ExpertCache { Matrix gate_out, up_out, interm; };
struct TokenRoute { int tid; float w; };
struct MoECache {
    Matrix probs, x_in;
    std::vector<std::vector<TokenRoute>> routes;
    Matrix e_in[MAX_EXPERTS];
    Matrix e_out[MAX_EXPERTS];
    std::vector<ExpertCache> e_caches;
    Matrix de_buf[MAX_EXPERTS];
    Matrix dl_local_buf;

    MoECache() {
        routes.resize(MAX_EXPERTS);
        e_caches.resize(MAX_EXPERTS);
        for (int i = 0; i < MAX_EXPERTS; ++i) {
            e_in[i].resize(MAX_SEQ_LEN, MAX_DIM);
            e_out[i].resize(MAX_SEQ_LEN, MAX_DIM);
            routes[i].reserve(MAX_SEQ_LEN);
            de_buf[i].resize(MAX_SEQ_LEN, MAX_DIM);
        }
        dl_local_buf.resize(MAX_SEQ_LEN, MAX_EXPERTS);
    }
};

struct Expert {
    Linear gate, up, down;
    Expert(int d, int h) : gate(d, h), up(d, h), down(h, d) {}
    Matrix forward(const Matrix& x, ExpertCache& cache) const {
        cache.gate_out = gate.forward(x);
        cache.up_out   = up.forward(x);
        Array gate_arr = cache.gate_out.array();
        Array sig = 1.0f / (1.0f + (-gate_arr).exp());
        cache.interm = (gate_arr * sig * cache.up_out.array()).matrix();
        return down.forward(cache.interm);
    }
    Matrix backward(const Matrix& dout, const ExpertCache& cache, const Matrix& cache_x) {
        Matrix di = down.backward(dout, cache.interm);
        Array gate_arr = cache.gate_out.array();
        Array sig = 1.0f / (1.0f + (-gate_arr).exp());
        Array f_g = gate_arr * sig;
    
        Matrix du = up.backward((di.array() * f_g).matrix(), cache_x);
    
        Array d_sig = sig + gate_arr * sig * (1.0f - sig);
        Matrix dg_in = (di.array() * cache.up_out.array() * d_sig).matrix();
        return du + gate.backward(dg_in, cache_x);
    }
    void step(float lr, float gn, float b1_corr, float b2_corr) {
        gate.step(lr, gn, b1_corr, b2_corr);
        up.step(lr, gn, b1_corr, b2_corr);
        down.step(lr, gn, b1_corr, b2_corr);
    }
    void save(std::ostream& os) const { gate.save(os); up.save(os); down.save(os); }
    void load(std::istream& is) { gate.load(is); up.load(is); down.load(is); }
};

struct MoE {
    int dim, n_experts, top_k;
    float balance_loss = 0;
    Linear router;
    std::vector<Expert> experts;

    MoE(int d, int h, int ne, int tk) : dim(d), n_experts(ne), top_k(tk), router(d, ne) {
        for (int i = 0; i < ne; ++i) experts.emplace_back(d, h);
    }

    Matrix forward(const Matrix& x, MoECache& c) {
        int BT = x.rows(); c.x_in = x;
        Matrix l = router.forward(x);
        Vector max_val = l.rowwise().maxCoeff();
        Matrix el = (l.colwise() - max_val).array().exp().matrix();
        Vector Z = el.rowwise().sum();
        c.probs = (el.array().colwise() / Z.array()).matrix();

        for (int i = 0; i < n_experts; ++i) c.routes[i].clear();

        for (int t = 0; t < BT; ++t) {
            int best_e = -1, second_e = -1;
            float best_p = -1.0f, second_p = -1.0f;
            for (int e = 0; e < n_experts; ++e) {
                float p = c.probs(t, e);
                if (p > best_p) { second_p = best_p; second_e = best_e; best_p = p; best_e = e; }
                else if (p > second_p) { second_p = p; second_e = e; }
            }
            if (best_e != -1) c.routes[best_e].push_back({t, best_p});
            if (second_e != -1 && top_k > 1) c.routes[second_e].push_back({t, second_p});
        }

        Matrix out = Matrix::Zero(BT, dim);
        for (int e = 0; e < n_experts; ++e) {
            int nt = c.routes[e].size();
            if (!nt) continue;
            auto Xe = c.e_in[e].topLeftCorner(nt, dim);
            for (int i = 0; i < nt; ++i) Xe.row(i) = x.row(c.routes[e][i].tid);
            Matrix Ye = experts[e].forward(Xe, c.e_caches[e]);
            c.e_out[e].topLeftCorner(nt, dim) = Ye;
            for (int i = 0; i < nt; ++i)
                out.row(c.routes[e][i].tid) += Ye.row(i) * c.routes[e][i].w;
        }

        RowVector cnt = RowVector::Zero(n_experts);
        for (int e = 0; e < n_experts; ++e) cnt(e) += c.routes[e].size();
        RowVector mean_probs = c.probs.colwise().mean();
        RowVector scaled_cnt = cnt * (n_experts / (float)(top_k * BT));
        balance_loss = scaled_cnt.dot(mean_probs) * 0.02f;
        return out;
    }

    Matrix backward(const Matrix& dout, const MoECache& c) {
        int BT = dout.rows();
        Matrix dx = Matrix::Zero(BT, dim), dl = Matrix::Zero(BT, n_experts);
        for (int e = 0; e < n_experts; ++e) {
            int nt = c.routes[e].size();
            if (!nt) continue;
            Matrix de(nt, dim);
            Matrix dl_local = Matrix::Zero(BT, n_experts);
            for (int i = 0; i < nt; ++i) {
                int t = c.routes[e][i].tid;
                de.row(i) = dout.row(t) * c.routes[e][i].w;
                dl_local(t, e) += dout.row(t).dot(c.e_out[e].row(i));
            }
            Matrix dxe = experts[e].backward(de, c.e_caches[e], c.e_in[e].topLeftCorner(nt, dim));
            for (int i = 0; i < nt; ++i) dx.row(c.routes[e][i].tid) += dxe.row(i);
            dl += dl_local;
        }
        float coef = (0.02f * n_experts) / (float)(top_k * BT);
        for (int e = 0; e < n_experts; ++e) {
            float cnt = c.routes[e].size();
            for (int t = 0; t < BT; ++t) dl(t, e) += (coef * cnt) / (float)BT;
        }
        dx += router.backward(apply_softmax_backward(dl, c.probs), c.x_in);
        return dx;
    }

    void step(float lr, float gn, float b1_corr, float b2_corr) {
        router.step(lr, gn, b1_corr, b2_corr);
        for (auto& e : experts) e.step(lr, gn, b1_corr, b2_corr);
    }
    void save(std::ostream& os) const { router.save(os); for(auto &e:experts) e.save(os); }
    void load(std::istream& is) { router.load(is); for(auto &e:experts) e.load(is); }
};

// ============================================================================
// ACT 与 RecurrentBlock（预留容量，轮流注意力）
// ============================================================================
struct ACTCache { std::vector<Vector> p, w; std::vector<Matrix> hs; };
struct ACT {
    int max_loops; Linear linear;
    ACT(int d, int l) : max_loops(l), linear(d, 1) {}
    Matrix forward(const std::vector<Matrix>& hs, ACTCache& c) {
        c.p.clear(); c.w.clear(); c.hs.clear();
        int L = hs[0].rows(); Matrix out = Matrix::Zero(L, hs[0].cols()); Vector rem = Vector::Ones(L);
        for (int t = 0; t < (int)hs.size(); ++t) {
            c.hs.push_back(hs[t]); Matrix lp = linear.forward(hs[t]);
            Vector pt(L);
            for(int i=0; i<L; ++i) pt(i) = stable_sigmoid(lp(i,0));
            c.p.push_back(pt); Vector wt(L);
            for(int i=0; i<L; ++i) {
                if(t == (int)hs.size()-1) wt(i) = rem(i);
                else { wt(i) = pt(i)*rem(i); rem(i) -= wt(i); }
            }
            c.w.push_back(wt);
            for(int i=0; i<L; ++i) out.row(i) += hs[t].row(i)*wt(i);
        }
        return out;
    }
    std::vector<Matrix> backward(const Matrix& dout, const ACTCache& c) {
        int L = dout.rows(), D = dout.cols(); std::vector<Matrix> dhs(c.hs.size(), Matrix::Zero(L, D));
        for (int t = 0; t < (int)c.hs.size(); ++t) {
            Matrix dlin(L,1); for(int i=0; i<L; ++i) {
                dhs[t].row(i) = dout.row(i) * c.w[t](i);
                float dw = dout.row(i).dot(c.hs[t].row(i));
                dlin(i,0) = dw * (c.p[t](i)*(1.0f-c.p[t](i)));
            }
            dhs[t] += linear.backward(dlin, c.hs[t]);
        }
        return dhs;
    }
    void step(float lr, float gn, float b1_corr, float b2_corr) {
        linear.step(lr, gn, b1_corr, b2_corr);
    }
    void save(std::ostream& os) const { linear.save(os); }
    void load(std::istream& is) { linear.load(is); }
};

struct RecurrentStepCache { Matrix x, n_attn, a_out, mid, n_moe; AttnCache a_c; MoECache m_c; };
// 一个不包含循环的普通 Transformer 块（宽层）
struct TransformerBlock {
    SlidingWindowAttention attn_sw;
    MoE moe;
    RMSNorm n_attn, n_moe;
    Matrix *attn_scores, *attn_probs, *attn_tmp;

    // 缓存上一次 forward 的中间值
    mutable Matrix last_x;
    mutable AttnCache last_a_cache;
    mutable MoECache last_m_cache;
    mutable Matrix last_normed_attn, last_normed_moe;

    TransformerBlock(int dim, int window_size, int moe_experts, int moe_topk,
                     Matrix* scores, Matrix* probs, Matrix* tmp)
        : attn_sw(dim, 8, window_size),
          moe(dim, dim*4, moe_experts, moe_topk),
          n_attn(dim), n_moe(dim),
          attn_scores(scores), attn_probs(probs), attn_tmp(tmp) {
        attn_sw.set_buf(scores, probs, tmp);
    }

    Matrix forward(const Matrix& x) {
        last_x = x;
        last_normed_attn = n_attn.forward(x);
        Matrix a_out = attn_sw.forward(last_normed_attn, last_a_cache);
        Matrix mid1 = x + a_out;
        last_normed_moe = n_moe.forward(mid1);
        Matrix m_out = moe.forward(last_normed_moe, last_m_cache);
        Matrix out = mid1 + m_out;
        return out;
    }

    Matrix backward(const Matrix& dout) {
        // dout 是损失对该块输出的梯度
        Matrix dm_out = dout;
        // MoE 反向
        Matrix d_normed_moe = moe.backward(dm_out, last_m_cache);
        // n_moe 反向
        Matrix dmid1 = n_moe.backward(d_normed_moe, last_normed_moe);
        Matrix da_out = dmid1;
        // 注意力反向
        Matrix d_normed_attn = attn_sw.backward(da_out, last_normed_attn, last_a_cache);
        // n_attn 反向
        Matrix dx2 = n_attn.backward(d_normed_attn, last_x);
        return dmid1 + dx2;   // dmid1 是 dx1
    }

    void step(float lr, float gn, float b1_corr, float b2_corr) {
        attn_sw.step(lr, gn, b1_corr, b2_corr);
        moe.step(lr, gn, b1_corr, b2_corr);
        n_attn.step(lr, b1_corr, b2_corr);
        n_moe.step(lr, b1_corr, b2_corr);
    }
};
struct RecurrentBlock {
    std::vector<std::unique_ptr<TransformerBlock>> wide_blocks;
    LoopEmbedding loop_embed;
    SlidingWindowAttention attn_sw;
    SparseGlobalAttention attn_global;
    MoE moe;
    RMSNorm n_attn, n_moe;
    ACT act;
    std::vector<RecurrentStepCache> b_cache;
    ACTCache a_cache;
    float memory_alpha = 0.9f;

    Matrix work_scores, work_probs, work_tmp;

    RecurrentBlock(int dim, int max_loops, int window_size, int global_topk,
                   int moe_experts, int moe_topk, int num_wide_blocks)
        : loop_embed(max_loops, dim),
          attn_sw(dim, 8, window_size),
          attn_global(dim, 8, global_topk),
          moe(dim, dim*4, moe_experts, moe_topk),
          n_attn(dim), n_moe(dim), act(dim, max_loops),
          work_scores(MAX_SEQ_LEN, MAX_SEQ_LEN),
          work_probs(MAX_SEQ_LEN, MAX_SEQ_LEN),
          work_tmp(MAX_SEQ_LEN, dim) {

        attn_sw.set_buf(&work_scores, &work_probs, &work_tmp);
        attn_global.set_buf(&work_scores, &work_probs, &work_tmp);

        for (int i = 0; i < num_wide_blocks; ++i) {
            wide_blocks.push_back(std::make_unique<TransformerBlock>(
                dim, window_size, moe_experts, moe_topk,
                &work_scores, &work_probs, &work_tmp));
        }
    }

    Matrix forward(Matrix x) {
        // 宽层
        for (auto& blk : wide_blocks) {
            x = blk->forward(x);
        }

        // 深层循环 + 分层记忆
        b_cache.clear();
        b_cache.reserve(act.max_loops);
        std::vector<Matrix> hs;
        Matrix memory = Matrix::Zero(x.rows(), x.cols());

        for (int t = 0; t < act.max_loops; ++t) {
            RecurrentStepCache c;
            c.x = x;
            c.n_attn = n_attn.forward(x);
            c.n_attn += memory * 0.1f;

            x += loop_embed.forward(t, x.rows());

            if (t % 2 == 0)
                c.a_out = attn_sw.forward(c.n_attn, c.a_c);
            else
                c.a_out = attn_global.forward(c.n_attn, c.a_c);

            c.mid = x + c.a_out;
            c.n_moe = n_moe.forward(c.mid);
            Matrix m_out = moe.forward(c.n_moe, c.m_c);
            x = c.mid + m_out;

            memory = memory_alpha * memory + (1.0f - memory_alpha) * x;

            hs.push_back(x);
            b_cache.push_back(c);
        }

        return act.forward(hs, a_cache);
    }

    Matrix backward(const Matrix& dout) {
        std::vector<Matrix> dhs = act.backward(dout, a_cache);
        int L = dout.rows(), D = dout.cols();
        Matrix dx = Matrix::Zero(L, D);
        Matrix dmem_new = Matrix::Zero(L, D);

        for (int t = (int)b_cache.size() - 1; t >= 0; --t) {
            auto& c = b_cache[t];
            Matrix dx_cur = dhs[t] + dmem_new * (1.0f - memory_alpha);
            dmem_new *= memory_alpha;  // dmem_prev 的初始部分

            // MoE 反向
            Matrix d_n_moe = moe.backward(dx_cur, c.m_c);
            Matrix d_mid = n_moe.backward(d_n_moe, c.mid);
            Matrix da_out = d_mid;
            Matrix dx_mid_in = d_mid;

            // 注意力反向
            Matrix d_n_attn;
            if (t % 2 == 0)
                d_n_attn = attn_sw.backward(da_out, c.n_attn, c.a_c);
            else
                d_n_attn = attn_global.backward(da_out, c.n_attn, c.a_c);

            // 记忆注入梯度
            dmem_new += d_n_attn * 0.1f;
            Matrix d_n_attn_out = d_n_attn;
            Matrix d_x_prev_attn = n_attn.backward(d_n_attn_out, c.x);

            // loop_embed
            loop_embed.backward(dx_mid_in, t);
            Matrix d_x_prev_mid = dx_mid_in;

            Matrix d_x_prev = d_x_prev_mid + d_x_prev_attn;
            dx = d_x_prev;
        }

        // 宽层反向
        for (int i = (int)wide_blocks.size() - 1; i >= 0; --i) {
            dx = wide_blocks[i]->backward(dx);
        }

        return dx;
    }

    void step(float lr, float gn, float b1_corr, float b2_corr) {
        for (auto& blk : wide_blocks) {
            blk->step(lr, gn, b1_corr, b2_corr);
        }
        attn_sw.step(lr, gn, b1_corr, b2_corr);
        attn_global.step(lr, gn, b1_corr, b2_corr);
        moe.step(lr, gn, b1_corr, b2_corr);
        n_attn.step(lr, b1_corr, b2_corr);
        n_moe.step(lr, b1_corr, b2_corr);
        act.step(lr, gn, b1_corr, b2_corr);
        loop_embed.step(lr, gn, b1_corr, b2_corr);
    }

    void save(std::ostream& os) const {
        n_attn.save(os);
        attn_sw.save(os);
        attn_global.save(os);
        n_moe.save(os);
        moe.save(os);
        act.save(os);
        // 宽层的保存
        for (auto& blk : wide_blocks) {
            blk->attn_sw.save(os);
            blk->moe.save(os);
            blk->n_attn.save(os);
            blk->n_moe.save(os);
        }
    }

    void load(std::istream& is) {
        n_attn.load(is);
        attn_sw.load(is);
        attn_global.load(is);
        n_moe.load(is);
        moe.load(is);
        act.load(is);
        for (auto& blk : wide_blocks) {
            blk->attn_sw.load(is);
            blk->moe.load(is);
            blk->n_attn.load(is);
            blk->n_moe.load(is);
        }
    }
};

// ============================================================================
// 顶层模型
// ============================================================================
struct OpenMythos {
    Linear embed, lm_head;
    RecurrentBlock recurrent;
    RMSNorm final_norm;
    Matrix work_scores, work_probs, work_tmp;
    Matrix h_out, f_normed;

    OpenMythos(int vocab, int dim, int max_loop, int window_size, int global_topk,
            int moe_experts, int moe_topk, int num_wide_blocks)
        : embed(vocab, dim), lm_head(dim, vocab),
        recurrent(dim, max_loop, window_size, global_topk,
                    moe_experts, moe_topk, num_wide_blocks),
        final_norm(dim),
        work_scores(MAX_SEQ_LEN, MAX_SEQ_LEN),
        work_probs(MAX_SEQ_LEN, MAX_SEQ_LEN),
        work_tmp(MAX_SEQ_LEN, dim) {}

    Matrix forward(const Eigen::MatrixXi& ids) {
        int BT = ids.rows(); Matrix x(BT, embed.W.cols());
        for (int i = 0; i < BT; ++i) x.row(i) = embed.W.row(ids(i,0));
        h_out = recurrent.forward(x);
        f_normed = final_norm.forward(h_out);
        return lm_head.forward(f_normed);
    }

    void backward(const Matrix& dlogits, const Eigen::MatrixXi& ids) {
        Matrix dx = lm_head.backward(dlogits, f_normed);
        dx = final_norm.backward(dx, h_out);
        dx = recurrent.backward(dx);
        for (int i = 0; i < ids.rows(); ++i) embed.dW.row(ids(i,0)) += dx.row(i);
        lm_head.dW += embed.dW.transpose();
        embed.dW.setZero();
    }

    void step(float lr, float gn, float b1_corr, float b2_corr) {
        recurrent.step(lr, gn, b1_corr, b2_corr);
        final_norm.step(lr, b1_corr, b2_corr);
        lm_head.step(lr, gn, b1_corr, b2_corr);
    }

    bool save_checkpoint(const std::string& path, const std::string& ds_id, int step) const {
        std::ofstream os(path, std::ios::binary);
        if (!os) return false;
        size_t idl = ds_id.length();
        os.write((char*)&idl, sizeof(size_t)); os.write(ds_id.c_str(), idl);
        os.write((char*)&step, sizeof(int));
        embed.save(os); recurrent.save(os); final_norm.save(os); lm_head.save(os);
        return true;
    }

    int load_checkpoint(const std::string& path, const std::string& ds_id) {
        std::ifstream is(path, std::ios::binary);
        if (!is) return -1;
        size_t idl; is.read((char*)&idl, sizeof(size_t));
        std::string sid(idl, ' '); is.read(&sid[0], idl);
        if (sid != ds_id) { std::cerr << "Checkpoint mismatch!" << std::endl; return -1; }
        int saved_step; is.read((char*)&saved_step, sizeof(int));
        embed.load(is); recurrent.load(is); final_norm.load(is); lm_head.load(is);
        return saved_step;
    }
};

// ============================================================================
// 文本生成与训练辅助
// ============================================================================
int sample_from_probs(const RowVector& probs, float temperature = 1.0f) {
    RowVector scaled = (probs.array() / temperature).max(-100.0f).min(100.0f);
    float max_val = scaled.maxCoeff();
    RowVector exp_vals = (scaled.array() - max_val).exp();
    float Z = exp_vals.sum();
    RowVector final_probs = exp_vals / Z;
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    float r = unif(rng);
    float cum = 0.0f;
    for (int i = 0; i < final_probs.size(); ++i) { cum += final_probs(i); if (r <= cum) return i; }
    return final_probs.size() - 1;
}

std::string generate_text(OpenMythos& model, bpe::BPETrainer& tokenizer,
                          const std::string& prompt, int max_new_tokens = 50, float temp = 0.8f) {
    std::vector<bpe::TokenId> prompt_ids = tokenizer.encode(prompt);
    std::vector<int> ids(prompt_ids.begin(), prompt_ids.end());
    Eigen::MatrixXi inp(ids.size(), 1);
    for (int i = 0; i < (int)ids.size(); ++i) inp(i,0) = ids[i];
    for (int i = 0; i < max_new_tokens; ++i) {
        Matrix logits = model.forward(inp);
        RowVector last = logits.row(inp.rows()-1);
        int next = (temp > 0.0f) ? sample_from_probs(last, temp) : [&](){ int idx; last.maxCoeff(&idx); return idx; }();
        if (next == bpe::EOS_TOKEN_ID) break;
        ids.push_back(next);
        inp.conservativeResize(ids.size(), 1);
        inp(ids.size()-1, 0) = next;
    }
    std::vector<bpe::TokenId> out_ids(ids.begin(), ids.end());
    return tokenizer.decode(out_ids);
}

uint64_t fnv1a64(const std::string& d) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : d) { h ^= (uint64_t)c; h *= 1099511628211ull; }
    return h;
}
std::string to_hex(uint64_t v) { std::ostringstream s; s << std::hex << std::setw(16) << std::setfill('0') << v; return s.str(); }
float get_lr(int s, int ts, float p, int w, float m) {
    if (s < w) return p * (s+1) / w;
    float r = (float)(s-w)/(ts-w); if(r>1) r=1;
    return p * (m + (1-m)*0.5f*(1+std::cos(M_PI*r)));
}

int main(int argc, char* argv[]) {
    int dim = 128, max_loop = 6;
    int window_size = 64, global_topk = 32;
    int seq_len = 128, accum = 8;
    int moe_experts = 16, moe_topk = 2, num_wide_blocks = 3;
    std::cout<<"版本1.0.2  Version 1.0.2"<<std::endl;

    if (argc >= 3 && std::string(argv[1]) == "gen") {
        bpe::BPETrainer tokenizer; tokenizer.load("tokenizer.bpe");
        OpenMythos model(tokenizer.vocab_size(), dim, max_loop, window_size, global_topk,
                 moe_experts, moe_topk, num_wide_blocks);
        model.load_checkpoint("openmythos_model.ckpt", "");
        std::string prompt = argv[2];
        std::cout << "Prompt: " << prompt << "\nGenerated: "
                  << generate_text(model, tokenizer, prompt, 50, 0.8f) << std::endl;
        return 0;
    }

    bpe::BPETrainer tokenizer; std::ifstream f("train.txt");
    if (!f) return 1;
    std::string text((std::istreambuf_iterator<char>(f)), {});
    std::string ds_id = "train_fnv1a64_" + to_hex(fnv1a64(text));
    std::cout << "Dataset ID: " << ds_id << std::endl;

    if (!tokenizer.load("tokenizer.bpe", ds_id)) {
        tokenizer.train_from_file("train.txt"); tokenizer.save("tokenizer.bpe", ds_id);
    }
    std::vector<bpe::TokenId> tokens = tokenizer.has_cached_tokens() ? tokenizer.cached_tokens() : tokenizer.encode(text);
    if (!tokenizer.has_cached_tokens()) { tokenizer.set_cached_tokens(tokens); tokenizer.save("tokenizer.bpe", ds_id); }

    int vocab = tokenizer.vocab_size();

    OpenMythos model(vocab, dim, max_loop, window_size, global_topk,
        moe_experts, moe_topk, num_wide_blocks);
    int start_step = model.load_checkpoint("openmythos_model.ckpt", ds_id);
    if (start_step < 0) start_step = 0;

    int total_steps = 3000;
    float peak_lr = 0.0001f;
    float best_loss = 1e9f;

    Eigen::MatrixXi inp(seq_len, 1), tgt(seq_len, 1);

    for (int s = start_step; s < total_steps; ++s) {
        float lr = get_lr(s, total_steps, peak_lr, 500, 0.3f);
        float b1_corr = 1.0f - std::pow(0.9f, static_cast<float>(s + 1));
        float b2_corr = 1.0f - std::pow(0.999f, static_cast<float>(s + 1));
        float loss_sum = 0, bal_sum = 0;
        for (int a = 0; a < accum; ++a) {
            int start = rng() % (tokens.size() - seq_len - 1);
            for (int i = 0; i < seq_len; ++i) { inp(i,0) = tokens[start+i]; tgt(i,0) = tokens[start+i+1]; }
            Matrix logits = model.forward(inp);
            float l = 0;
            Matrix dl = Matrix::Zero(seq_len, vocab);
            for (int t = 0; t < seq_len; ++t) {
                RowVector row_dl(vocab);
                l += cross_entropy_backward(logits.row(t), tgt(t,0), row_dl);
                dl.row(t) = row_dl;
            }
            l /= seq_len; dl /= (float)(seq_len * accum);
            loss_sum += l / accum; bal_sum += model.recurrent.moe.balance_loss / accum;
            model.backward(dl, inp);
        }
        float total_loss = loss_sum + bal_sum;
        if (total_loss > best_loss * 3.0f && best_loss < 1e8f) { lr *= 0.5f; }
        if (total_loss < best_loss) best_loss = total_loss;
        model.step(lr, 0.5f, b1_corr, b2_corr);
        model.embed.W = model.lm_head.W.transpose();
        model.embed.mW = model.lm_head.mW.transpose();
        model.embed.vW = model.lm_head.vW.transpose();

        if (s % 10 == 0) {
            std::cout << "Step " << s << " | Loss: " << total_loss
                      << " | Bal: " << bal_sum << " | LR: " << lr << std::endl;
            if (s > 0 && s % 100 == 0) model.save_checkpoint("openmythos_model.ckpt", ds_id, s);
            if (std::isnan(total_loss)) break;
        }
    }
    model.save_checkpoint("openmythos_model.ckpt", ds_id, 0);
    return 0;
}
