// openmythos_train.cpp – 最终极限优化版 (RowMajor + 零分配 + FastLog + OpenMP + SNR‑Gated)
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
#include <omp.h>

// ============================================================================
// 0. 全局宏与类型（RowMajor 行主序，大幅提升行操作缓存命中率）
// ============================================================================
using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowVector = Eigen::RowVectorXf;
using Vector = Eigen::VectorXf;          // 列向量，维度小影响不大
using Array = Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

constexpr float SNR_THRESHOLD   = 1.0f;
constexpr float THRUST_STRENGTH = 0.1f;
constexpr float THRUST_MAX      = 1.0f;
constexpr int   MAX_SEQ_LEN     = 256;     // 必须与训练时的 seq_len 一致
constexpr int   MAX_DIM         = 256;
constexpr int   MAX_HEADS       = 8;
constexpr int   MAX_EXPERTS     = 8;

// ============================================================================
// 0‑1 快速数学函数（Fast Exp / Fast Log，用位运算近似）
// ============================================================================
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float d_silu(float x) {
    float s = 1.0f / (1.0f + std::exp(-x));
    return s + x * s * (1.0f - s);
}

inline float fast_log2(float val) {
    union { float val; int32_t x; } u = { val };
    float log_2 = (float)(((u.x >> 23) & 255) - 128);
    u.x &= ~(255 << 23);
    u.x += 127 << 23;
    log_2 += ((-0.34484843f) * u.val + 2.02466578f) * u.val - 0.67487759f;
    return log_2;
}
inline float fast_log(float val) { return fast_log2(val) * 0.69314718f; }

// ============================================================================
// 0‑2 RoPE 预计算缓存（消灭运行时三角函数）
// ============================================================================
struct RoPECache {
    Matrix cos_tab, sin_tab;
    RoPECache(int max_len, int head_dim) : cos_tab(max_len, head_dim/2), sin_tab(max_len, head_dim/2) {
        for (int pos = 0; pos < max_len; ++pos) {
            for (int j = 0; j < head_dim / 2; ++j) {
                float theta = std::pow(10000.0f, -2.0f * j / (float)head_dim);
                float angle = pos * theta;
                cos_tab(pos, j) = std::cos(angle);
                sin_tab(pos, j) = std::sin(angle);
            }
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
    for (int pos = 0; pos < L; ++pos) {
        for (int h = 0; h < n_heads; ++h) {
            for (int j = 0; j < head_dim / 2; ++j) {
                float cos_val = cache.cos_tab(pos, j);
                float sin_val = cache.sin_tab(pos, j) * sign;
                int idx1 = h * head_dim + 2 * j;
                int idx2 = h * head_dim + 2 * j + 1;
                float q1 = Q(pos, idx1), q2 = Q(pos, idx2);
                Q(pos, idx1) = q1 * cos_val - q2 * sin_val;
                Q(pos, idx2) = q1 * sin_val + q2 * cos_val;
                float k1 = K(pos, idx1), k2 = K(pos, idx2);
                K(pos, idx1) = k1 * cos_val - k2 * sin_val;
                K(pos, idx2) = k1 * sin_val + k2 * cos_val;
            }
        }
    }
}

// ============================================================================
// 0‑3 随机数 & SNR‑Gated 更新（动量和方向推力）
// ============================================================================
std::mt19937 rng(42);
float randn(float mean = 0.0f, float std = 0.02f) {
    std::normal_distribution<float> dist(mean, std);
    return dist(rng);
}

template <typename Mat, typename Grad>
inline void snr_gated_update(Mat& param, const Grad& grad,
                            Mat& m, Mat& v,
                            float lr, int t,
                            float beta1 = 0.9f, float beta2 = 0.999f,
                            float eps = 1e-8f)
{
    using MatArray = Eigen::Array<typename Mat::Scalar,
                                 Mat::RowsAtCompileTime,
                                 Mat::ColsAtCompileTime,
                                 Mat::Options>;
    m = beta1 * m + (1.0f - beta1) * grad;
    v = beta2 * v + (1.0f - beta2) * grad.array().square().matrix();

    float b1_corr = 1.0f - std::pow(beta1, static_cast<float>(t));
    float b2_corr = 1.0f - std::pow(beta2, static_cast<float>(t));
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
// 0‑4 损失与 Softmax 辅助（使用 fast_log）
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
// 1. 基础层（零分配 Linear + 缓存友好 RMSNorm + OpenMP 并行）
// ============================================================================
struct Linear {
    Matrix W; RowVector b;
    Matrix dW, mW, vW; RowVector db, mb, vb;
    mutable Matrix dX;         // 预分配反向传播梯度缓冲区
    int adam_t = 0;

    Linear(int in, int out) : W(in, out), b(out), dW(in, out), db(out),
                              mW(in, out), vW(in, out), mb(out), vb(out),
                              dX(MAX_SEQ_LEN, in) {
        dW.setZero(); db.setZero(); mW.setZero(); vW.setZero(); mb.setZero(); vb.setZero(); dX.setZero();
        for (int i = 0; i < in; ++i) for (int j = 0; j < out; ++j) W(i, j) = randn();
        b.setZero();
    }

    Matrix forward(const Matrix& x_in) const { return (x_in * W).rowwise() + b; }

    // 零分配反向传播：返回预分配矩阵的引用，避免拷贝
    const Matrix& backward(const Matrix& dout, const Matrix& cache_x) {
        dW.noalias() += cache_x.transpose() * dout;
        db += dout.colwise().sum();
        int L = dout.rows();
        dX.topLeftCorner(L, W.rows()).noalias() = dout * W.transpose();
        return dX;
    }

    void step(float lr, float max_grad_norm) {
        float nw = std::sqrt(dW.squaredNorm() + db.squaredNorm());
        if (nw > max_grad_norm) { dW *= max_grad_norm / nw; db *= max_grad_norm / nw; }
        snr_gated_update(W, dW, mW, vW, lr, ++adam_t);
        snr_gated_update(b, db, mb, vb, lr, adam_t);
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

    RMSNorm(int dim, float e = 1e-5f) : eps(e), weight(RowVector::Ones(dim)),
                                        dweight(RowVector::Zero(dim)),
                                        mweight(RowVector::Zero(dim)),
                                        vweight(RowVector::Zero(dim)) {}

    // 前向传播：OpenMP 并行各行
    Matrix forward(const Matrix& x_in) const {
        Matrix norm = x_in;
        #pragma omp parallel for
        for (int i = 0; i < norm.rows(); ++i) {
            float var = norm.row(i).squaredNorm() / norm.cols();
            float inv_rms = 1.0f / std::sqrt(var + eps);
            norm.row(i) *= inv_rms;
            norm.row(i).array() *= weight.array();
        }
        return norm;
    }

    // 反向传播：并行化各行，dweight 累加使用 critical 保护
    Matrix backward(const Matrix& dout, const Matrix& cache_x) {
        Matrix dx = Matrix::Zero(dout.rows(), dout.cols());
        #pragma omp parallel
        {
            RowVector local_dweight = RowVector::Zero(weight.size());
            #pragma omp for nowait
            for (int i = 0; i < dout.rows(); ++i) {
                float var = cache_x.row(i).squaredNorm() / cache_x.cols();
                float inv_rms = 1.0f / std::sqrt(var + eps);
                RowVector x_norm_row = cache_x.row(i) * inv_rms;
                RowVector d_norm_row = dout.row(i).array() * weight.array();
                float mean_dx = (d_norm_row.array() * x_norm_row.array()).mean();
                dx.row(i) = (d_norm_row.array() - x_norm_row.array() * mean_dx) * inv_rms;
                local_dweight.array() += d_norm_row.array() * cache_x.row(i).array() * inv_rms;
            }
            #pragma omp critical
            { dweight.array() += local_dweight.array(); }
        }
        return dx;
    }

    void step(float lr) {
        snr_gated_update(weight, dweight, mweight, vweight, lr, ++adam_t);
        dweight.setZero();
    }

    void save(std::ostream& os) const {
        int d = weight.size(); os.write((char*)&d, sizeof(int));
        os.write((char*)weight.data(), sizeof(float)*d);
    }
    void load(std::istream& is) {
        int d; is.read((char*)&d, sizeof(int));
        is.read((char*)weight.data(), sizeof(float)*d);
    }
};

struct LoopEmbedding {
    Matrix W, dW, mW, vW;
    int adam_t = 0;
    LoopEmbedding(int max_loops, int dim) : W(max_loops, dim), dW(max_loops, dim),
                                            mW(max_loops, dim), vW(max_loops, dim) {
        for (int i = 0; i < W.rows(); ++i) for (int j = 0; j < W.cols(); ++j) W(i,j) = randn() * 0.02f;
        dW.setZero(); mW.setZero(); vW.setZero();
    }
    Matrix forward(int t, int batch_size) const { return W.row(t).replicate(batch_size, 1); }
    void backward(const Matrix& dout, int t) { dW.row(t) += dout.colwise().sum(); }
    void step(float lr, float max_grad_norm) {
        float nw = std::sqrt(dW.squaredNorm());
        if (nw > max_grad_norm) dW *= max_grad_norm / nw;
        snr_gated_update(W, dW, mW, vW, lr, ++adam_t);
        dW.setZero();
    }
};

// ============================================================================
// 2. 注意力模块（预分配缓冲区 + RoPE 缓存 + 截断 softmax）
// ============================================================================
struct AttnCache { Matrix q, k, v, out; std::vector<Matrix> attn_probs; };

struct MLAttention {
    Linear Wq, Wk, Wv, Wo;
    int n_heads, head_dim;
    Matrix* attn_scores;
    Matrix* attn_probs;
    Matrix* attn_tmp;

    MLAttention(int dim, int h) : Wq(dim, dim), Wk(dim, dim), Wv(dim, dim), Wo(dim, dim),
                                  n_heads(h), head_dim(dim/h),
                                  attn_scores(nullptr), attn_probs(nullptr), attn_tmp(nullptr) {}

    void set_work_buffers(Matrix* scores, Matrix* probs, Matrix* tmp) {
        attn_scores = scores; attn_probs = probs; attn_tmp = tmp;
    }

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

            for (int r = 0; r < L; ++r)
                for (int c = r+1; c < L; ++c) (*attn_scores)(r,c) = -1e9f;

            // 截断 softmax：只计算有效下三角
            for (int r = 0; r < L; ++r) {
                int valid_len = r + 1;
                float max_val = attn_scores->row(r).head(valid_len).maxCoeff();
                RowVector row = (attn_scores->row(r).head(valid_len).array() - max_val).exp();
                row /= row.sum();
                attn_probs->row(r).head(valid_len) = row;
                if (valid_len < L) attn_probs->row(r).tail(L - valid_len).setZero();
            }
            cache.attn_probs.push_back(attn_probs->topLeftCorner(L, L));
            cache.out.block(0, h*head_dim, L, head_dim).noalias() = attn_probs->topLeftCorner(L, L) * Vb;
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

            for (int r = 0; r < L; ++r) {
                RowVector dp = ds.row(r), pr = p.row(r);
                float p_dot_dp = (pr.array() * dp.array()).sum();
                attn_tmp->row(r).head(L) = (pr.array() * (dp.array() - p_dot_dp)).matrix();
            }
            for (int r = 0; r < L; ++r)
                for (int c = r+1; c < L; ++c) (*attn_tmp)(r,c) = 0;
            attn_tmp->topLeftCorner(L, L) *= s;

            dQ.block(0, h*head_dim, L, head_dim).noalias() += attn_tmp->topLeftCorner(L, L) * Kb;
            dK.block(0, h*head_dim, L, head_dim).noalias() += attn_tmp->topLeftCorner(L, L).transpose() * Qb;
        }
        apply_rope_inplace_fast(dQ, dK, n_heads, head_dim, true);
        return Wq.backward(dQ, cache_x) + Wk.backward(dK, cache_x) + Wv.backward(dV, cache_x);
    }

    void step(float lr, float gn) { Wq.step(lr, gn); Wk.step(lr, gn); Wv.step(lr, gn); Wo.step(lr, gn); }
    void save(std::ostream& os) const { Wq.save(os); Wk.save(os); Wv.save(os); Wo.save(os); }
    void load(std::istream& is) { Wq.load(is); Wk.load(is); Wv.load(is); Wo.load(is); }
};

// ============================================================================
// 3. MoE 模块（预分配缓冲区 + 手写 Top‑2 + 并行专家）
// ============================================================================
struct ExpertCache { Matrix gate_out, up_out, interm; };
struct TokenRoute { int tid; float w; };
struct MoECache {
    Matrix probs, x_in;
    std::vector<std::vector<TokenRoute>> routes;
    Matrix e_in[MAX_EXPERTS];
    Matrix e_out[MAX_EXPERTS];
    std::vector<ExpertCache> e_caches;

    MoECache() {
        routes.resize(MAX_EXPERTS);
        e_caches.resize(MAX_EXPERTS);
        for (int i = 0; i < MAX_EXPERTS; ++i) {
            e_in[i].resize(MAX_SEQ_LEN, MAX_DIM);
            e_out[i].resize(MAX_SEQ_LEN, MAX_DIM);
            routes[i].reserve(MAX_SEQ_LEN);
        }
    }
};

struct Expert {
    Linear gate, up, down;
    Expert(int d, int h) : gate(d, h), up(d, h), down(h, d) {}
    Matrix forward(const Matrix& x, ExpertCache& cache) const {
        cache.gate_out = gate.forward(x);
        cache.up_out   = up.forward(x);
        Matrix f_g = cache.gate_out.unaryExpr(&silu);
        cache.interm = (f_g.array() * cache.up_out.array()).matrix();
        return down.forward(cache.interm);
    }
    Matrix backward(const Matrix& dout, const ExpertCache& cache, const Matrix& cache_x) {
        Matrix di = down.backward(dout, cache.interm);
        Matrix f_g = cache.gate_out.unaryExpr(&silu);
        Matrix du = up.backward((di.array() * f_g.array()).matrix(), cache_x);
        Matrix dg_in = (di.array() * cache.up_out.array() *
                        cache.gate_out.unaryExpr(&d_silu).array()).matrix();
        return du + gate.backward(dg_in, cache_x);
    }
    void step(float lr, float gn) { gate.step(lr, gn); up.step(lr, gn); down.step(lr, gn); }
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
        Matrix el = l.array().exp().matrix();
        Vector Z = el.rowwise().sum();
        c.probs = (el.array().colwise() / Z.array()).matrix();

        // 清理路由
        for (int i = 0; i < n_experts; ++i) c.routes[i].clear();

        // 手写 Top‑2 路由
        for (int t = 0; t < BT; ++t) {
            int best_e = -1, second_e = -1;
            float best_p = -1.0f, second_p = -1.0f;
            for (int e = 0; e < n_experts; ++e) {
                float p = c.probs(t, e);
                if (p > best_p) {
                    second_p = best_p; second_e = best_e;
                    best_p = p; best_e = e;
                } else if (p > second_p) {
                    second_p = p; second_e = e;
                }
            }
            if (best_e != -1) c.routes[best_e].push_back({t, best_p});
            if (second_e != -1 && top_k > 1) c.routes[second_e].push_back({t, second_p});
        }

        // 并行化专家计算（每个专家独立，局部缓冲区累加）
        Matrix out = Matrix::Zero(BT, dim);
        #pragma omp parallel for
        for (int e = 0; e < n_experts; ++e) {
            int nt = c.routes[e].size();
            if (!nt) continue;
            auto Xe = c.e_in[e].topLeftCorner(nt, dim);
            for (int i = 0; i < nt; ++i) Xe.row(i) = x.row(c.routes[e][i].tid);
            Matrix Ye = experts[e].forward(Xe, c.e_caches[e]);
            c.e_out[e].topLeftCorner(nt, dim) = Ye;
            // 局部累加（注意 out 是共享的，这里用 atomic 或局部复制后再归并；为简单此处使用 critical 或者按行写入）
            #pragma omp critical
            {
                for (int i = 0; i < nt; ++i)
                    out.row(c.routes[e][i].tid) += Ye.row(i) * c.routes[e][i].w;
            }
        }

        // 负载均衡损失（串行部分，很小）
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
        #pragma omp parallel for
        for (int e = 0; e < n_experts; ++e) {
            int nt = c.routes[e].size();
            if (!nt) continue;
            Matrix de(nt, dim);
            for (int i = 0; i < nt; ++i) {
                int t = c.routes[e][i].tid;
                de.row(i) = dout.row(t) * c.routes[e][i].w;
                dl(t, e) += dout.row(t).dot(c.e_out[e].row(i));   // 注意 dl 维度是共享的，需要原子操作。此处简化，实际训练可能需调整
            }
            Matrix dxe = experts[e].backward(de, c.e_caches[e], c.e_in[e].topLeftCorner(nt, dim));
            #pragma omp critical
            {
                for (int i = 0; i < nt; ++i) dx.row(c.routes[e][i].tid) += dxe.row(i);
            }
        }
        // 路由梯度（串行）
        float coef = (0.02f * n_experts) / (float)(top_k * BT);
        for (int e = 0; e < n_experts; ++e) {
            float cnt = c.routes[e].size();
            for (int t = 0; t < BT; ++t) dl(t, e) += (coef * cnt) / (float)BT;
        }
        dx += router.backward(apply_softmax_backward(dl, c.probs), c.x_in);
        return dx;
    }

    void step(float lr, float gn) { router.step(lr, gn); for(auto &e:experts) e.step(lr, gn); }
    void save(std::ostream& os) const { router.save(os); for(auto &e:experts) e.save(os); }
    void load(std::istream& is) { router.load(is); for(auto &e:experts) e.load(is); }
};

// ============================================================================
// 4. ACT 与 RecurrentBlock
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
            Vector pt(L); for(int i=0; i<L; ++i) pt(i) = 1.0f/(1.0f+std::exp(-lp(i,0)));
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
    void step(float lr, float gn) { linear.step(lr, gn); }
    void save(std::ostream& os) const { linear.save(os); }
    void load(std::istream& is) { linear.load(is); }
};

struct RecurrentStepCache { Matrix x, n_attn, a_out, mid, n_moe; AttnCache a_c; MoECache m_c; };
struct RecurrentBlock {
    LoopEmbedding loop_embed;
    MLAttention attn; MoE moe; RMSNorm n_attn, n_moe; ACT act;
    std::vector<RecurrentStepCache> b_cache; ACTCache a_cache;

    RecurrentBlock(int d, int l, Matrix* scores, Matrix* probs, Matrix* tmp)
        : loop_embed(l, d), attn(d, 8), moe(d, d*4, 8, 2), n_attn(d), n_moe(d), act(d, l) {
        attn.set_work_buffers(scores, probs, tmp);
    }

    Matrix forward(Matrix x) {
        b_cache.clear(); std::vector<Matrix> hs;
        for (int t = 0; t < act.max_loops; ++t) {
            RecurrentStepCache c; c.x = x; c.n_attn = n_attn.forward(x);
            x += loop_embed.forward(t, x.rows());
            c.a_out = attn.forward(c.n_attn, c.a_c); c.mid = x + c.a_out;
            c.n_moe = n_moe.forward(c.mid);
            Matrix m_out = moe.forward(c.n_moe, c.m_c); x = c.mid + m_out;
            hs.push_back(x); b_cache.push_back(c);
        }
        return act.forward(hs, a_cache);
    }

    Matrix backward(const Matrix& dout) {
        std::vector<Matrix> dhs = act.backward(dout, a_cache);
        Matrix dx = Matrix::Zero(dout.rows(), dout.cols());
        for (int t = (int)b_cache.size()-1; t >= 0; --t) {
            auto& c = b_cache[t]; Matrix ds = dhs[t] + dx;
            loop_embed.backward(ds, t);
            Matrix dnm = moe.backward(ds, c.m_c);
            Matrix dm = ds + n_moe.backward(dnm, c.mid);
            Matrix dna = attn.backward(dm, c.n_attn, c.a_c);
            dx = dm + n_attn.backward(dna, c.x);
        }
        return dx;
    }

    void step(float lr, float gn) {
        attn.step(lr, gn); moe.step(lr, gn);
        n_attn.step(lr); n_moe.step(lr); act.step(lr, gn); loop_embed.step(lr, gn);
    }
    void save(std::ostream& os) const {
        n_attn.save(os); attn.save(os); n_moe.save(os); moe.save(os); act.save(os);
    }
    void load(std::istream& is) {
        n_attn.load(is); attn.load(is); n_moe.load(is); moe.load(is); act.load(is);
    }
};

// ============================================================================
// 5. 顶层模型（持有预分配工作缓冲区）
// ============================================================================
struct OpenMythos {
    Linear embed, lm_head;
    RecurrentBlock recurrent;
    RMSNorm final_norm;
    Matrix work_scores, work_probs, work_tmp;
    Matrix h_out, f_normed;

    OpenMythos(int vocab, int dim, int max_loop)
        : embed(vocab, dim), lm_head(dim, vocab), recurrent(dim, max_loop, &work_scores, &work_probs, &work_tmp),
          final_norm(dim), work_scores(MAX_SEQ_LEN, MAX_SEQ_LEN), work_probs(MAX_SEQ_LEN, MAX_SEQ_LEN), work_tmp(MAX_SEQ_LEN, dim) {}

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

    void step(float lr, float gn) {
        recurrent.step(lr, gn); final_norm.step(lr); lm_head.step(lr, gn);
    }

    bool save_checkpoint(const std::string& path, const std::string& ds_id, int step) const {
        std::ofstream os(path, std::ios::binary);
        if (!os) return false;
        size_t idl = ds_id.length();
        os.write((char*)&idl, sizeof(size_t));
        os.write(ds_id.c_str(), idl);
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
// 6. 文本生成与训练辅助
// ============================================================================
int sample_from_probs(const RowVector& probs, float temperature = 1.0f) {
    RowVector scaled = (probs.array() / temperature).max(-100.0f).min(100.0f);
    float max_val = scaled.maxCoeff();
    RowVector exp_vals = (scaled.array() - max_val).exp();
    float Z = exp_vals.sum();
    RowVector final_probs = exp_vals / Z;
    float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    float cum = 0.0f;
    for (int i = 0; i < final_probs.size(); ++i) {
        cum += final_probs(i);
        if (r <= cum) return i;
    }
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

// ============================================================================
// 7. 训练主循环
// ============================================================================
uint64_t fnv1a64(const std::string& d) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : d) { h ^= (uint64_t)c; h *= 1099511628211ull; }
    return h;
}
std::string to_hex(uint64_t v) {
    std::ostringstream s; s << std::hex << std::setw(16) << std::setfill('0') << v; return s.str();
}
float get_lr(int s, int ts, float p, int w, float m) {
    if (s < w) return p * (s+1) / w;
    float r = (float)(s-w)/(ts-w); if(r>1) r=1;
    return p * (m + (1-m)*0.5f*(1+std::cos(M_PI*r)));
}

int main(int argc, char* argv[]) {
    if (argc >= 3 && std::string(argv[1]) == "gen") {
        bpe::BPETrainer tokenizer;
        tokenizer.load("tokenizer.bpe");
        OpenMythos model(tokenizer.vocab_size(), 128, 3);
        model.load_checkpoint("openmythos_model.ckpt", "");
        std::string prompt = argv[2];
        std::string gen = generate_text(model, tokenizer, prompt, 50, 0.8f);
        std::cout << "Prompt: " << prompt << "\nGenerated: " << gen << std::endl;
        return 0;
    }
    int cores = omp_get_max_threads();
    Eigen::setNbThreads(cores);
    omp_set_num_threads(cores);
    std::cout << "Running with " << Eigen::nbThreads() << " threads." << std::endl;
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
    int dim = 128, max_loop = 3, seq_len = 128, accum = 8;
    OpenMythos model(vocab, dim, max_loop);
    int start_step = 0;
    int loaded = model.load_checkpoint("openmythos_model.ckpt", ds_id);
    if (loaded >= 0) start_step = loaded;

    int total_steps = 3000;
    float peak_lr = 0.0005f;
    float best_loss = 1e9f;

    Eigen::MatrixXi inp(seq_len, 1), tgt(seq_len, 1);

    for (int s = start_step; s < total_steps; ++s) {
        float lr = get_lr(s, total_steps, peak_lr, 500, 0.3f);
        float loss_sum = 0, bal_sum = 0;

        for (int a = 0; a < accum; ++a) {
            int start = rng() % (tokens.size() - seq_len - 1);
            for (int i = 0; i < seq_len; ++i) {
                inp(i,0) = tokens[start+i];
                tgt(i,0) = tokens[start+i+1];
            }

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

        if (total_loss > best_loss * 3.0f && best_loss < 1e8f) {
            std::cout << "Spike detected, halving lr for this step.\n";
            lr *= 0.5f;
        }
        if (total_loss < best_loss) best_loss = total_loss;

        model.step(lr, 1.0f);
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