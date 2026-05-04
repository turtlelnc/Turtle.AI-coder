// openmythos_train.cpp – 纯 C++ BPTT 终极无状态版 (彻底修复 Eigen 广播断言)
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

using Matrix = Eigen::MatrixXf;
using RowVector = Eigen::RowVectorXf;
using Vector = Eigen::VectorXf;
using Array = Eigen::ArrayXf;

// ============================================================================
// 0. 工具函数与激活函数
// ============================================================================
std::mt19937 rng(42);
float randn(float mean = 0.0f, float std = 0.02f) {
    std::normal_distribution<float> dist(mean, std);
    return dist(rng);
}

inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
inline float d_silu(float x) {
    float s = 1.0f / (1.0f + std::exp(-x));
    return s + x * s * (1.0f - s);
}

float cross_entropy_backward(const RowVector& logits, int target, RowVector& dlogits) {
    float max_logit = logits.maxCoeff();
    RowVector shifted = (logits.array() - max_logit).matrix();
    RowVector exp_shifted = shifted.array().exp().matrix();
    float Z = exp_shifted.sum();
    RowVector log_softmax = (shifted.array() - std::log(Z)).matrix();
    RowVector softmax = exp_shifted / Z;

    float loss = -log_softmax(target);
    dlogits = softmax;
    dlogits(target) -= 1.0f;
    return loss;
}

Matrix apply_softmax_backward(const Matrix& dout, const Matrix& probs) {
    Matrix dlogits = Matrix::Zero(dout.rows(), dout.cols());
    for(int i = 0; i < dout.rows(); ++i) {
        RowVector dp = dout.row(i);
        RowVector p = probs.row(i);
        float p_dot_dp = (p.array() * dp.array()).sum();
        dlogits.row(i) = (p.array() * (dp.array() - p_dot_dp)).matrix();
    }
    return dlogits;
}

template <typename Mat, typename Grad>
inline void adam_update(Mat& param, const Grad& grad, Mat& m, Mat& v,
                        float lr, int t, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) {
    m = beta1 * m + (1.0f - beta1) * grad;
    v = beta2 * v + (1.0f - beta2) * grad.array().square().matrix();
    float b1_corr = 1.0f - std::pow(beta1, static_cast<float>(t));
    float b2_corr = 1.0f - std::pow(beta2, static_cast<float>(t));
    auto m_hat = m.array() / b1_corr;
    auto v_hat = v.array() / b2_corr;
    param -= (lr * m_hat / (v_hat.sqrt() + eps)).matrix();
}

void apply_rope_inplace(Matrix& Q, Matrix& K, int num_heads, int head_dim, bool inverse = false) {
    int seq_len = Q.rows();
    float sign = inverse ? -1.0f : 1.0f;
    for (int pos = 0; pos < seq_len; ++pos) {
        for (int h = 0; h < num_heads; ++h) {
            for (int j = 0; j < head_dim / 2; ++j) {
                float theta = std::pow(10000.0f, -2.0f * j / static_cast<float>(head_dim));
                float angle = pos * theta * sign;
                float cos_val = std::cos(angle);
                float sin_val = std::sin(angle);
                int idx1 = h * head_dim + 2 * j;
                int idx2 = h * head_dim + 2 * j + 1;
                float q1 = Q(pos, idx1); float q2 = Q(pos, idx2);
                Q(pos, idx1) = q1 * cos_val - q2 * sin_val; Q(pos, idx2) = q1 * sin_val + q2 * cos_val;
                float k1 = K(pos, idx1); float k2 = K(pos, idx2);
                K(pos, idx1) = k1 * cos_val - k2 * sin_val; K(pos, idx2) = k1 * sin_val + k2 * cos_val;
            }
        }
    }
}

// ============================================================================
// 1. 基础网络层
// ============================================================================
struct Linear {
    Matrix W; RowVector b;
    Matrix dW, mW, vW; RowVector db, mb, vb;
    int adam_t = 0;

    Linear(int in, int out) : W(in, out), b(out), dW(in, out), db(out) {
        dW.setZero(); db.setZero(); mW.setZero(in, out); vW.setZero(in, out);
        mb.setZero(out); vb.setZero(out);
        for (int i = 0; i < in; ++i) for (int j = 0; j < out; ++j) W(i, j) = randn();
        b.setZero();
    }

    Matrix forward(const Matrix& x_in) const { return (x_in * W).rowwise() + b; }
    Matrix backward(const Matrix& dout, const Matrix& cache_x) {
        dW.noalias() += cache_x.transpose() * dout;
        db += dout.colwise().sum();
        return dout * W.transpose();
    }

    void save(std::ostream& os) const {
        int r = W.rows(), c = W.cols();
        os.write((char*)&r, sizeof(int)); os.write((char*)&c, sizeof(int));
        os.write((char*)W.data(), sizeof(float)*W.size()); os.write((char*)b.data(), sizeof(float)*b.size());
        os.write((char*)mW.data(), sizeof(float)*mW.size()); os.write((char*)vW.data(), sizeof(float)*vW.size());
        os.write((char*)mb.data(), sizeof(float)*mb.size()); os.write((char*)vb.data(), sizeof(float)*vb.size());
        os.write((char*)&adam_t, sizeof(int));
    }
    void load(std::istream& is) {
        int r, c; is.read((char*)&r, sizeof(int)); is.read((char*)&c, sizeof(int));
        is.read((char*)W.data(), sizeof(float)*W.size()); is.read((char*)b.data(), sizeof(float)*b.size());
        is.read((char*)mW.data(), sizeof(float)*mW.size()); is.read((char*)vW.data(), sizeof(float)*vW.size());
        is.read((char*)mb.data(), sizeof(float)*mb.size()); is.read((char*)vb.data(), sizeof(float)*vb.size());
        is.read((char*)&adam_t, sizeof(int));
    }

    void step(float lr, float max_grad_norm) {
        float nw = std::sqrt(dW.squaredNorm() + db.squaredNorm());
        if (nw > max_grad_norm) { dW *= max_grad_norm / nw; db *= max_grad_norm / nw; }
        adam_update(W, dW, mW, vW, lr, ++adam_t);
        adam_update(b, db, mb, vb, lr, adam_t);
        dW.setZero(); db.setZero();
    }
};
struct ExpertCache {
    Matrix gate_out;
    Matrix up_out;
    Matrix interm;
};
struct RMSNorm {
    float eps;
    RowVector weight, dweight, mweight, vweight;
    int adam_t = 0;

    RMSNorm(int dim, float e = 1e-5f) : eps(e), weight(RowVector::Ones(dim)), dweight(RowVector::Zero(dim)) {
        mweight.setZero(dim); vweight.setZero(dim);
    }
    
    // 【核心修复】：拆分计算，显式按行广播 weight，彻底解决 CwiseBinaryOp 崩溃
    Matrix forward(const Matrix& x_in) const {
        Array rms = (x_in.array().square().rowwise().mean() + eps).sqrt().max(1e-8f);
        Matrix norm = (x_in.array().colwise() / rms).matrix();
        norm.array().rowwise() *= weight.array(); 
        return norm;
    }
    
    // 【核心修复】：为安全起见，将 dx 也进行显式的 matrix 转换
    Matrix backward(const Matrix& dout, const Matrix& cache_x) {
        int D = weight.size();
        Array rms = (cache_x.array().square().rowwise().mean() + eps).sqrt().max(1e-8f);
        Matrix d_norm = (dout.array().rowwise() * weight.array()).matrix();
        Matrix x_norm = (cache_x.array().colwise() / rms).matrix();
        
        Array mean_dx = (d_norm.array() * x_norm.array()).rowwise().mean();
        
        Matrix dx = ((d_norm.array() - x_norm.array().colwise() * mean_dx).colwise() / rms).matrix();
        dweight += (dout.array() * x_norm.array()).colwise().sum().matrix();
        return dx;
    }
    void save(std::ostream& os) const {
        int d = weight.size(); os.write((char*)&d, sizeof(int));
        os.write((char*)weight.data(), sizeof(float)*d);
        os.write((char*)mweight.data(), sizeof(float)*d);
        os.write((char*)vweight.data(), sizeof(float)*d);
        os.write((char*)&adam_t, sizeof(int));
    }
    void load(std::istream& is) {
        int d; is.read((char*)&d, sizeof(int));
        is.read((char*)weight.data(), sizeof(float)*d);
        is.read((char*)mweight.data(), sizeof(float)*d);
        is.read((char*)vweight.data(), sizeof(float)*d);
        is.read((char*)&adam_t, sizeof(int));
    }
    void step(float lr) { adam_update(weight, dweight, mweight, vweight, lr, ++adam_t); dweight.setZero(); }
};

// ============================================================================
// 2. 核心组件
// ============================================================================

struct AttnCache { Matrix q, k, v, out; std::vector<Matrix> attn_probs; };
struct MLAttention {
    Linear Wq, Wk, Wv, Wo;
    int n_heads, head_dim;
    MLAttention(int dim, int h) : Wq(dim, dim), Wk(dim, dim), Wv(dim, dim), Wo(dim, dim), n_heads(h), head_dim(dim/h) {}
    Matrix forward(const Matrix& x, AttnCache& cache) const {
        int L = x.rows(); cache.q = Wq.forward(x); cache.k = Wk.forward(x); cache.v = Wv.forward(x);
        apply_rope_inplace(cache.q, cache.k, n_heads, head_dim);
        cache.attn_probs.clear(); cache.out.setZero(L, Wv.W.cols());
        float s = 1.0f/std::sqrt((float)head_dim);
        for(int h=0; h<n_heads; ++h) {
            Matrix Q = cache.q.block(0, h*head_dim, L, head_dim);
            Matrix K = cache.k.block(0, h*head_dim, L, head_dim);
            Matrix scores = (Q * K.transpose()) * s;
            for(int r=0; r<L; ++r) for(int c=r+1; c<L; ++c) scores(r,c) = -1e9f;
            Matrix p(L, L);
            for(int r=0; r<L; ++r) {
                float m = scores.row(r).maxCoeff();
                RowVector row = (scores.row(r).array()-m).exp();
                p.row(r) = row / row.sum();
            }
            cache.attn_probs.push_back(p);
            cache.out.block(0, h*head_dim, L, head_dim) = p * cache.v.block(0, h*head_dim, L, head_dim);
        }
        return Wo.forward(cache.out);
    }
    Matrix backward(const Matrix& dout, const Matrix& cache_x, const AttnCache& cache) {
        int L = dout.rows(); Matrix d_out = Wo.backward(dout, cache.out);
        Matrix dQ(L, d_out.cols()), dK(L, d_out.cols()), dV(L, d_out.cols());
        float s = 1.0f/std::sqrt((float)head_dim);
        for(int h=0; h<n_heads; ++h) {
            Matrix dh = d_out.block(0, h*head_dim, L, head_dim);
            const Matrix& p = cache.attn_probs[h];
            dV.block(0, h*head_dim, L, head_dim) = p.transpose() * dh;
            Matrix ds = dh * cache.v.block(0, h*head_dim, L, head_dim).transpose();
            Matrix dsp(L, L);
            for(int r=0; r<L; ++r) {
                RowVector dp = ds.row(r), pr = p.row(r);
                dsp.row(r) = (pr.array() * (dp.array() - (pr.array()*dp.array()).sum())).matrix();
            }
            for(int r=0; r<L; ++r) for(int c=r+1; c<L; ++c) dsp(r,c) = 0;
            dsp *= s;
            dQ.block(0, h*head_dim, L, head_dim) = dsp * cache.k.block(0, h*head_dim, L, head_dim);
            dK.block(0, h*head_dim, L, head_dim) = dsp.transpose() * cache.q.block(0, h*head_dim, L, head_dim);
        }
        apply_rope_inplace(dQ, dK, n_heads, head_dim, true);
        return Wq.backward(dQ, cache_x) + Wk.backward(dK, cache_x) + Wv.backward(dV, cache_x);
    }
    void save(std::ostream& os) const { Wq.save(os); Wk.save(os); Wv.save(os); Wo.save(os); }
    void load(std::istream& is) { Wq.load(is); Wk.load(is); Wv.load(is); Wo.load(is); }
    void step(float lr, float gn) { Wq.step(lr, gn); Wk.step(lr, gn); Wv.step(lr, gn); Wo.step(lr, gn); }
};
struct LoopEmbedding {
    Matrix W, dW, mW, vW;
    int adam_t = 0;

    LoopEmbedding(int max_loops, int dim) {
        W = Matrix::Random(max_loops, dim) * 0.02f;
        dW = Matrix::Zero(max_loops, dim);
        mW = Matrix::Zero(max_loops, dim);
        vW = Matrix::Zero(max_loops, dim);
    }

    Matrix forward(int t, int batch_size) const {
        return W.row(t).replicate(batch_size, 1);
    }

    void backward(const Matrix& dout, int t) {
        dW.row(t) += dout.colwise().sum();
    }

    void save(std::ostream& os) const {
        int r = W.rows(), c = W.cols();
        os.write((char*)&r, sizeof(int)); os.write((char*)&c, sizeof(int));
        os.write((char*)W.data(), sizeof(float)*W.size());
        os.write((char*)mW.data(), sizeof(float)*mW.size());
        os.write((char*)vW.data(), sizeof(float)*vW.size());
        os.write((char*)&adam_t, sizeof(int));
    }

    void load(std::istream& is) {
        int r, c; is.read((char*)&r, sizeof(int)); is.read((char*)&c, sizeof(int));
        is.read((char*)W.data(), sizeof(float)*W.size());
        is.read((char*)mW.data(), sizeof(float)*mW.size());
        is.read((char*)vW.data(), sizeof(float)*vW.size());
        is.read((char*)&adam_t, sizeof(int));
    }

    void step(float lr, float max_grad_norm) {
        float nw = std::sqrt(dW.squaredNorm());
        if (nw > max_grad_norm) { dW *= max_grad_norm / nw; }
        adam_update(W, dW, mW, vW, lr, ++adam_t);
        dW.setZero();
    }
};
struct Expert {
    Linear gate, up, down;
    Expert(int d, int h) : gate(d,h), up(d,h), down(h,d) {}
    
    Matrix forward(const Matrix& x, ExpertCache& cache) const {
        cache.gate_out = gate.forward(x);
        cache.up_out = up.forward(x);
        Matrix f_g = cache.gate_out.unaryExpr(&silu);
        cache.interm = (f_g.array() * cache.up_out.array()).matrix();
        return down.forward(cache.interm);
    }
    
    Matrix backward(const Matrix& dout, const ExpertCache& cache, const Matrix& cache_x) {
        // 直接从缓存取值，省去巨大的 gate.forward 和 up.forward 矩阵乘法开销
        Matrix di = down.backward(dout, cache.interm);
        Matrix f_g = cache.gate_out.unaryExpr(&silu);
        Matrix du = up.backward((di.array() * f_g.array()).matrix(), cache_x);
        Matrix dg_in = (di.array() * cache.up_out.array() * cache.gate_out.unaryExpr(&d_silu).array()).matrix();
        return du + gate.backward(dg_in, cache_x);
    }
    
    void save(std::ostream& os) const { gate.save(os); up.save(os); down.save(os); }
    void load(std::istream& is) { gate.load(is); up.load(is); down.load(is); }
    void step(float lr, float gn) { gate.step(lr, gn); up.step(lr, gn); down.step(lr, gn); }
};

struct TokenRoute { int tid; float w; };
struct MoECache { 
    Matrix probs, x_in; 
    std::vector<std::vector<TokenRoute>> routes; 
    std::vector<Matrix> e_in, e_out; 
    std::vector<ExpertCache> e_caches; // 新增：保存每个专家的激活缓存
};

struct MoE {
    int dim, n_experts, top_k; 
    float balance_loss = 0; 
    Linear router; 
    std::vector<Expert> experts;
    
    MoE(int d, int h, int ne, int tk) : dim(d), n_experts(ne), top_k(tk), router(d, ne) {
        for(int i=0; i<ne; ++i) experts.emplace_back(d, h);
    }
    
    Matrix forward(const Matrix& x, MoECache& c) {
        int BT = x.rows(); c.x_in = x; 
        Matrix l = router.forward(x);
        Matrix el = l.array().exp().matrix(); 
        Vector Z = el.rowwise().sum();
        c.probs = (el.array().colwise() / Z.array()).matrix();
        
        c.routes.assign(n_experts, {}); 
        c.e_in.resize(n_experts); 
        c.e_out.resize(n_experts);
        c.e_caches.resize(n_experts); // 初始化专家缓存空间
        
        for(int t=0; t<BT; ++t) {
            std::vector<std::pair<float, int>> p;
            for(int e=0; e<n_experts; ++e) p.push_back({c.probs(t,e), e});
            std::sort(p.rbegin(), p.rend());
            for(int k=0; k<top_k; ++k) c.routes[p[k].second].push_back({t, p[k].first});
        }
        
        Matrix out = Matrix::Zero(BT, dim);
        for(int e=0; e<n_experts; ++e) {
            int nt = c.routes[e].size(); 
            if(!nt) continue;
            Matrix Xe(nt, dim); 
            for(int i=0; i<nt; ++i) Xe.row(i) = x.row(c.routes[e][i].tid);
            c.e_in[e] = Xe; 
            
            // 传入对应专家的缓存对象
            Matrix Ye = experts[e].forward(Xe, c.e_caches[e]); 
            c.e_out[e] = Ye;
            
            for(int i=0; i<nt; ++i) out.row(c.routes[e][i].tid) += Ye.row(i) * c.routes[e][i].w;
        }
        
        RowVector count = RowVector::Zero(n_experts);
        for(int e=0; e<n_experts; ++e) count(e) += c.routes[e].size();
        RowVector mean_probs = c.probs.colwise().mean();
        RowVector scaled_count = count * (n_experts / (float)(top_k * BT));
        balance_loss = scaled_count.dot(mean_probs) * 0.02f;
        
        return out;
    }
    
    Matrix backward(const Matrix& dout, const MoECache& c) {
        int BT = dout.rows(); 
        Matrix dx = Matrix::Zero(BT, dim), dl = Matrix::Zero(BT, n_experts);
        for(int e=0; e<n_experts; ++e) {
            int nt = c.routes[e].size(); 
            if(!nt) continue;
            Matrix de(nt, dim); 
            for(int i=0; i<nt; ++i) {
                int t = c.routes[e][i].tid; 
                de.row(i) = dout.row(t) * c.routes[e][i].w;
                dl(t,e) += dout.row(t).dot(c.e_out[e].row(i));
            }
            // 传入前向计算时留下的缓存
            Matrix dxe = experts[e].backward(de, c.e_caches[e], c.e_in[e]);
            for(int i=0; i<nt; ++i) dx.row(c.routes[e][i].tid) += dxe.row(i);
        }
        float coef = (0.02f * n_experts) / (float)(top_k * BT);
        for(int e=0; e<n_experts; ++e) {
            float cnt = c.routes[e].size();
            for(int t=0; t<BT; ++t) dl(t, e) += (coef * cnt) / (float)BT;
        }
        dx += router.backward(apply_softmax_backward(dl, c.probs), c.x_in);
        return dx;
    }
    void save(std::ostream& os) const { router.save(os); for(auto& e:experts) e.save(os); }
    void load(std::istream& is) { router.load(is); for(auto& e:experts) e.load(is); }
    void step(float lr, float gn) { router.step(lr, gn); for(auto& e:experts) e.step(lr, gn); }
};

struct ACTCache { std::vector<Vector> p, w; std::vector<Matrix> hs; };
struct ACT {
    int max_loops; Linear linear;
    ACT(int d, int l) : max_loops(l), linear(d, 1) {}
    Matrix forward(const std::vector<Matrix>& hs, ACTCache& c) const {
        c.p.clear(); c.w.clear(); c.hs.clear();
        int L = hs[0].rows(); Matrix out = Matrix::Zero(L, hs[0].cols()); Vector rem = Vector::Ones(L);
        for(int t=0; t<(int)hs.size(); ++t) {
            c.hs.push_back(hs[t]); Matrix lp = linear.forward(hs[t]);
            Vector pt(L); for(int i=0; i<L; ++i) pt(i) = 1.0f/(1.0f+std::exp(-lp(i,0)));
            c.p.push_back(pt); Vector wt(L);
            for(int i=0; i<L; ++i) {
                if(t == (int)hs.size()-1) wt(i) = rem(i);
                else { wt(i) = pt(i)*rem(i); rem(i) -= wt(i); }
            }
            c.w.push_back(wt); for(int i=0; i<L; ++i) out.row(i) += hs[t].row(i)*wt(i);
        }
        return out;
    }
    std::vector<Matrix> backward(const Matrix& dout, const ACTCache& c) {
        int L = dout.rows(), D = dout.cols(); std::vector<Matrix> dhs(c.hs.size(), Matrix::Zero(L, D));
        for(int t=0; t<(int)c.hs.size(); ++t) {
            Matrix dlin(L, 1); for(int i=0; i<L; ++i) {
                dhs[t].row(i) = dout.row(i) * c.w[t](i);
                float dw = dout.row(i).dot(c.hs[t].row(i));
                dlin(i,0) = dw * (c.p[t](i)*(1.0f-c.p[t](i)));
            }
            dhs[t] += linear.backward(dlin, c.hs[t]);
        }
        return dhs;
    }
    void save(std::ostream& os) const { linear.save(os); }
    void load(std::istream& is) { linear.load(is); }
    void step(float lr, float gn) { linear.step(lr, gn); }
};

// ============================================================================
// 3. 模型
// ============================================================================

struct RecurrentStepCache { Matrix x, n_attn, a_out, mid, n_moe; AttnCache a_c; MoECache m_c; };
struct RecurrentBlock {
    LoopEmbedding loop_embed; // 【新增】
    MLAttention attn; MoE moe; RMSNorm n_attn, n_moe; ACT act;
    std::vector<RecurrentStepCache> b_cache; ACTCache a_cache;
    RecurrentBlock(int d, int l) : loop_embed(l, d), attn(d, 8), moe(d, d*4, 8, 2), n_attn(d), n_moe(d), act(d, l) {}
    
    Matrix forward(Matrix x) {
        b_cache.clear(); std::vector<Matrix> hs;
        for(int t=0; t<act.max_loops; ++t) {
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
        std::vector<Matrix> dhs = act.backward(dout, a_cache); Matrix dx = Matrix::Zero(dout.rows(), dout.cols());
        for(int t=(int)b_cache.size()-1; t>=0; --t) {
            auto& c = b_cache[t]; Matrix ds = dhs[t] + dx;
            loop_embed.backward(ds, t);
            Matrix dnm = moe.backward(ds, c.m_c);
            Matrix dm = ds + n_moe.backward(dnm, c.mid);
            Matrix dna = attn.backward(dm, c.n_attn, c.a_c);
            dx = dm + n_attn.backward(dna, c.x);
        }
        return dx;
    }
    void save(std::ostream& os) const { n_attn.save(os); attn.save(os); n_moe.save(os); moe.save(os); act.save(os); }
    void load(std::istream& is) { n_attn.load(is); attn.load(is); n_moe.load(is); moe.load(is); act.load(is); }
    void step(float lr, float gn) { attn.step(lr, gn); moe.step(lr, gn); n_attn.step(lr); n_moe.step(lr); act.step(lr, gn);loop_embed.step(lr, gn); }
};

struct OpenMythos {
    Linear embed, lm_head; RecurrentBlock recurrent; RMSNorm final_norm;
    Matrix h_out, f_normed;
    OpenMythos(int v, int d, int l) : embed(v, d), lm_head(d, v), recurrent(d, l), final_norm(d) {}

    Matrix forward(const Eigen::MatrixXi& ids) {
        int BT = ids.rows(); Matrix x(BT, embed.W.cols());
        for(int i=0; i<BT; ++i) x.row(i) = embed.W.row(ids(i,0));
        h_out = recurrent.forward(x); f_normed = final_norm.forward(h_out);
        return lm_head.forward(f_normed);
    }
    
    void backward(const Matrix& dlogits, const Eigen::MatrixXi& ids) {
        Matrix dx = lm_head.backward(dlogits, f_normed);
        dx = final_norm.backward(dx, h_out);
        dx = recurrent.backward(dx);
        for(int i=0; i<ids.rows(); ++i) embed.dW.row(ids(i,0)) += dx.row(i);

        // 【核心修复：Weight Tying 梯度同步】
        // 将输入端的梯度叠加到输出端，保证 Adam 优化器拿到的是完整梯度
        lm_head.dW += embed.dW.transpose();
        
        // 清空 embed 的梯度，防止 embed.step() 产生幽灵计算
        embed.dW.setZero(); 
    }
    
    void step(float lr, float gn) { 
        recurrent.step(lr, gn); 
        final_norm.step(lr); 
        lm_head.step(lr, gn); 
        // embed.step(lr, gn); 可以注释掉，因为它的权重完全由 lm_head 决定了
    }

    bool save_checkpoint(const std::string& path, const std::string& ds_id, int step) const {
        std::ofstream os(path, std::ios::binary);
        if(!os) return false;
        size_t idl = ds_id.length();
        os.write((char*)&idl, sizeof(size_t));
        os.write(ds_id.c_str(), idl);
        os.write((char*)&step, sizeof(int)); 
        embed.save(os); recurrent.save(os); final_norm.save(os); lm_head.save(os);
        std::cout << "Checkpoint saved at step " << step << std::endl;
        return true;
    }
    
    int load_checkpoint(const std::string& path, const std::string& ds_id) {
        std::ifstream is(path, std::ios::binary);
        if(!is) return -1;
        size_t idl; is.read((char*)&idl, sizeof(size_t));
        std::string sid(idl, ' '); is.read(&sid[0], idl);
        if(sid != ds_id) { std::cerr << "Checkpoint mismatch!" << std::endl; return -1; }
        int saved_step;
        is.read((char*)&saved_step, sizeof(int)); 
        embed.load(is); recurrent.load(is); final_norm.load(is); lm_head.load(is);
        std::cout << "Successfully loaded checkpoint from step " << saved_step << std::endl;
        return saved_step;
    }
};
int sample_from_probs(const RowVector& probs, float temperature = 1.0f) {
    // 先应用温度
    RowVector logits_scaled = (probs.array() / temperature).max(-100.0f).min(100.0f);
    // 计算 softmax
    float max_val = logits_scaled.maxCoeff();
    RowVector exp_vals = (logits_scaled.array() - max_val).exp();
    float Z = exp_vals.sum();
    RowVector final_probs = exp_vals / Z;

    // 轮盘赌采样
    float r = static_cast<float>(rand()) / RAND_MAX;
    float cumulative = 0.0f;
    for (int i = 0; i < final_probs.size(); ++i) {
        cumulative += final_probs(i);
        if (r <= cumulative) return i;
    }
    return final_probs.size() - 1; // 保险
}

// 生成文本的核心函数
std::string generate_text(OpenMythos& model, bpe::BPETrainer& tokenizer,
                          const std::string& prompt, int max_new_tokens = 50,
                          float temperature = 0.8f) {
    // 编码 prompt
    std::vector<bpe::TokenId> prompt_ids = tokenizer.encode(prompt);
    std::vector<int> current_ids(prompt_ids.begin(), prompt_ids.end());

    // 生成循环
    for (int i = 0; i < max_new_tokens; ++i) {
        // 构造输入矩阵
        int L = static_cast<int>(current_ids.size());
        Eigen::MatrixXi input_ids(L, 1);
        for (int j = 0; j < L; ++j) input_ids(j, 0) = current_ids[j];

        // 前向传播
        Matrix logits = model.forward(input_ids);

        // 获取最后一个位置的 logits
        RowVector last_logits = logits.row(L - 1);

        // 采样下一个 token
        int next_token;
        if (temperature > 0.0f) {
            next_token = sample_from_probs(last_logits, temperature);
        } else {
            // 贪心解码
            last_logits.maxCoeff(&next_token);
        }

        // 如果生成了 EOS token，停止
        if (next_token == bpe::EOS_TOKEN_ID) break;

        current_ids.push_back(next_token);
    }

    // 解码成文本
    std::vector<bpe::TokenId> result_ids(current_ids.begin(), current_ids.end());
    return tokenizer.decode(result_ids);
}
// ============================================================================
// 4. 训练
// ============================================================================

uint64_t fnv1a64(const std::string& d) {
    uint64_t h = 1469598103934665603ull;
    for(unsigned char c : d) { h ^= (uint64_t)c; h *= 1099511628211ull; }
    return h;
}
std::string to_hex(uint64_t v) { std::ostringstream s; s << std::hex << std::setw(16) << std::setfill('0') << v; return s.str(); }
float get_lr(int s, int ts, float p, int w, float m) {
    if(s < w) return p * (s+1)/w;
    float r = (float)(s-w)/(ts-w); if(r>1) r=1;
    return p * (m + (1-m)*0.5f*(1+std::cos(M_PI*r)));
}

int main(int argc, char* argv[]) {
    bpe::BPETrainer tokenizer; std::ifstream f("train.txt");
    if(!f) return 1; std::string text((std::istreambuf_iterator<char>(f)), {});
    std::string ds_id = "train_fnv1a64_" + to_hex(fnv1a64(text));
    std::cout << "Dataset ID: " << ds_id << std::endl;

    if(!tokenizer.load("tokenizer.bpe", ds_id)) {
        tokenizer.train_from_file("train.txt"); tokenizer.save("tokenizer.bpe", ds_id);
    }
    std::vector<bpe::TokenId> tokens = tokenizer.has_cached_tokens() ? tokenizer.cached_tokens() : tokenizer.encode(text);
    if(!tokenizer.has_cached_tokens()) { tokenizer.set_cached_tokens(tokens); tokenizer.save("tokenizer.bpe", ds_id); }

    int vocab = tokenizer.vocab_size(), dim = 256, max_loop = 6, seq_len = 64, accum = 32;
    OpenMythos model(vocab, dim, max_loop);
    int start_step = 0;
    int loaded_step = model.load_checkpoint("openmythos_model.ckpt", ds_id);
    if (loaded_step >= 0) start_step = loaded_step;
    if (argc >= 3 && std::string(argv[1]) == "gen") {
        std::string prompt = argv[2];
        std::string generated = generate_text(model, tokenizer, prompt, 50, 0.8f);
        std::cout << "Prompt: " << prompt << std::endl;
        std::cout << "Generated: " << generated << std::endl;
        return 0;
    }
    int total_steps = 20000; float peak_lr = 0.0003f;
    for(int s=loaded_step; s<total_steps; ++s) {
        float lr = get_lr(s, total_steps, peak_lr, 500, 0.3f), loss_sum = 0, bal_sum = 0;
        for(int a=0; a<accum; ++a) {
            int start = rng() % (tokens.size()-seq_len-1);
            Eigen::MatrixXi inp(seq_len, 1), tgt(seq_len, 1);
            for(int i=0; i<seq_len; ++i) { inp(i,0) = tokens[start+i]; tgt(i,0) = tokens[start+i+1]; }
            Matrix logits = model.forward(inp);
            float l = 0; Matrix dl = Matrix::Zero(seq_len, vocab);
            for(int t=0; t<seq_len; ++t) {
                RowVector row_dl(vocab); l += cross_entropy_backward(logits.row(t), tgt(t,0), row_dl);
                dl.row(t) = row_dl;
            }
            l /= seq_len; dl /= (float)(seq_len * accum);
            loss_sum += l/accum; bal_sum += model.recurrent.moe.balance_loss/accum;
            model.backward(dl, inp);
        }
        model.step(lr, 1.0f);
        model.embed.W = model.lm_head.W.transpose();
        model.embed.mW = model.lm_head.mW.transpose();
        model.embed.vW = model.lm_head.vW.transpose();

        if(s % 10 == 0) {
            std::cout << "Step " << s << " | Loss: " << loss_sum + bal_sum << " | Bal: " << bal_sum << " | LR: " << lr << std::endl;
            if(s>0 && s%50==0) model.save_checkpoint("openmythos_model.ckpt", ds_id,s);
            if(std::isnan(loss_sum)) break;
        }
    }
    model.save_checkpoint("openmythos_model.ckpt", ds_id,0);
    return 0;
}