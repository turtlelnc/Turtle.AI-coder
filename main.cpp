// main.cpp – 融合 Claude 对齐 + M3 AMX 极限吞吐优化版 (Batched MoE & JIT Graph)
#include <mlx/mlx.h>
#include <mlx/backend/metal/metal.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>
#include <fstream>
#include <functional>
#include <numeric>
#include "BPE.h"
#include <thread>
#include <mutex>
#include <filesystem>
#include <algorithm>
#include <atomic>

using namespace mlx::core;
using namespace bpe;

// 优化器护栏参数
constexpr float SNR_THRESHOLD   = 1.0f;
constexpr float THRUST_STRENGTH = 0.1f;
constexpr float THRUST_MAX      = 1.0f;

std::mt19937 rng(42);

// ============================================================================
// 1. 全局参数管理器 (FP16/FP32 混合精度适配)
// ============================================================================
struct ModelParams {
    std::vector<array*> ptrs;
    std::vector<array> ms;
    std::vector<array> vs;

    void register_param(array& p) {
        ptrs.push_back(&p);
        ms.push_back(zeros(p.shape(), float32));
        vs.push_back(zeros(p.shape(), float32));
    }

    std::vector<array> get_values() const {
        std::vector<array> vals;
        for (auto p : ptrs) vals.push_back(*p);
        return vals;
    }

    void set_values(const std::vector<array>& vals) {
        for (size_t i = 0; i < ptrs.size(); ++i) {
            *(ptrs[i]) = vals[i];
        }
    }

    void save(std::ostream& os) const {
        size_t n = ptrs.size();
        os.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (size_t i = 0; i < n; ++i) {
            eval({*ptrs[i]});
            size_t bytes = ptrs[i]->nbytes();
            os.write(reinterpret_cast<const char*>(&bytes), sizeof(bytes));
            os.write(reinterpret_cast<const char*>(ptrs[i]->data<char>()), bytes);
        }
        for (size_t i = 0; i < n; ++i) {
            eval({ms[i], vs[i]});
            size_t bytes_m = ms[i].nbytes();
            os.write(reinterpret_cast<const char*>(&bytes_m), sizeof(bytes_m));
            os.write(reinterpret_cast<const char*>(ms[i].data<char>()), bytes_m);
            size_t bytes_v = vs[i].nbytes();
            os.write(reinterpret_cast<const char*>(&bytes_v), sizeof(bytes_v));
            os.write(reinterpret_cast<const char*>(vs[i].data<char>()), bytes_v);
        }
    }

    bool load(std::istream& is) {
        size_t n;
        is.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (n != ptrs.size()) {
            std::cerr << "Checkpoint param count mismatch! " << n << " vs " << ptrs.size() << std::endl;
            return false;
        }
        for (size_t i = 0; i < n; ++i) {
            size_t bytes;
            is.read(reinterpret_cast<char*>(&bytes), sizeof(bytes));
            std::vector<char> tmp(bytes);
            is.read(tmp.data(), bytes);
            *ptrs[i] = array(tmp.data(), ptrs[i]->shape(), ptrs[i]->dtype());
        }
        for (size_t i = 0; i < n; ++i) {
            size_t bytes;
            is.read(reinterpret_cast<char*>(&bytes), sizeof(bytes));
            std::vector<char> tmp_m(bytes);
            is.read(tmp_m.data(), bytes);
            ms[i] = array(tmp_m.data(), ms[i].shape(), ms[i].dtype());
            
            is.read(reinterpret_cast<char*>(&bytes), sizeof(bytes));
            std::vector<char> tmp_v(bytes);
            is.read(tmp_v.data(), bytes);
            vs[i] = array(tmp_v.data(), vs[i].shape(), vs[i].dtype());
        }
        return true;
    }
};

// ============================================================================
// 2. 自定义 SNR-Gated 优化器 (算子融合友好版)
// ============================================================================
void apply_snr_gated_update(ModelParams& mp, const std::vector<array>& grads,
                            float lr, float b1_corr, float b2_corr, float max_grad_norm = 0.1f) {
    float beta1 = 0.9f;
    float beta2 = 0.999f;
    array eps = array(1e-8f, float32);
    std::vector<array> to_eval;

    for (size_t i = 0; i < mp.ptrs.size(); ++i) {
        array& param = *(mp.ptrs[i]);
        const array& grad = grads[i];
        array& m = mp.ms[i];
        array& v = mp.vs[i];

        array grad_f32 = astype(grad, float32);
        array gnorm = sqrt(sum(square(grad_f32)));
        array clipped_grad = where(greater(gnorm, array(max_grad_norm, float32)),
                                   multiply(grad_f32, divide(array(max_grad_norm, float32), add(gnorm, eps))),
                                   grad_f32);

        m = add(multiply(array(beta1, float32), m), multiply(array(1.0f - beta1, float32), clipped_grad));
        v = add(multiply(array(beta2, float32), v), multiply(array(1.0f - beta2, float32), square(clipped_grad)));

        array m_hat = divide(m, array(b1_corr, float32));
        array v_hat = divide(v, array(b2_corr, float32));

        array adam_step = multiply(array(lr, float32), divide(m_hat, add(sqrt(v_hat), eps)));

        array snr = divide(abs(m_hat), add(sqrt(v_hat), eps));
        array dampening = multiply(multiply(negative(m_hat), array(lr, float32)), array(THRUST_STRENGTH * 0.5f, float32));
        dampening = maximum(minimum(dampening, array(THRUST_MAX * lr, float32)), array(-THRUST_MAX * lr, float32));

        array snr_mask = greater(snr, array(SNR_THRESHOLD, float32));
        array total_step = add(adam_step, where(snr_mask, dampening, zeros_like(dampening)));

        param = subtract(param, astype(total_step, param.dtype()));

        to_eval.push_back(param);
        to_eval.push_back(m);
        to_eval.push_back(v);
    }
    // 统一抛给 MLX 后端，交由 Metal Shader 编译器自动融合数学运算算子
    eval(to_eval);
}

// ============================================================================
// 3. 基础神经网络层 (原生 float16 支持)
// ============================================================================
struct Linear {
    array W, b;
    Linear(int in, int out, ModelParams& mp)
        : W(random::normal({in, out}, 0.0f, 0.02f, float16)),
          b(zeros({out}, float16))
    {
        mp.register_param(W);
        mp.register_param(b);
    }
    array operator()(const array& x) const { return add(matmul(x, W), b); }
};

struct RMSNorm {
    float eps;
    array weight;
    RMSNorm(int dim, ModelParams& mp, float e = 1e-4f)
        : eps(e), weight(ones({dim}, float16))
    {
        mp.register_param(weight);
    }
    array operator()(const array& x) const {
        array x_f32 = astype(x, float32);
        array var = mean(square(x_f32), -1, true);
        array inv_rms = rsqrt(add(var, array(eps, float32)));
        array out_f32 = multiply(x_f32, inv_rms);
        return multiply(astype(out_f32, float16), weight);
    }
};

struct LoopEmbedding {
    array W;
    LoopEmbedding(int max_loops, int dim, ModelParams& mp)
        : W(random::normal({max_loops, dim}, 0.0f, 0.02f, float16))
    {
        mp.register_param(W);
    }
    array operator()(int t, int batch_size) const {
        array wt = slice(W, {t, 0}, {t+1, W.shape(1)});
        return broadcast_to(wt, {batch_size, W.shape(1)});
    }
};

// ============================================================================
// 4. 注意力机制与 3D 张量并行 Batched MoE
// ============================================================================
struct SlidingWindowAttention {
    Linear Wq, Wk, Wv, Wo;
    int n_heads, head_dim, window_size;

    SlidingWindowAttention(int dim, int h, int ws, ModelParams& mp)
        : Wq(dim, dim, mp), Wk(dim, dim, mp), Wv(dim, dim, mp), Wo(dim, dim, mp),
          n_heads(h), head_dim(dim/h), window_size(ws) {}

    array operator()(const array& x) const {
        int L = x.shape(0);
        array q = Wq(x); array k = Wk(x); array v = Wv(x);
        
        q = transpose(reshape(q, {L, n_heads, head_dim}), {1, 0, 2});
        k = transpose(reshape(k, {L, n_heads, head_dim}), {1, 0, 2});
        v = transpose(reshape(v, {L, n_heads, head_dim}), {1, 0, 2});

        q = fast::rope(q, head_dim, false, 10000.0f, 1.0f, 0);
        k = fast::rope(k, head_dim, false, 10000.0f, 1.0f, 0);

        float scale = 1.0f / std::sqrt((float)head_dim);
        array scores = multiply(matmul(q, transpose(k, {0, 2, 1})), array(scale, q.dtype()));

        array idx = arange(L);
        array diff = subtract(expand_dims(idx, 1), expand_dims(idx, 0));
        array mask = logical_and(greater_equal(diff, array(0)), less(diff, array(window_size)));
        
        scores = where(mask, scores, array(-1e4f, scores.dtype()));
        array probs = astype(softmax(astype(scores, float32), -1), float16);
        array out = matmul(probs, v);
        
        out = reshape(transpose(out, {1, 0, 2}), {L, n_heads * head_dim});
        return Wo(out);
    }
};

struct SparseGlobalAttention {
    Linear Wq, Wk, Wv, Wo;
    int n_heads, head_dim, topk_n;

    SparseGlobalAttention(int dim, int h, int tk, ModelParams& mp)
        : Wq(dim, dim, mp), Wk(dim, dim, mp), Wv(dim, dim, mp), Wo(dim, dim, mp),
          n_heads(h), head_dim(dim/h), topk_n(tk) {}

    array operator()(const array& x) const {
        int L = x.shape(0);
        array q = Wq(x); array k = Wk(x); array v = Wv(x);
        
        q = transpose(reshape(q, {L, n_heads, head_dim}), {1, 0, 2});
        k = transpose(reshape(k, {L, n_heads, head_dim}), {1, 0, 2});
        v = transpose(reshape(v, {L, n_heads, head_dim}), {1, 0, 2});

        q = fast::rope(q, head_dim, false, 10000.0f, 1.0f, 0);
        k = fast::rope(k, head_dim, false, 10000.0f, 1.0f, 0);

        float scale = 1.0f / std::sqrt((float)head_dim);
        array scores = multiply(matmul(q, transpose(k, {0, 2, 1})), array(scale, q.dtype()));

        array idx = arange(L);
        array mask_c = greater_equal(expand_dims(idx, 1), expand_dims(idx, 0));
        scores = where(mask_c, scores, array(-1e4f, scores.dtype()));

        int k_actual = std::min(topk_n, L);
        array topk_vals = topk(scores, k_actual, -1);
        array threshold = slice(topk_vals, {0, 0, k_actual - 1}, {n_heads, L, k_actual});
        scores = where(greater_equal(scores, threshold), scores, array(-1e4f, scores.dtype()));

        array probs = astype(softmax(astype(scores, float32), -1), float16);
        array out = matmul(probs, v);
        out = reshape(transpose(out, {1, 0, 2}), {L, n_heads * head_dim});
        return Wo(out);
    }
};

// 【硬核重构：彻底淘汰低效的 C++ for 循环】
// 将所有专家的权重拼接为三维张量 (E, in, out)，利用 AMX 矩阵协处理器一次性完成所有专家的并行推断
struct BatchedMoE {
    int dim, n_experts, top_k;
    Linear router;
    array W_gate, W_up, W_down;

    BatchedMoE(int d, int h, int ne, int tk, ModelParams& mp)
        : dim(d), n_experts(ne), top_k(tk), router(d, ne, mp),
          // 权重形状：(E, 输入维度, 输出维度)
          W_gate(random::normal({ne, d, h}, 0.0f, 0.02f, float16)),
          W_up(random::normal({ne, d, h}, 0.0f, 0.02f, float16)),
          W_down(random::normal({ne, h, d}, 0.0f, 0.02f, float16)) {
        mp.register_param(W_gate);
        mp.register_param(W_up);
        mp.register_param(W_down);
    }

    array operator()(const array& x) const {
        int L = x.shape(0);
        array l = router(x); // (L, E)
        array probs = astype(softmax(astype(l, float32), -1), float16);

        array topk_vals = topk(probs, top_k, -1);
        array threshold = slice(topk_vals, {0, top_k - 1}, {L, top_k});
        array route_probs = where(greater_equal(probs, threshold), probs, zeros_like(probs));
        route_probs = divide(route_probs, expand_dims(sum(route_probs, -1), -1)); // (L, E)

        // 统一内存架构的神迹：广播扩张几乎是零拷贝 (Zero-copy) 级别的内存占用
        // 将 x (L, d) 无感扩张为 (E, L, d)，以便于跟 W_gate (E, d, h) 进行 3D 批处理矩阵乘法
        array x_expanded = broadcast_to(expand_dims(x, 0), {n_experts, L, dim});

        // 仅发射 3 次大规模 GPU Kernel 指令，彻底吃满算力
        array g = sigmoid(matmul(x_expanded, W_gate)); // -> (E, L, h)
        array up = matmul(x_expanded, W_up);           // -> (E, L, h)
        array hidden = multiply(g, up);                // -> (E, L, h)
        array out_experts = matmul(hidden, W_down);    // -> (E, L, d)

        // 把 route_probs 从 (L, E) 转置并扩张为 (E, L, 1)，作为掩码直接压扁相加
        array w = expand_dims(transpose(route_probs, {1, 0}), -1);
        array weighted_out = multiply(out_experts, w); // (E, L, d)
        
        return sum(weighted_out, 0); // 最终坍缩回 (L, d)
    }
};

struct ACT {
    int max_loops;
    Linear linear;
    ACT(int d, int l, ModelParams& mp) : max_loops(l), linear(d, 1, mp) {}

    array operator()(const std::vector<array>& hs) const {
        int L = hs[0].shape(0);
        array out = zeros_like(hs[0]);
        array rem = ones({L, 1}, float16);

        for (size_t t = 0; t < hs.size(); ++t) {
            array pt = sigmoid(linear(hs[t]));
            array wt = (t == hs.size() - 1) ? rem : multiply(pt, rem);
            rem = subtract(rem, wt);
            out = add(out, multiply(hs[t], wt));
        }
        return out;
    }
};

// ============================================================================
// 5. 模型块装配
// ============================================================================
struct TransformerBlock {
    SlidingWindowAttention attn_sw;
    BatchedMoE moe;
    RMSNorm n_attn, n_moe;

    TransformerBlock(int dim, int ws, int ne, int tk, ModelParams& mp)
        : attn_sw(dim, 8, ws, mp), moe(dim, dim * 4, ne, tk, mp),
          n_attn(dim, mp), n_moe(dim, mp) {}

    array operator()(const array& x) const {
        array mid = add(x, attn_sw(n_attn(x)));
        return add(mid, moe(n_moe(mid)));
    }
};

struct RecurrentBlock {
    std::vector<std::shared_ptr<TransformerBlock>> wide_blocks;
    LoopEmbedding loop_embed;
    SlidingWindowAttention attn_sw;
    SparseGlobalAttention attn_global;
    BatchedMoE moe;
    RMSNorm n_attn, n_moe;
    ACT act;
    float memory_alpha = 0.9f;

    RecurrentBlock(int dim, int max_loops, int ws, int gk, int ne, int tk, int nw, ModelParams& mp)
        : loop_embed(max_loops, dim, mp), attn_sw(dim, 8, ws, mp), attn_global(dim, 8, gk, mp),
          moe(dim, dim * 4, ne, tk, mp), n_attn(dim, mp), n_moe(dim, mp), act(dim, max_loops, mp) {
        for (int i = 0; i < nw; ++i) {
            wide_blocks.push_back(std::make_shared<TransformerBlock>(dim, ws, ne, tk, mp));
        }
    }

    array operator()(array x) const {
        for (auto& blk : wide_blocks) x = blk->operator()(x);
        
        std::vector<array> hs;
        array memory = zeros_like(x);

        for (int t = 0; t < act.max_loops; ++t) {
            array n_a = add(n_attn(x), multiply(memory, array(0.1f, float16)));
            x = add(x, loop_embed(t, x.shape(0)));
            array a_out = (t % 2 == 0) ? attn_sw(n_a) : attn_global(n_a);
            
            array mid = add(x, a_out);
            x = add(mid, moe(n_moe(mid)));
            
            memory = add(multiply(array(memory_alpha, float16), memory), multiply(array(1.0f - memory_alpha, float16), x));
            hs.push_back(x);
        }
        return act(hs);
    }
};

struct OpenMythos {
    ModelParams mp;
    Linear lm_head;
    RecurrentBlock recurrent;
    RMSNorm final_norm;

    OpenMythos(int vocab, int dim, int ml, int ws, int gk, int ne, int tk, int nw)
        : lm_head(dim, vocab, mp),
          recurrent(dim, ml, ws, gk, ne, tk, nw, mp),
          final_norm(dim, mp) {}

    array operator()(const array& ids) const {
        array embed = transpose(lm_head.W, {1, 0});
        array x = take(embed, ids, 0);
        array h = recurrent(x);
        return lm_head(final_norm(h));
    }
};

// ============================================================================
// 6. 辅助函数（强制 FP32 计算交叉熵）
// ============================================================================
array cross_entropy_loss(const array& logits, const array& targets, int ignore_index = -100, float entropy_beta = 0.01f) {
    array logits_f32 = astype(logits, float32);

    array max_logits = max(logits_f32, -1, true);
    array shifted = subtract(logits_f32, max_logits);
    array exp_shifted = exp(shifted);
    array sum_exp = sum(exp_shifted, -1, true);
    array log_probs = subtract(shifted, log(sum_exp));

    array gathered = take_along_axis(log_probs, expand_dims(targets, -1), -1);
    gathered = reshape(gathered, {-1});
    array mask = not_equal(targets, array(ignore_index, targets.dtype()));
    array mask_f = astype(mask, float32);
    array base_loss = multiply(negative(gathered), mask_f);

    array probs = divide(exp_shifted, sum_exp);
    array entropy = multiply(negative(probs), log_probs);
    array token_entropy = sum(entropy, -1);
    array confidence_penalty = multiply(base_loss, exp(negative(token_entropy)));

    array total_token_loss = add(base_loss, multiply(array(entropy_beta, float32), confidence_penalty));
    
    float denom = sum(mask_f).item<float>() + 1e-8f;
    return sum(total_token_loss) / array(denom, float32);
}

int sample_from_probs(const array& logits, float temp = 1.0f) {
    array scaled = divide(astype(logits, float32), array(temp, float32));
    array probs = softmax(scaled, -1);
    eval({probs});
    
    const float* p_data = probs.data<float>();
    std::uniform_real_distribution<float> unif(0.0f, 1.0f);
    float r = unif(rng), cum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cum += p_data[i];
        if (r <= cum) return i;
    }
    return probs.size() - 1;
}

uint64_t fnv1a64(const std::string& d) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : d) { h ^= (uint64_t)c; h *= 1099511628211ull; }
    return h;
}

float get_lr(int step, int total_steps, float peak_lr, int warmup_steps) {
    if (step < warmup_steps) return peak_lr * (step + 1) / warmup_steps;
    float progress = float(step - warmup_steps) / (total_steps - warmup_steps);
    if (progress > 1.0f) progress = 1.0f;
    return peak_lr * 0.5f * (1.0f + std::cos(M_PI * progress));
}

// ============================================================================
// 7. 训练主逻辑
// ============================================================================
int main(int argc, char* argv[]) {
    set_default_device(Device::gpu);
    std::cout << "OpenMythos v4.0 - AMX Accelerated Engine on: " << default_device().type << std::endl;

    // ---------- 超参数 (为 8GB 设计的基础配置) ----------
    int dim = 64;
    int max_loop = 3;
    int window_size = 64;
    int global_topk = 16;
    int seq_len = 64;
    int accum = 16;
    int moe_experts = 8;
    int moe_topk = 1;
    int num_wide_blocks = 2;
    int total_steps = 5000;
    float peak_lr = 5e-5f;

    // ---------- CLI 参数解析辅助 ----------
    auto get_arg = [&](const std::string& key, const std::string& def) -> std::string {
        for (int i = 1; i + 1 < argc; ++i)
            if (std::string(argv[i]) == key) return argv[i + 1];
        return def;
    };
    auto get_int_arg = [&](const std::string& key, int def) -> int {
        std::string v = get_arg(key, "");
        return v.empty() ? def : std::stoi(v);
    };

    // ---------- 推理生成模式 ----------
    if (argc >= 3 && std::string(argv[1]) == "gen") {
        std::string model_path = get_arg("--model", "openmythos_mlx.ckpt");
        std::string tok_path   = get_arg("--tokenizer", "tokenizer.bpe");
        int max_tokens         = get_int_arg("--max-tokens", 200);
        float temp             = std::stof(get_arg("--temp", "0.7"));

        BPETrainer tokenizer;
        if (!tokenizer.load(tok_path)) {
            std::cerr << "ERROR: tokenizer not found at " << tok_path << std::endl;
            return 1;
        }
        OpenMythos model(tokenizer.vocab_size(), dim, max_loop, window_size, global_topk,
                         moe_experts, moe_topk, num_wide_blocks);
        {
            std::ifstream is(model_path, std::ios::binary);
            if (is) {
                int dummy_step;
                is.read(reinterpret_cast<char*>(&dummy_step), sizeof(int));
                model.mp.load(is);
            } else {
                std::cerr << "WARN: checkpoint not found at " << model_path << ", using random weights" << std::endl;
            }
        }

        std::string prompt = argv[2];
        std::vector<TokenId> ids = tokenizer.encode(prompt, true);

        std::cout << "READY\n" << std::flush;

        for (int i = 0; i < max_tokens; ++i) {
            array input = array(ids.data(), {static_cast<int>(ids.size())}, int32);
            array logits = model(input);
            array last_logit = slice(logits,
                                     {static_cast<int>(ids.size()) - 1, 0},
                                     {static_cast<int>(ids.size()), static_cast<int>(tokenizer.vocab_size())});
            int next = sample_from_probs(last_logit, temp);
            if (next == bpe::EOS_TOKEN_ID) break;
            ids.push_back(next);

            std::string piece = tokenizer.decode({static_cast<bpe::TokenId>(next)}, true);
            std::cout << "TOKEN:" << piece << "\n" << std::flush;
        }
        std::cout << "END\n" << std::flush;
        return 0;
    }

    // ---------- 数据准备 ----------
    const std::string data_dir = get_arg("--data-dir", "train_files");
    const std::string ckpt_path_arg = get_arg("--model-out", "openmythos_mlx.ckpt");
    const std::string tok_path_arg  = get_arg("--tokenizer-out", "tokenizer.bpe");
    if (get_int_arg("--steps", 0) > 0) total_steps = get_int_arg("--steps", total_steps);
    
    std::vector<std::string> file_paths;
    if (std::filesystem::exists(data_dir)) {
        for (const auto& entry : std::filesystem::directory_iterator(data_dir)) {
            if (entry.is_regular_file()) file_paths.push_back(entry.path().string());
        }
    }
    if (file_paths.empty()) { std::cerr << "No training files inside " << data_dir << "!" << std::endl; return 1; }
    std::cout << "找到 " << file_paths.size() << " 个训练文件" << std::endl;

    std::vector<std::string> all_texts;
    all_texts.reserve(file_paths.size());
    for (const auto& path : file_paths) {
        std::ifstream in(path);
        if (!in) continue;
        all_texts.push_back(std::string((std::istreambuf_iterator<char>(in)), {}));
    }

    // ---------- 分词器处理 ----------
    BPETrainer tokenizer;
    if (!tokenizer.load(tok_path_arg)) {
        std::cout << "未找到 " << tok_path_arg << "，开始训练新分词器..." << std::endl;
        BPEConfig config;
        config.vocab_size    = 16384;
        config.min_frequency = 2;
        tokenizer = BPETrainer(config);

        std::vector<std::string> specials = {
            "[UNK]", "[PAD]", "[BOS]", "[EOS]",
            "\n", "    ",
            "<THINK>", "</THINK>",
            "<VERIFY_PASSED>", "<CORRECT>",
            "<CLASS_DECL:", "<STRUCT_DECL:", "<FUNCTION_DECL:",
            "<FIELD_DECL:", "<VAR_DECL:", "<PARM_DECL:", ">",
            "type=\"", "template"
        };
        for (const auto& t : specials) tokenizer.add_special_token(t);

        tokenizer.train_from_texts(all_texts);
        tokenizer.save(tok_path_arg);
        std::cout << "分词器训练完成并保存。" << std::endl;
    }

    // ---------- 多线程全局预分词 ----------
    std::cout << "正在多线程预分词..." << std::endl;
    std::vector<std::vector<int>> tokenized_datasets(all_texts.size());
    std::mutex print_mutex;
    
    unsigned int num_threads = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> workers;
    std::atomic<size_t> processed{0};
    const size_t total = all_texts.size();
    
    for (unsigned int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            for (size_t i = t; i < total; i += num_threads) {
                const auto& code = all_texts[i];
                if (code.empty()) continue;
    
                auto file_tokens = tokenizer.encode(code);
                std::vector<int> full_tokens;
                full_tokens.reserve(2 + file_tokens.size());
                full_tokens.push_back(FILE_START_TOKEN_ID);
                for (auto id : file_tokens) full_tokens.push_back(static_cast<int>(id));
                full_tokens.push_back(FILE_END_TOKEN_ID);
                tokenized_datasets[i] = std::move(full_tokens);
    
                size_t done = processed.fetch_add(1) + 1;
                if (done % 500 == 0 || done == total) {
                    std::lock_guard<std::mutex> lock(print_mutex);
                    std::cout << "\r  预分词进度: " << done << "/" << total
                              << " (" << (done * 100 / total) << "%)" << std::flush;
                }
            }
        });
    }
    for (auto& w : workers) w.join();
    
    tokenized_datasets.erase(
        std::remove_if(tokenized_datasets.begin(), tokenized_datasets.end(),
                       [](const auto& v) { return v.empty(); }),
        tokenized_datasets.end());
    std::cout << "\n✅ 预分词完成，有效序列数: " << tokenized_datasets.size() << std::endl;
    
    all_texts.clear();
    all_texts.shrink_to_fit();

    // ---------- 模型初始化 ----------
    OpenMythos model(tokenizer.vocab_size(), dim, max_loop, window_size, global_topk,
                     moe_experts, moe_topk, num_wide_blocks);

    // ---------- 断点续训自动检测 ----------
    int start_step = 0;
    const std::string ckpt_path = ckpt_path_arg;
    {
        std::ifstream is(ckpt_path, std::ios::binary);
        if (is) {
            is.read(reinterpret_cast<char*>(&start_step), sizeof(int));
            if (model.mp.load(is)) {
                std::cout << "从第 " << start_step << " 步成功恢复训练" << std::endl;
            } else {
                std::cerr << "Checkpoint 损坏，重归初始状态训练" << std::endl;
                start_step = 0;
            }
        }
    }

    // ---------- 极限吞吐训练循环 ----------
    array inp = zeros({seq_len}, int32);
    array tgt = zeros({seq_len}, int32);

    auto loss_fn = [&](const std::vector<array>& params) -> array {
        model.mp.set_values(params);
        array logits = model(inp);
        return cross_entropy_loss(logits, tgt, PAD_TOKEN_ID);
    };

    std::vector<int> argnums(model.mp.ptrs.size());
    std::iota(argnums.begin(), argnums.end(), 0);
    auto grad_fn = value_and_grad(loss_fn, argnums);

    for (int s = start_step; s < total_steps; ++s) {
        float lr = get_lr(s, total_steps, peak_lr, 1000);
        float b1_corr = 1.0f - std::pow(0.9f, s + 1);
        float b2_corr = 1.0f - std::pow(0.999f, s + 1);

        std::vector<array> accum_grads;
        accum_grads.reserve(model.mp.ptrs.size());
        for (size_t i = 0; i < model.mp.ptrs.size(); ++i) {
            accum_grads.push_back(zeros_like(*(model.mp.ptrs[i])));
        }
        array step_loss_arr = array(0.0f, float32);

        for (int a = 0; a < accum; ++a) {
            const auto& full_tokens = tokenized_datasets[rng() % tokenized_datasets.size()];
            int total_len = full_tokens.size();
            int win_start = 0;
            if (total_len > seq_len + 1) {
                int max_start = total_len - (seq_len + 1);
                win_start = rng() % (max_start + 1);
            }
            int win_end = std::min(win_start + seq_len + 1, total_len);
            int cur_len = win_end - win_start;

            std::vector<int> inp_v(seq_len, PAD_TOKEN_ID);
            std::vector<int> tgt_v(seq_len, PAD_TOKEN_ID);
            std::copy(full_tokens.begin() + win_start, full_tokens.begin() + win_start + cur_len - 1, inp_v.begin());
            std::copy(full_tokens.begin() + win_start + 1, full_tokens.begin() + win_start + cur_len, tgt_v.begin());

            inp = array(inp_v.data(), {seq_len}, int32);
            tgt = array(tgt_v.data(), {seq_len}, int32);

            auto [loss, grads] = grad_fn(model.mp.get_values());
            step_loss_arr = add(step_loss_arr, loss);
            for (size_t i = 0; i < grads.size(); ++i) {
                accum_grads[i] = add(accum_grads[i], grads[i]);
            }

            // 【延迟计算图编译】：摸索出 M3 内存墙与运算深度的极限平衡点
            // 将评测频率从 1 放宽到 4，给 MLX Compiler 留出算子融合的空间，以换取极高的吞吐效率
            if ((a + 1) % 4 == 0 || a == accum - 1) {
                eval(accum_grads);
                eval({step_loss_arr});
            }
        }

        for (auto& g : accum_grads) g = divide(g, array(static_cast<float>(accum), float16));

        array global_sq_norm = array(0.0f, float32);
        for (const auto& g : accum_grads) {
            global_sq_norm = add(global_sq_norm, sum(square(astype(g, float32))));
        }
        array gnorm = sqrt(global_sq_norm);
        array scale = where(greater(gnorm, array(0.5f, float32)),
                            divide(array(0.5f, float32), gnorm),
                            array(1.0f, float32));
        for (auto& g : accum_grads) g = multiply(g, astype(scale, float16));

        apply_snr_gated_update(model.mp, accum_grads, lr, b1_corr, b2_corr, 1e6f);

        mlx::core::metal::clear_cache();

        if (s % 10 == 0) {
            float avg_loss = step_loss_arr.item<float>() / accum;
            std::cout << "Step " << s << " | Loss: " << std::fixed << std::setprecision(4) << avg_loss
                      << " | LR: " << std::scientific << lr << std::endl;
        }
        if (s > 0 && s % 100 == 0) {
            std::ofstream os(ckpt_path, std::ios::binary);
            if (os) {
                os.write(reinterpret_cast<const char*>(&s), sizeof(int));
                model.mp.save(os);
                std::cout << "Checkpoint saved successfully at step " << s << std::endl;
            }
        }
    }

    return 0;
}
