#include "BPE.h"
#include <iostream>
#include <algorithm>
#include <cstdint>
#include <queue>
#include <unordered_set>
#include <tuple>
#include <cstring>

namespace bpe {
namespace {
constexpr const char* kBpeMagic = "OM_BPE_V1";
}

struct TokenPairHash {
    size_t operator()(const std::pair<TokenId, TokenId>& p) const {
        return std::hash<TokenId>()(p.first) ^ (std::hash<TokenId>()(p.second) << 1);
    }
};

// ---------- 构造函数与基础词汇表 ----------
BPETrainer::BPETrainer() { build_initial_vocab(); }
BPETrainer::BPETrainer(const BPEConfig& config) : config_(config) { build_initial_vocab(); }

void BPETrainer::build_initial_vocab() {
    vocab_.clear(); id_to_vocab_.clear();
    vocab_[config_.unk_token] = UNK_TOKEN_ID; id_to_vocab_[UNK_TOKEN_ID] = config_.unk_token;
    vocab_[config_.bos_token] = BOS_TOKEN_ID; id_to_vocab_[BOS_TOKEN_ID] = config_.bos_token;
    vocab_[config_.eos_token] = EOS_TOKEN_ID; id_to_vocab_[EOS_TOKEN_ID] = config_.eos_token;
    vocab_[config_.pad_token] = PAD_TOKEN_ID; id_to_vocab_[PAD_TOKEN_ID] = config_.pad_token;
    for (int i = 0; i < 256; ++i) {
        std::string s(1, static_cast<unsigned char>(i));
        TokenId new_id = static_cast<TokenId>(vocab_.size());
        vocab_[s] = new_id;
        id_to_vocab_[new_id] = s;
    }
}

// ---------- 核心训练（优化版 + 构建快速编码表） ----------
bool BPETrainer::train_from_texts(const std::vector<std::string>& texts) {
    build_initial_vocab();
    merge_rules_.clear();
    dataset_id_.clear();
    cached_tokens_.clear();

    // 1. 预分词并统计单词频率
    std::unordered_map<std::string, size_t> word_freqs;
    for (const auto& text : texts) {
        auto it = std::sregex_iterator(text.begin(), text.end(), config_.pattern);
        for (; it != std::sregex_iterator(); ++it) {
            word_freqs[it->str()]++;
        }
    }

    // 2. 将每个单词转换为初始 token ID 序列，并建立索引
    struct WordEntry {
        std::vector<TokenId> seq;
        size_t freq;
    };
    std::vector<WordEntry> words;
    std::vector<std::string> original_words; // 记录原始单词，用于构建快速编码表
    words.reserve(word_freqs.size());
    original_words.reserve(word_freqs.size());

    std::unordered_map<std::pair<TokenId, TokenId>, int, TokenPairHash> pair_counts;
    std::unordered_map<std::pair<TokenId, TokenId>, std::unordered_set<int>, TokenPairHash> pair_words;

    for (const auto& [word, freq] : word_freqs) {
        WordEntry entry;
        for (unsigned char c : word) {
            entry.seq.push_back(4 + static_cast<TokenId>(c)); // 字节 ID 从 4 开始
        }
        entry.freq = freq;
        int word_idx = static_cast<int>(words.size());
        words.push_back(entry);
        original_words.push_back(word);

        const auto& seq = words.back().seq;
        for (size_t i = 0; i + 1 < seq.size(); ++i) {
            auto pair = std::make_pair(seq[i], seq[i + 1]);
            pair_counts[pair] += static_cast<int>(freq);
            pair_words[pair].insert(word_idx);
        }
    }

    // 3. 优先队列
    using HeapItem = std::pair<int, std::pair<TokenId, TokenId>>;
    auto cmp = [](const HeapItem& a, const HeapItem& b) { return a.first < b.first; };
    std::priority_queue<HeapItem, std::vector<HeapItem>, decltype(cmp)> heap(cmp);
    for (const auto& [pair, count] : pair_counts) {
        if (count >= static_cast<int>(config_.min_frequency))
            heap.push({count, pair});
    }

    // 4. 合并循环
    while (vocab_.size() < config_.vocab_size && !heap.empty()) {
        auto top = heap.top();
        heap.pop();
        int count = top.first;
        auto pair = top.second;

        auto it_count = pair_counts.find(pair);
        if (it_count == pair_counts.end() || it_count->second != count) continue;
        if (count < static_cast<int>(config_.min_frequency)) continue;

        TokenId a = pair.first;
        TokenId b = pair.second;
        TokenId new_id = static_cast<TokenId>(vocab_.size());

        std::string first_str = id_to_vocab_.at(a);
        std::string second_str = id_to_vocab_.at(b);
        std::string merged_str = first_str + second_str;
        vocab_[merged_str] = new_id;
        id_to_vocab_[new_id] = merged_str;
        merge_rules_.push_back({first_str, second_str, merged_str, new_id});

        std::unordered_set<int> affected_words = pair_words[pair];
        pair_counts.erase(pair);
        pair_words.erase(pair);

        for (int word_idx : affected_words) {
            auto& seq = words[word_idx].seq;
            size_t freq = words[word_idx].freq;

            for (size_t i = 0; i + 1 < seq.size(); ) {
                if (seq[i] == a && seq[i + 1] == b) {
                    if (i > 0) {
                        auto old_pair = std::make_pair(seq[i - 1], a);
                        if (--pair_counts[old_pair] == 0) pair_counts.erase(old_pair);
                        pair_words[old_pair].erase(word_idx);
                    }
                    if (i + 2 < seq.size()) {
                        auto old_pair = std::make_pair(b, seq[i + 2]);
                        if (--pair_counts[old_pair] == 0) pair_counts.erase(old_pair);
                        pair_words[old_pair].erase(word_idx);
                    }
                    seq.erase(seq.begin() + i, seq.begin() + i + 2);
                    seq.insert(seq.begin() + i, new_id);

                    if (i > 0) {
                        auto new_pair = std::make_pair(seq[i - 1], new_id);
                        pair_counts[new_pair] += static_cast<int>(freq);
                        pair_words[new_pair].insert(word_idx);
                    }
                    if (i + 1 < seq.size()) {
                        auto new_pair = std::make_pair(new_id, seq[i + 1]);
                        pair_counts[new_pair] += static_cast<int>(freq);
                        pair_words[new_pair].insert(word_idx);
                    }
                } else {
                    ++i;
                }
            }
        }

        for (const auto& [p, cnt] : pair_counts) {
            if (cnt >= static_cast<int>(config_.min_frequency))
                heap.push({cnt, p});
        }
    }

    // 5. 构建快速编码映射表（单词 → token 序列）
    fast_vocab_.clear();
    for (size_t i = 0; i < words.size(); ++i) {
        fast_vocab_[original_words[i]] = words[i].seq;
    }

    return true;
}

// ---------- 快速编码（直接查表，仅适用于训练文本中的单词） ----------
std::vector<TokenId> BPETrainer::encode_fast(const std::string& text, bool add_special) const {
    std::vector<TokenId> ids;
    if (add_special) ids.push_back(BOS_TOKEN_ID);
    auto it = std::sregex_iterator(text.begin(), text.end(), config_.pattern);
    for (; it != std::sregex_iterator(); ++it) {
        const std::string& word = it->str();
        auto found = fast_vocab_.find(word);
        if (found != fast_vocab_.end()) {
            ids.insert(ids.end(), found->second.begin(), found->second.end());
        } else {
            // 未在训练集中出现的单词，回退到慢速编码（合并规则）
            for (const auto& t : apply_merges(word)) {
                ids.push_back(vocab_.count(t) ? vocab_.at(t) : UNK_TOKEN_ID);
            }
        }
    }
    if (add_special) ids.push_back(EOS_TOKEN_ID);
    return ids;
}

// ---------- 原有慢速编码（保留给新文本） ----------
std::vector<TokenId> BPETrainer::encode(const std::string& text, bool add_special) const {
    // 如果已构建快速表，优先使用
    if (!fast_vocab_.empty()) return encode_fast(text, add_special);

    std::vector<TokenId> ids;
    if (add_special) ids.push_back(BOS_TOKEN_ID);
    auto it = std::sregex_iterator(text.begin(), text.end(), config_.pattern);
    for (; it != std::sregex_iterator(); ++it) {
        for (const auto& t : apply_merges(it->str())) {
            ids.push_back(vocab_.count(t) ? vocab_.at(t) : UNK_TOKEN_ID);
        }
    }
    if (add_special) ids.push_back(EOS_TOKEN_ID);
    return ids;
}

// ---------- 其余辅助函数（与之前一样） ----------
bool BPETrainer::train_from_file(const std::string& path) {
    std::ifstream f(path); if (!f) return false;
    std::vector<std::string> lines; std::string line;
    while (std::getline(f, line)) lines.push_back(line);
    return train_from_texts(lines);
}

std::vector<std::string> BPETrainer::apply_merges(const std::string& word) const {
    std::vector<std::string> tokens;
    for (unsigned char c : word) tokens.push_back(std::string(1, c));
    for (const auto& r : merge_rules_) {
        std::vector<std::string> next;
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (i < tokens.size() - 1 && tokens[i] == r.first && tokens[i + 1] == r.second) {
                next.push_back(r.merged);
                i++;
            } else {
                next.push_back(tokens[i]);
            }
        }
        tokens = std::move(next);
    }
    return tokens;
}

std::string BPETrainer::decode(const std::vector<TokenId>& ids, bool skip_special) const {
    std::string res = "";
    for (TokenId id : ids) {
        if (skip_special && id <= 3) continue;
        if (id_to_vocab_.count(id)) res += id_to_vocab_.at(id);
    }
    return res;
}

std::string BPETrainer::id_to_token(TokenId id) const {
    return id_to_vocab_.count(id) ? id_to_vocab_.at(id) : config_.unk_token;
}

bool BPETrainer::save(const std::string& path, const std::string& dataset_id) const {
    std::ofstream out(path, std::ios::binary); if (!out) return false;
    const uint32_t magic_len = static_cast<uint32_t>(std::strlen(kBpeMagic));
    out.write(reinterpret_cast<const char*>(&magic_len), sizeof(magic_len));
    out.write(kBpeMagic, magic_len);
    const std::string id = dataset_id.empty() ? dataset_id_ : dataset_id;
    const uint64_t id_len = static_cast<uint64_t>(id.size());
    out.write(reinterpret_cast<const char*>(&id_len), sizeof(id_len));
    if (id_len > 0) out.write(id.data(), static_cast<std::streamsize>(id_len));
    size_t sz = merge_rules_.size(); out.write((char*)&sz, sizeof(sz));
    for (const auto& r : merge_rules_) {
        size_t s1 = r.first.size(), s2 = r.second.size(), s3 = r.merged.size();
        out.write((char*)&s1, sizeof(s1)); out.write(r.first.data(), s1);
        out.write((char*)&s2, sizeof(s2)); out.write(r.second.data(), s2);
        out.write((char*)&s3, sizeof(s3)); out.write(r.merged.data(), s3);
        out.write((char*)&r.token_id, sizeof(r.token_id));
    }
    uint64_t token_count = static_cast<uint64_t>(cached_tokens_.size());
    out.write(reinterpret_cast<const char*>(&token_count), sizeof(token_count));
    if (token_count > 0) {
        out.write(reinterpret_cast<const char*>(cached_tokens_.data()),
                  static_cast<std::streamsize>(token_count * sizeof(TokenId)));
    }
    return true;
}

bool BPETrainer::load(const std::string& path, const std::string& expected_dataset_id) {
    std::ifstream in(path, std::ios::binary); if (!in) return false;
    build_initial_vocab();
    uint32_t magic_len = 0;
    in.read(reinterpret_cast<char*>(&magic_len), sizeof(magic_len));
    if (!in || magic_len == 0 || magic_len > 1024) return false;
    std::string magic(magic_len, '\0');
    in.read(&magic[0], static_cast<std::streamsize>(magic_len));
    if (!in || magic != kBpeMagic) return false;
    uint64_t id_len = 0;
    in.read(reinterpret_cast<char*>(&id_len), sizeof(id_len));
    if (!in || id_len > (1ull << 20)) return false;
    dataset_id_.assign(static_cast<size_t>(id_len), '\0');
    if (id_len > 0) in.read(&dataset_id_[0], static_cast<std::streamsize>(id_len));
    if (!in) return false;
    if (!expected_dataset_id.empty() && expected_dataset_id != dataset_id_) return false;
    size_t sz; in.read((char*)&sz, sizeof(sz));
    for (size_t i = 0; i < sz; ++i) {
        size_t s1, s2, s3;
        in.read((char*)&s1, sizeof(s1)); std::string f(s1, ' '); in.read(&f[0], s1);
        in.read((char*)&s2, sizeof(s2)); std::string s(s2, ' '); in.read(&s[0], s2);
        in.read((char*)&s3, sizeof(s3)); std::string m(s3, ' '); in.read(&m[0], s3);
        TokenId id; in.read((char*)&id, sizeof(id));
        merge_rules_.push_back({f, s, m, id});
        vocab_[m] = id;
        id_to_vocab_[id] = m;
    }
    uint64_t token_count = 0;
    in.read(reinterpret_cast<char*>(&token_count), sizeof(token_count));
    if (!in) return false;
    if (token_count > (1ull << 31)) return false;
    cached_tokens_.assign(static_cast<size_t>(token_count), 0);
    if (token_count > 0) {
        in.read(reinterpret_cast<char*>(cached_tokens_.data()),
                static_cast<std::streamsize>(token_count * sizeof(TokenId)));
        if (!in) return false;
    }
    return true;
}

void BPETrainer::clear() {
    vocab_.clear();
    id_to_vocab_.clear();
    merge_rules_.clear();
    fast_vocab_.clear();
    dataset_id_.clear();
    cached_tokens_.clear();
}

} // namespace bpe