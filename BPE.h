#ifndef BPE_H
#define BPE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <map>
#include <regex>
#include <fstream>
#include <cstdint>

namespace bpe {

using TokenId = int32_t;

// ���� ID ����
constexpr TokenId UNK_TOKEN_ID = 0;
constexpr TokenId BOS_TOKEN_ID = 1;
constexpr TokenId EOS_TOKEN_ID = 2;
constexpr TokenId PAD_TOKEN_ID = 3;

struct MergeRule {
    std::string first;
    std::string second;
    std::string merged;
    TokenId token_id;
};

struct BPEConfig {
    size_t vocab_size = 2000;
    size_t min_frequency = 1;
    std::string unk_token = "<unk>";
    std::string bos_token = "<s>";
    std::string eos_token = "</s>";
    std::string pad_token = "<pad>";
    // GPT-2 ������򣬴����ո�Ͷ��ֽ��ַ�
    std::regex pattern = std::regex(R"('s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z\x80-\xff]+| ?[0-9]+| ?[^\s\w\x80-\xff]+|\s+)");
};

class BPETrainer {
public:
    BPETrainer();
    explicit BPETrainer(const BPEConfig& config);
    ~BPETrainer() = default;
    std::vector<TokenId> encode_fast(const std::string& text, bool add_special = false) const;
    bool train_from_file(const std::string& file_path);
    bool train_from_texts(const std::vector<std::string>& texts);
    std::vector<TokenId> encode(const std::string& text, bool add_special = false) const;
    std::string decode(const std::vector<TokenId>& ids, bool skip_special = false) const;
    bool save(const std::string& file_path, const std::string& dataset_id = "") const;
    bool load(const std::string& file_path, const std::string& expected_dataset_id = "");
    std::string dataset_id() const { return dataset_id_; }
    void set_cached_tokens(const std::vector<TokenId>& tokens) { cached_tokens_ = tokens; }
    const std::vector<TokenId>& cached_tokens() const { return cached_tokens_; }
    bool has_cached_tokens() const { return !cached_tokens_.empty(); }
    
    size_t vocab_size() const { return vocab_.size(); }
    std::string id_to_token(TokenId id) const;
    void clear();

private:
    void build_initial_vocab();
    std::vector<std::string> apply_merges(const std::string& word) const;

    BPEConfig config_;
    std::unordered_map<std::string, TokenId> vocab_;
    std::unordered_map<TokenId, std::string> id_to_vocab_;
    std::vector<MergeRule> merge_rules_;
    std::map<std::pair<std::string, std::string>, std::string> merge_map_;
    mutable std::unordered_map<std::string, std::vector<TokenId>> fast_vocab_;
    std::string dataset_id_;
    std::vector<TokenId> cached_tokens_;
};

} // namespace bpe
#endif
