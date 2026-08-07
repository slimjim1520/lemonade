#ifndef LEMON_SLOT_CACHE_MANAGER_H
#define LEMON_SLOT_CACHE_MANAGER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <filesystem>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace lemon {

struct MetaEntry {
    std::string key;
    std::string model_name;
    std::string recipe_fingerprint;
    int prefix_len;
    int wpb;
    std::vector<std::string> blocks;
    double timestamp;
};

class SlotCacheManager {
public:
    SlotCacheManager(const std::string& cache_dir);
    ~SlotCacheManager() = default;

    // Find the best matching candidate for a prompt
    std::pair<std::string, double> find_best_candidate(
        const std::string& model_name,
        const std::string& recipe_fingerprint,
        const std::vector<std::string>& prompt_blocks,
        double threshold);

    // Save a slot context to disk
    bool save_slot(int slot_id, const std::string& key, const std::string& cache_dir);

    // Restore a slot context from disk
    bool restore_slot(int slot_id, const std::string& key, const std::string& cache_dir);

    // Write meta file for a saved slot
    void write_meta_file(const std::string& cache_dir, const std::string& model_name,
                        const std::string& recipe_fingerprint, const std::string& key,
                        const std::vector<std::string>& blocks, int words_per_block);

    // Scan meta files for a model
    std::vector<MetaEntry> scan_meta_files(const std::string& model_dir);

private:
    std::string cache_dir_;
    mutable std::mutex cache_mutex_;

    // Compute LCP (Longest Common Prefix) ratio
    double compute_lcp_ratio(const std::vector<std::string>& blocks1, 
                           const std::vector<std::string>& blocks2) const;

    // Extract prompt text for similarity comparison
    std::string extract_prompt_for_similarity(const json& request) const;
    
    // Helper to convert prompt to word blocks
    std::vector<std::string> prompt_to_word_blocks(const std::string& prompt, int words_per_block) const;
    
    // Helper to compute SHA256 hash
    std::string sha256(const std::string& input) const;
};

} // namespace lemon

#endif // LEMON_SLOT_CACHE_MANAGER_H