#ifndef LEMON_SLOT_CACHE_MANAGER_H
#define LEMON_SLOT_CACHE_MANAGER_H

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <functional>
#include <filesystem>
#include "nlohmann/json.hpp"

using json = nlohmann::json;

namespace lemon {

struct MetaEntry {
    std::string key;
    std::string model_id;
    int prefix_len;
    int wpb;
    std::vector<std::string> blocks;
    double timestamp;
};

struct SlotCacheStats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};
    std::atomic<uint64_t> saves{0};
    std::atomic<uint64_t> restore_failures{0};
    std::atomic<double> total_restore_time_ms{0.0};
};

class SlotCacheManager {
public:
    SlotCacheManager(const std::string& cache_dir);
    ~SlotCacheManager();

    // Find the best matching candidate for a prompt
    std::pair<std::string, double> find_best_candidate(
        const std::string& model_id,
        const std::vector<std::string>& prompt_blocks,
        double threshold);

    // Save a slot context to disk
    bool save_slot(int slot_id, const std::string& key, const std::string& cache_dir);

    // Restore a slot context from disk
    bool restore_slot(int slot_id, const std::string& key, const std::string& cache_dir);

    // Delete a cache entry (KV file + meta file) for a stale/corrupt key
    void delete_cache_entry(const std::string& cache_dir, const std::string& model_id,
                            const std::string& key);

    // Delete the entire cache subdir for a model (KV files + meta dir)
    void delete_model_cache(const std::string& model_id);

    // Inspect the cache: returns JSON with per-model entry lists and totals
    json inspect_cache();

    // Delete meta files (and their KV files) older than max_age_seconds.
    // Returns {deleted_count, freed_bytes}.
    struct CleanupResult { size_t deleted_count; size_t freed_bytes; };
    CleanupResult cleanup_by_age(double max_age_seconds, bool dry_run = false);

    // If total cache size exceeds max_gb, evict oldest entries until under limit.
    // Returns {deleted_count, freed_bytes}.
    CleanupResult cleanup_by_size(double max_gb, bool dry_run = false);

    // Throttled inline cleanup: runs age + size cleanup using the provided
    // limits, but skips if called within throttle_seconds of the last run.
    // Intended for call from the save path.
    void maybe_cleanup(double max_age_seconds, double max_gb,
                        double throttle_seconds = 60.0);

    // Start a background thread that periodically runs age + size cleanup.
    // interval_seconds <= 0 means don't start.
    void start_cleanup_thread(double interval_seconds,
                              std::function<double()> get_max_age,
                              std::function<double()> get_max_gb);
    void stop_cleanup_thread();

    // Write meta file for a saved slot
    void write_meta_file(const std::string& cache_dir, const std::string& model_id,
                        const std::string& key,
                        const std::vector<std::string>& blocks, int words_per_block);

    // Scan meta files for a model
    std::vector<MetaEntry> scan_meta_files(const std::string& model_dir);
    
    // Stats accessors
    uint64_t get_hits() const { return stats_.hits.load(); }
    uint64_t get_misses() const { return stats_.misses.load(); }
    uint64_t get_saves() const { return stats_.saves.load(); }
    uint64_t get_restore_failures() const { return stats_.restore_failures.load(); }
    double get_hit_rate() const;
    double get_avg_restore_time() const;
    
    // Get the base cache directory
    const std::string& get_cache_dir() const { return cache_dir_; }
    
    // Extract prompt text for similarity comparison
    std::string extract_prompt_for_similarity(const json& request) const;
    
    // Helper to convert prompt to word blocks
    std::vector<std::string> prompt_to_word_blocks(const std::string& prompt, int words_per_block) const;

    // Helper to compute SHA256 hash
    std::string sha256(const std::string& input) const;

private:
    std::string cache_dir_;
    mutable std::mutex cache_mutex_;
    SlotCacheStats stats_;

    std::atomic<double> last_cleanup_time_{0.0};

    std::thread cleanup_thread_;
    std::atomic<bool> cleanup_stop_{true};
    double cleanup_interval_{0.0};
    std::function<double()> cleanup_get_max_age_;
    std::function<double()> cleanup_get_max_gb_;

    // Compute LCP (Longest Common Prefix) ratio
    double compute_lcp_ratio(const std::vector<std::string>& blocks1, 
                           const std::vector<std::string>& blocks2) const;

};

} // namespace lemon

#endif // LEMON_SLOT_CACHE_MANAGER_H