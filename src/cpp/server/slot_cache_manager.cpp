#include "lemon/slot_cache_manager.h"
#include "lemon/utils/json_utils.h"
#include <lemon/utils/aixlog.hpp>
#include <openssl/evp.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <regex>
#include <sstream>
#include <iomanip>

namespace lemon {

SlotCacheManager::SlotCacheManager(const std::string& cache_dir) 
    : cache_dir_(cache_dir) {}

std::string SlotCacheManager::sha256(const std::string& input) const {
    unsigned char hash[EVP_MAX_MD_SIZE];
    unsigned int hash_len = 0;
    EVP_Digest(input.c_str(), input.size(), hash, &hash_len, EVP_sha256(), nullptr);
    
    std::stringstream ss;
    for (unsigned int i = 0; i < hash_len; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return ss.str();
}

std::pair<std::string, double> SlotCacheManager::find_best_candidate(
    const std::string& model_id,
    const std::vector<std::string>& prompt_blocks,
    double threshold) {
    
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    // Build the model-specific directory path (sanitize model_id for filesystem)
    std::string safe_model_id = model_id;
    size_t pos = 0;
    while ((pos = safe_model_id.find('/', pos)) != std::string::npos) {
        safe_model_id.replace(pos, 1, "_");
        pos += 1;
    }
    std::filesystem::path model_dir = std::filesystem::path(cache_dir_) / safe_model_id / "meta";
    
    LOG(DEBUG, "SlotCache") << "find_best_candidate: model_id=" << model_id 
                            << " blocks=" << prompt_blocks.size()
                            << " threshold=" << threshold
                            << " dir=" << model_dir.string() << std::endl;
    
    // Check if directory exists
    if (!std::filesystem::exists(model_dir)) {
        LOG(DEBUG, "SlotCache") << "find_best_candidate: directory does not exist: " << model_dir << std::endl;
        stats_.misses.fetch_add(1);
        return {};
    }
    
    // List directory contents for debugging
    try {
        int total_files = 0;
        for (const auto& e : std::filesystem::directory_iterator(model_dir)) {
            total_files++;
        }
        LOG(DEBUG, "SlotCache") << "find_best_candidate: directory has " << total_files << " total entries" << std::endl;
    } catch (const std::exception& e) {
        LOG(WARNING, "SlotCache") << "find_best_candidate: error listing directory: " << e.what() << std::endl;
    }
    
    double best_ratio = 0.0;
    std::string best_key;
    int meta_count = 0;
    
    // Scan all meta files
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
        LOG(DEBUG, "SlotCache") << "find_best_candidate: entry=" << entry.path().filename().string() 
                                << " is_regular=" << entry.is_regular_file()
                                << " ext=" << entry.path().extension().string() << std::endl;
        std::string filename = entry.path().filename().string();
        if (entry.is_regular_file() && filename.size() > 10 &&
            filename.substr(filename.size() - 10) == ".meta.json") {
            meta_count++;
            try {
                // Read and parse the meta file
                std::ifstream file(entry.path());
                json meta_data;
                file >> meta_data;
                
                // Validate the meta file contains expected fields
                if (!meta_data.contains("model_id") || 
                    !meta_data.contains("blocks")) {
                    LOG(DEBUG, "SlotCache") << "find_best_candidate: meta file missing fields: " << entry.path() << std::endl;
                    continue;
                }
                
                // Check if this meta file matches the current model
                if (meta_data["model_id"] != model_id) {
                    LOG(DEBUG, "SlotCache") << "find_best_candidate: model_id mismatch: " << meta_data["model_id"] << " != " << model_id << std::endl;
                    continue;
                }
                
                // Compute LCP ratio
                auto meta_blocks = meta_data["blocks"].get<std::vector<std::string>>();
                double ratio = compute_lcp_ratio(prompt_blocks, meta_blocks);
                
                LOG(DEBUG, "SlotCache") << "find_best_candidate: candidate ratio=" << ratio 
                                        << " (req_blocks=" << prompt_blocks.size() 
                                        << " meta_blocks=" << meta_blocks.size() << ")" << std::endl;
                
                if (ratio >= threshold && ratio > best_ratio) {
                    best_ratio = ratio;
                    best_key = meta_data["key"];
                }
            } catch (const std::exception& e) {
                LOG(WARNING, "SlotCache") << "find_best_candidate: error reading meta file: " << e.what() << std::endl;
                continue;
            }
        }
    }
    
    LOG(DEBUG, "SlotCache") << "find_best_candidate: scanned " << meta_count << " meta files, "
                            << "best_ratio=" << best_ratio << " best_key=" << (best_key.empty() ? "(none)" : best_key.substr(0, 16)) << std::endl;
    
    // Track stats
    if (!best_key.empty()) {
        stats_.hits.fetch_add(1);
    } else {
        stats_.misses.fetch_add(1);
    }
    
    return {best_key, best_ratio};
}

bool SlotCacheManager::save_slot(int slot_id, const std::string& key, const std::string& cache_dir) {
    // This is called after LlamaCppServer has already saved the slot via llama-server's API
    // We just need to update the meta file timestamp
    std::filesystem::path meta_file = std::filesystem::path(cache_dir) / "meta" / (key + ".meta.json");
    
    if (std::filesystem::exists(meta_file)) {
        try {
            std::ifstream file(meta_file);
            json meta_data;
            file >> meta_data;
            
            // Update timestamp
            auto now = std::chrono::system_clock::now();
            meta_data["timestamp"] = std::chrono::duration<double>(now.time_since_epoch()).count();
            
            std::ofstream out_file(meta_file);
            out_file << meta_data.dump(2);
            
            return true;
        } catch (const std::exception& e) {
            LOG(WARNING, "SlotCache") << "Failed to update meta file: " << e.what() << std::endl;
            return false;
        }
    }
    
    return false;
}

bool SlotCacheManager::restore_slot(int slot_id, const std::string& key, const std::string& cache_dir) {
    // Check if the cache file exists
    std::filesystem::path cache_file = std::filesystem::path(cache_dir) / key;
    
    if (!std::filesystem::exists(cache_file)) {
        LOG(WARNING, "SlotCache") << "Cache file not found: " << cache_file << std::endl;
        return false;
    }
    
    // The actual restore is done by calling llama-server's /slots/{id}?action=restore endpoint
    // This is handled by LlamaCppServer::slots_action()
    return true;
}

void SlotCacheManager::write_meta_file(const std::string& cache_dir, const std::string& model_id,
                                       const std::string& key,
                                       const std::vector<std::string>& blocks, int words_per_block) {
    try {
        std::filesystem::path meta_dir = std::filesystem::path(cache_dir) / "meta";
        std::filesystem::create_directories(meta_dir);
        
        json meta_data;
        meta_data["key"] = key;
        meta_data["model_id"] = model_id;
        meta_data["wpb"] = words_per_block;
        meta_data["blocks"] = blocks;
        meta_data["prefix_len"] = blocks.size();
        
        auto now = std::chrono::system_clock::now();
        meta_data["timestamp"] = std::chrono::duration<double>(now.time_since_epoch()).count();
        
        std::filesystem::path meta_file = meta_dir / (key + ".meta.json");
        std::ofstream file(meta_file);
        file << meta_data.dump(2);
        
        LOG(DEBUG, "SlotCache") << "Wrote meta file: " << meta_file 
                                << " model_id=" << model_id 
                                << " blocks=" << blocks.size() 
                                << " wpb=" << words_per_block << std::endl;
    } catch (const std::exception& e) {
        LOG(WARNING, "SlotCache") << "Failed to write meta file: " << e.what() << std::endl;
    }
}

std::vector<MetaEntry> SlotCacheManager::scan_meta_files(const std::string& model_dir) {
    std::vector<MetaEntry> entries;
    
    std::filesystem::path meta_dir = std::filesystem::path(cache_dir_) / model_dir / "meta";
    
    if (!std::filesystem::exists(meta_dir)) {
        return entries;
    }
    
    for (const auto& entry : std::filesystem::directory_iterator(meta_dir)) {
        std::string filename = entry.path().filename().string();
        if (entry.is_regular_file() && filename.size() > 10 &&
            filename.substr(filename.size() - 10) == ".meta.json") {
            try {
                std::ifstream file(entry.path());
                json meta_data;
                file >> meta_data;
                
                MetaEntry meta_entry;
                meta_entry.key = meta_data.value("key", "");
                meta_entry.model_id = meta_data.value("model_id", "");
                meta_entry.prefix_len = meta_data.value("prefix_len", 0);
                meta_entry.wpb = meta_data.value("wpb", 0);
                meta_entry.blocks = meta_data.value("blocks", std::vector<std::string>());
                meta_entry.timestamp = meta_data.value("timestamp", 0.0);
                
                entries.push_back(meta_entry);
            } catch (const std::exception& e) {
                // Skip corrupted or unreadable meta files
                continue;
            }
        }
    }
    
    // Sort by timestamp (newest first)
    std::sort(entries.begin(), entries.end(),
        [](const MetaEntry& a, const MetaEntry& b) {
            return a.timestamp > b.timestamp;
        });
    
    return entries;
}

double SlotCacheManager::compute_lcp_ratio(const std::vector<std::string>& blocks1, 
                                         const std::vector<std::string>& blocks2) const {
    if (blocks1.empty() || blocks2.empty()) {
        return 0.0;
    }
    
    size_t lcp = 0;
    size_t min_len = std::min(blocks1.size(), blocks2.size());
    
    for (size_t i = 0; i < min_len; ++i) {
        if (blocks1[i] == blocks2[i]) {
            lcp++;
        } else {
            break;
        }
    }
    
    // Return ratio of LCP to the shorter sequence length
    return static_cast<double>(lcp) / static_cast<double>(min_len);
}

std::string SlotCacheManager::extract_prompt_for_similarity(const json& request) const {
    std::string prompt = "";
    
    if (request.contains("messages")) {
        // Extract content from all messages (matching proxycache raw_prefix)
        for (const auto& message : request["messages"]) {
            if (message.contains("content")) {
                std::string content;
                if (message["content"].is_string()) {
                    content = message["content"].get<std::string>();
                } else {
                    content = message["content"].dump();
                }
                // Strip whitespace (matching proxycache)
                size_t start = content.find_first_not_of(" \t\n\r");
                size_t end = content.find_last_not_of(" \t\n\r");
                if (start != std::string::npos && end != std::string::npos) {
                    content = content.substr(start, end - start + 1);
                } else {
                    content = "";
                }
                if (!content.empty()) {
                    if (!prompt.empty()) {
                        prompt += "\n\n";
                    }
                    prompt += content;
                }
            }
        }
    }
    
    LOG(DEBUG, "SlotCache") << "extract_prompt_for_similarity: " << prompt.size() << " chars" << std::endl;
    return prompt;
}

std::vector<std::string> SlotCacheManager::prompt_to_word_blocks(const std::string& prompt, int words_per_block) const {
    std::vector<std::string> blocks;
    
    if (prompt.empty()) {
        return blocks;
    }
    
    // Lowercase and extract words using \w+ regex (matching proxycache behavior)
    std::string lower = prompt;
    std::transform(lower.begin(), lower.end(), lower.begin(), 
                    [](unsigned char c) { return std::tolower(c); });
    
    std::regex word_regex(R"(\w+)");
    std::vector<std::string> words;
    auto words_begin = std::sregex_iterator(lower.begin(), lower.end(), word_regex);
    auto words_end = std::sregex_iterator();
    for (auto it = words_begin; it != words_end; ++it) {
        words.push_back(it->str());
    }
    
    LOG(DEBUG, "SlotCache") << "prompt_to_word_blocks: " << words.size() << " words, " 
                            << (words.size() + words_per_block - 1) / words_per_block << " blocks" << std::endl;
    
    // Create blocks
    for (size_t i = 0; i < words.size(); i += words_per_block) {
        std::string block = "";
        for (size_t j = i; j < i + words_per_block && j < words.size(); ++j) {
            if (!block.empty()) {
                block += " ";
            }
            block += words[j];
        }
        blocks.push_back(sha256(block));
    }
    
    return blocks;
}

double SlotCacheManager::get_hit_rate() const {
    uint64_t hits = stats_.hits.load();
    uint64_t misses = stats_.misses.load();
    uint64_t total = hits + misses;
    return (total > 0) ? static_cast<double>(hits) / total : 0.0;
}

double SlotCacheManager::get_avg_restore_time() const {
    uint64_t hits = stats_.hits.load();
    double total_time = stats_.total_restore_time_ms.load();
    return (hits > 0) ? total_time / hits : 0.0;
}

} // namespace lemon