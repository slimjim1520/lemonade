#include "lemon/slot_cache_manager.h"
#include "utils/json_utils.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include <regex>
#include <sstream>
#include <iomanip>
#include <openssl/sha.h>

namespace lemon {

SlotCacheManager::SlotCacheManager(const std::string& cache_dir) 
    : cache_dir_(cache_dir) {}

std::string SlotCacheManager::sha256(const std::string& input) const {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256_ctx;
    SHA256_Init(&sha256_ctx);
    SHA256_Update(&sha256_ctx, input.c_str(), input.size());
    SHA256_Final(hash, &sha256_ctx);
    
    std::stringstream ss;
    for (int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
    }
    return ss.str();
}

std::pair<std::string, double> SlotCacheManager::find_best_candidate(
    const std::string& model_name,
    const std::string& recipe_fingerprint,
    const std::vector<std::string>& prompt_blocks,
    double threshold) {
    
    std::lock_guard<std::mutex> lock(cache_mutex_);
    
    // Build the model-specific directory path
    std::filesystem::path model_dir = std::filesystem::path(cache_dir_) / model_name / "meta";
    
    // Check if directory exists
    if (!std::filesystem::exists(model_dir)) {
        return {};
    }
    
    double best_ratio = 0.0;
    std::string best_key;
    
    // Scan all meta files
    for (const auto& entry : std::filesystem::directory_iterator(model_dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".meta.json") {
            try {
                // Read and parse the meta file
                std::ifstream file(entry.path());
                json meta_data;
                file >> meta_data;
                
                // Validate the meta file contains expected fields
                if (!meta_data.contains("model_name") || 
                    !meta_data.contains("recipe_fingerprint") ||
                    !meta_data.contains("blocks")) {
                    continue;
                }
                
                // Check if this meta file matches the current model and recipe
                if (meta_data["model_name"] != model_name ||
                    meta_data["recipe_fingerprint"] != recipe_fingerprint) {
                    continue;
                }
                
                // Compute LCP ratio
                auto meta_blocks = meta_data["blocks"].get<std::vector<std::string>>();
                double ratio = compute_lcp_ratio(prompt_blocks, meta_blocks);
                
                if (ratio >= threshold && ratio > best_ratio) {
                    best_ratio = ratio;
                    best_key = meta_data["key"];
                }
            } catch (const std::exception& e) {
                // Skip corrupted or unreadable meta files
                continue;
            }
        }
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

void SlotCacheManager::write_meta_file(const std::string& cache_dir, const std::string& model_name,
                                       const std::string& recipe_fingerprint, const std::string& key,
                                       const std::vector<std::string>& blocks, int words_per_block) {
    try {
        std::filesystem::path meta_dir = std::filesystem::path(cache_dir) / "meta";
        std::filesystem::create_directories(meta_dir);
        
        json meta_data;
        meta_data["key"] = key;
        meta_data["model_name"] = model_name;
        meta_data["recipe_fingerprint"] = recipe_fingerprint;
        meta_data["wpb"] = words_per_block;
        meta_data["blocks"] = blocks;
        meta_data["prefix_len"] = blocks.size();
        
        auto now = std::chrono::system_clock::now();
        meta_data["timestamp"] = std::chrono::duration<double>(now.time_since_epoch()).count();
        
        std::filesystem::path meta_file = meta_dir / (key + ".meta.json");
        std::ofstream file(meta_file);
        file << meta_data.dump(2);
        
        LOG(DEBUG, "SlotCache") << "Wrote meta file: " << meta_file << std::endl;
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
        if (entry.is_regular_file() && entry.path().extension() == ".meta.json") {
            try {
                std::ifstream file(entry.path());
                json meta_data;
                file >> meta_data;
                
                MetaEntry meta_entry;
                meta_entry.key = meta_data.value("key", "");
                meta_entry.model_name = meta_data.value("model_name", "");
                meta_entry.recipe_fingerprint = meta_data.value("recipe_fingerprint", "");
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
        // Extract content from all messages
        for (const auto& message : request["messages"]) {
            if (message.contains("content")) {
                std::string content = message["content"].get<std::string>();
                if (!content.empty()) {
                    if (!prompt.empty()) {
                        prompt += "\n\n";
                    }
                    prompt += content;
                }
            }
        }
    }
    
    return prompt;
}

std::vector<std::string> SlotCacheManager::prompt_to_word_blocks(const std::string& prompt, int words_per_block) const {
    std::vector<std::string> blocks;
    
    if (prompt.empty()) {
        return blocks;
    }
    
    // Split prompt into words
    std::istringstream iss(prompt);
    std::vector<std::string> words;
    std::string word;
    while (iss >> word) {
        words.push_back(word);
    }
    
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

} // namespace lemon