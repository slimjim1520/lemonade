#include "lemon/slot_cache_guard.h"
#include "lemon/slot_cache_manager.h"
#include "lemon/runtime_config.h"
#include <lemon/utils/aixlog.hpp>

namespace lemon {

SlotSaveGuard::SlotSaveGuard(WrappedServer* server, int slot_id, const std::string& key, 
                             int version, bool is_big,
                             SlotCacheManager* cache_manager,
                             const std::string& model_id,
                             const std::string& prompt,
                             int words_per_block)
    : server_(server), slot_id_(slot_id), key_(key), version_(version), 
      is_big_(is_big), saved_(false),
      cache_manager_(cache_manager), model_id_(model_id),
      prompt_(prompt), words_per_block_(words_per_block) {
}

void SlotSaveGuard::save_to_cache() {
    if (!saved_ && is_big_ && server_) {
        auto* slots_server = dynamic_cast<ISlotsServer*>(server_);
        if (!slots_server) return;
        
        int context_version = slots_server->get_slot_context_version(slot_id_);
        
        if (context_version == version_ && server_->is_backend_alive()) {
            std::string safe_model_id = model_id_;
            size_t pos = 0;
            while ((pos = safe_model_id.find('/', pos)) != std::string::npos) {
                safe_model_id.replace(pos, 1, "_");
                pos += 1;
            }
            std::string model_cache_dir = cache_manager_->get_cache_dir() + "/" + safe_model_id;
            
            std::string save_key = cache_manager_->sha256(model_id_ + "\n" + prompt_);
            
            LOG(DEBUG, "SlotCache") << "save_to_cache: slot=" << slot_id_ 
                                   << " key=" << save_key.substr(0, 16) 
                                   << " model=" << model_id_
                                   << " cache_dir=" << model_cache_dir << std::endl;
            
            if (slots_server->save_slot(slot_id_, save_key, model_cache_dir)) {
                saved_ = true;
                LOG(DEBUG, "SlotCache") << "Saved slot " << slot_id_ << " with key " << save_key << std::endl;
                
                if (cache_manager_ && !model_id_.empty() && !prompt_.empty()) {
                    auto prompt_blocks = cache_manager_->prompt_to_word_blocks(prompt_, words_per_block_);
                    cache_manager_->write_meta_file(model_cache_dir, model_id_, save_key, prompt_blocks, words_per_block_);
                    slots_server->register_slot_assignment(slot_id_, save_key, prompt_blocks, words_per_block_);
                }
                if (auto* cfg = RuntimeConfig::global()) {
                    cache_manager_->maybe_cleanup(cfg->slot_cache_max_age_seconds(),
                                                  cfg->slot_cache_max_gb());
                }
            } else {
                LOG(WARNING, "SlotCache") << "Failed to save slot " << slot_id_ << std::endl;
            }
        } else {
            LOG(DEBUG, "SlotCache") << "Skipping save for slot " << slot_id_ 
                                    << " version match: " << (context_version == version_)
                                    << " server alive: " << server_->is_backend_alive() << std::endl;
        }
    }
}

SlotSaveGuard::~SlotSaveGuard() {
    save_to_cache();
}

SlotSaveGuard::SlotSaveGuard(SlotSaveGuard&& other) noexcept
    : server_(other.server_), slot_id_(other.slot_id_), key_(other.key_),
      version_(other.version_), is_big_(other.is_big_), saved_(other.saved_),
      cache_manager_(other.cache_manager_), model_id_(other.model_id_),
      prompt_(other.prompt_), words_per_block_(other.words_per_block_) {
    other.saved_ = true; // Mark as moved
}

SlotSaveGuard& SlotSaveGuard::operator=(SlotSaveGuard&& other) noexcept {
    if (this != &other) {
        if (!saved_ && is_big_ && server_) {
            save_to_cache();
        }
        
        server_ = other.server_;
        slot_id_ = other.slot_id_;
        key_ = other.key_;
        version_ = other.version_;
        is_big_ = other.is_big_;
        saved_ = other.saved_;
        cache_manager_ = other.cache_manager_;
        model_id_ = other.model_id_;
        prompt_ = other.prompt_;
        words_per_block_ = other.words_per_block_;
        
        other.saved_ = true;
    }
    return *this;
}

} // namespace lemon