#include "lemon/slot_cache_guard.h"
#include "lemon/slot_cache_manager.h"
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
        
        std::string context_key = slots_server->get_slot_context_key(slot_id_);
        int context_version = slots_server->get_slot_context_version(slot_id_);
        
        if (context_key == key_ && context_version == version_ && server_->is_backend_alive()) {
            std::string cache_dir = "";
            if (slots_server->save_slot(slot_id_, key_, cache_dir)) {
                saved_ = true;
                LOG(DEBUG, "SlotCache") << "Saved slot " << slot_id_ << " with key " << key_ << std::endl;
                
                // Write meta file for LCP-based matching (proxycache-style)
                if (cache_manager_ && !model_id_.empty() && !prompt_.empty()) {
                    auto prompt_blocks = cache_manager_->prompt_to_word_blocks(prompt_, words_per_block_);
                    std::string model_cache_dir = cache_manager_->get_cache_dir() + "/" + model_id_;
                    cache_manager_->write_meta_file(model_cache_dir, model_id_, key_, prompt_blocks, words_per_block_);
                }
            } else {
                LOG(WARNING, "SlotCache") << "Failed to save slot " << slot_id_ << std::endl;
            }
        } else {
            LOG(DEBUG, "SlotCache") << "Skipping save for slot " << slot_id_ 
                                    << " - context mismatch or server not alive" << std::endl;
        }
        
        slots_server->unregister_slot_assignment(slot_id_);
    }
}

SlotSaveGuard::~SlotSaveGuard() {
    save_to_cache();
}

SlotSaveGuard::SlotSaveGuard(SlotSaveGuard&& other) noexcept
    : server_(other.server_), slot_id_(other.slot_id_), key_(other.key_),
      version_(other.version_), is_big_(other.is_big_), saved_(other.saved_) {
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
        
        other.saved_ = true;
    }
    return *this;
}

} // namespace lemon