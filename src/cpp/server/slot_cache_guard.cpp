#include "lemon/slot_cache_guard.h"
#include <lemon/utils/aixlog.hpp>

namespace lemon {

SlotSaveGuard::SlotSaveGuard(WrappedServer* server, int slot_id, const std::string& key, 
                             int version, bool is_big)
    : server_(server), slot_id_(slot_id), key_(key), version_(version), 
      is_big_(is_big), saved_(false) {
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