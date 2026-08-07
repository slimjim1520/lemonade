#include "lemon/slot_cache_guard.h"
#include "lemon/backends/llamacpp/llamacpp_server.h"
#include "lemon/utils/logger.h"

namespace lemon {

SlotSaveGuard::SlotSaveGuard(WrappedServer* server, int slot_id, const std::string& key, 
                             int version, bool is_big)
    : server_(server), slot_id_(slot_id), key_(key), version_(version), 
      is_big_(is_big), saved_(false) {
    // Constructor does nothing special - save happens on destruction
}

SlotSaveGuard::~SlotSaveGuard() {
    if (!saved_ && is_big_ && server_) {
        // Check that we still own this slot and server is alive
        auto* llamacpp_server = dynamic_cast<LlamaCppServer*>(server_);
        if (llamacpp_server) {
            // Check slot ownership and version
            std::string context_key = llamacpp_server->get_slot_context_key(slot_id_);
            int context_version = llamacpp_server->get_slot_context_version(slot_id_);
            
            if (context_key == key_ && context_version == version_ && server_->is_backend_alive()) {
                // Save the slot context
                std::string cache_dir = ""; // Use default from server config
                if (llamacpp_server->save_slot(slot_id_, key_, cache_dir)) {
                    saved_ = true;
                    LOG(DEBUG, "SlotCache") << "Saved slot " << slot_id_ << " with key " << key_ << std::endl;
                } else {
                    LOG(WARNING, "SlotCache") << "Failed to save slot " << slot_id_ << std::endl;
                }
            } else {
                LOG(DEBUG, "SlotCache") << "Skipping save for slot " << slot_id_ 
                                        << " - context mismatch or server not alive" << std::endl;
            }
            
            // Unregister the slot assignment
            llamacpp_server->unregister_slot_assignment(slot_id_);
        }
    }
}

SlotSaveGuard::SlotSaveGuard(SlotSaveGuard&& other) noexcept
    : server_(other.server_), slot_id_(other.slot_id_), key_(other.key_),
      version_(other.version_), is_big_(other.is_big_), saved_(other.saved_) {
    other.saved_ = true; // Mark as moved
}

SlotSaveGuard& SlotSaveGuard::operator=(SlotSaveGuard&& other) noexcept {
    if (this != &other) {
        // Clean up current state if needed
        if (!saved_ && is_big_ && server_) {
            auto* llamacpp_server = dynamic_cast<LlamaCppServer*>(server_);
            if (llamacpp_server) {
                llamacpp_server->unregister_slot_assignment(slot_id_);
            }
        }
        
        server_ = other.server_;
        slot_id_ = other.slot_id_;
        key_ = other.key_;
        version_ = other.version_;
        is_big_ = other.is_big_;
        saved_ = other.saved_;
        
        other.saved_ = true; // Mark as moved
    }
    return *this;
}

} // namespace lemon