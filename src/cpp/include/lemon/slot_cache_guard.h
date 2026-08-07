#ifndef LEMON_SLOT_CACHE_GUARD_H
#define LEMON_SLOT_CACHE_GUARD_H

#include "wrapped_server.h"
#include "server_capabilities.h"
#include "backends/llamacpp/llamacpp_server.h"

namespace lemon {

class SlotSaveGuard {
private:
    WrappedServer* server_;
    int slot_id_;
    std::string key_;
    int version_;
    bool is_big_;
    bool saved_;

public:
    SlotSaveGuard(WrappedServer* server, int slot_id, const std::string& key, 
                  int version, bool is_big);
    ~SlotSaveGuard();
    
    // Delete copy constructor and assignment operator
    SlotSaveGuard(const SlotSaveGuard&) = delete;
    SlotSaveGuard& operator=(const SlotSaveGuard&) = delete;
    
    // Allow move constructor
    SlotSaveGuard(SlotSaveGuard&& other) noexcept;
    SlotSaveGuard& operator=(SlotSaveGuard&& other) noexcept;
};

} // namespace lemon

#endif // LEMON_SLOT_CACHE_GUARD_H