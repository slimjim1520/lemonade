#ifndef LEMON_SLOT_CACHE_GUARD_H
#define LEMON_SLOT_CACHE_GUARD_H

#include "wrapped_server.h"
#include "server_capabilities.h"

namespace lemon {

class SlotCacheManager;

class SlotSaveGuard {
private:
    WrappedServer* server_;
    int slot_id_;
    std::string key_;
    int version_;
    bool is_big_;
    bool saved_;
    SlotCacheManager* cache_manager_;
    std::string model_id_;
    std::string prompt_;
    int words_per_block_;
    
    void save_to_cache();

public:
    SlotSaveGuard(WrappedServer* server, int slot_id, const std::string& key, 
                  int version, bool is_big,
                  SlotCacheManager* cache_manager = nullptr,
                  const std::string& model_id = "",
                  const std::string& prompt = "",
                  int words_per_block = 100);
    ~SlotSaveGuard();
    
    SlotSaveGuard(const SlotSaveGuard&) = delete;
    SlotSaveGuard& operator=(const SlotSaveGuard&) = delete;
    
    SlotSaveGuard(SlotSaveGuard&& other) noexcept;
    SlotSaveGuard& operator=(SlotSaveGuard&& other) noexcept;
};

} // namespace lemon

#endif // LEMON_SLOT_CACHE_GUARD_H