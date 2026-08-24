#ifndef LANA_SHARED_H
#define LANA_SHARED_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "lana/error.h"
#include "lana/value.h"

typedef struct LanaVM LanaVM;
typedef struct LanaSharedInformation LanaSharedInformation;
typedef struct LanaCapabilityToken LanaCapabilityToken;

typedef enum {
    LANA_CAPABILITY_READ = 1u << 0,
    LANA_CAPABILITY_OBSERVE = 1u << 1,
    LANA_CAPABILITY_ADMIN = 1u << 2
} LanaCapabilityPermission;

LanaError lana_shared_information_create(LanaVM *source_vm,
                                         const Value *source,
                                         LanaSharedInformation **out,
                                         LanaCapabilityToken **admin);
void lana_shared_information_retain(LanaSharedInformation *shared);
void lana_shared_information_release(LanaSharedInformation *shared);

uint64_t lana_shared_information_identity(
    const LanaSharedInformation *shared);
uint64_t lana_shared_information_revision(
    const LanaSharedInformation *shared);

LanaError lana_shared_capability_grant(LanaCapabilityToken *admin,
                                       uint32_t permissions,
                                       LanaCapabilityToken **out);
LanaError lana_shared_capability_revoke(LanaCapabilityToken *admin,
                                        LanaCapabilityToken *target);
bool lana_shared_capability_allows(const LanaCapabilityToken *capability,
                                   uint32_t permissions);
LanaSharedInformation *lana_shared_capability_information(
    const LanaCapabilityToken *capability);

LanaError lana_shared_information_snapshot(LanaVM *destination,
                                           LanaSharedInformation *shared,
                                           LanaCapabilityToken *read,
                                           Value *out,
                                           uint64_t *revision);
LanaError lana_shared_information_at(LanaVM *destination,
                                     LanaSharedInformation *shared,
                                     LanaCapabilityToken *read,
                                     double effective_time,
                                     Value *out,
                                     uint64_t *revision);
LanaError lana_shared_information_observe(LanaVM *caller,
                                          LanaSharedInformation *shared,
                                          LanaCapabilityToken *observe,
                                          const Value *evidence,
                                          double effective_time,
                                          uint64_t *revision);
LanaError lana_shared_information_wait(LanaVM *destination,
                                       LanaSharedInformation *shared,
                                       LanaCapabilityToken *read,
                                       uint64_t after_revision,
                                       uint64_t timeout_milliseconds,
                                       Value *out,
                                       uint64_t *revision);

#endif
