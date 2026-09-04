#ifndef LANA_STORE_INTERNAL_H
#define LANA_STORE_INTERNAL_H

#include "codec.h"
#include "store.h"

#include <pthread.h>

typedef struct {
    char *key;
    unsigned char *data;
    size_t length;
    bool deleted;
} StoreMutation;

typedef struct {
    uint64_t id;
    uint64_t previous;
    StoreMutation *mutations;
    size_t count;
    unsigned char digest[32];
} StoreRevision;

typedef struct StoreEntry {
    char *key;
    unsigned char *data;
    size_t length;
    struct StoreEntry *next;
} StoreEntry;

struct LanaStore {
    char *path;
    FILE *journal;
    int journal_fd;
    long journal_offset;
    pthread_mutex_t lock;
    uint64_t current_rev;
    uint64_t snapshot_rev;
    uint64_t retention_boundary;
    StoreEntry **index;
    size_t index_size;
    StoreMutation *staged;
    size_t staged_count;
    size_t staged_capacity;
    StoreRevision *revisions;
    size_t revision_count;
    size_t revision_capacity;
};

#endif
