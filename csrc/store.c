#include "store_internal.h"

#include "lana/data.h"
#include "lana/sha256.h"

#include <errno.h>
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

#define STORE_SCHEMA 1u
#define STORE_LIMIT (256u * 1024u * 1024u)

typedef struct { unsigned char *data; size_t length; size_t capacity; } Bytes;

static LanaError bytes_add(Bytes *bytes, const void *data, size_t length) {
    size_t capacity;
    unsigned char *grown;
    if (length > SIZE_MAX - bytes->length || bytes->length + length > STORE_LIMIT)
        return LANA_ERR_LIMIT;
    if (bytes->length + length > bytes->capacity) {
        capacity = bytes->capacity == 0u ? 128u : bytes->capacity;
        while (capacity < bytes->length + length) capacity *= 2u;
        grown = realloc(bytes->data, capacity);
        if (grown == NULL) return LANA_ERR_OOM;
        bytes->data = grown;
        bytes->capacity = capacity;
    }
    memcpy(bytes->data + bytes->length, data, length);
    bytes->length += length;
    return LANA_OK;
}

static LanaError bytes_u32(Bytes *bytes, uint32_t value) {
    unsigned char encoded[4] = {
        (unsigned char)(value >> 24u), (unsigned char)(value >> 16u),
        (unsigned char)(value >> 8u), (unsigned char)value
    };
    return bytes_add(bytes, encoded, sizeof(encoded));
}

static LanaError bytes_u64(Bytes *bytes, uint64_t value) {
    unsigned char encoded[8];
    size_t index;
    for (index = 0u; index < 8u; ++index)
        encoded[7u - index] = (unsigned char)(value >> (index * 8u));
    return bytes_add(bytes, encoded, sizeof(encoded));
}

static bool read_u32(const unsigned char **cursor, const unsigned char *end,
                     uint32_t *value) {
    if ((size_t)(end - *cursor) < 4u) return false;
    *value = ((uint32_t)(*cursor)[0] << 24u) |
             ((uint32_t)(*cursor)[1] << 16u) |
             ((uint32_t)(*cursor)[2] << 8u) | (*cursor)[3];
    *cursor += 4u;
    return true;
}

static bool read_u64(const unsigned char **cursor, const unsigned char *end,
                     uint64_t *value) {
    size_t index;
    if ((size_t)(end - *cursor) < 8u) return false;
    *value = 0u;
    for (index = 0u; index < 8u; ++index) *value = (*value << 8u) | (*cursor)[index];
    *cursor += 8u;
    return true;
}

static char *store_path(const LanaStore *store, const char *name) {
    size_t length = strlen(store->path) + strlen(name) + 2u;
    char *path = malloc(length);
    if (path != NULL) (void)snprintf(path, length, "%s/%s", store->path, name);
    return path;
}

LanaError lana_store_get_path(LanaStore *store, char **out_path) {
    if (store == NULL || out_path == NULL) return LANA_ERR_INVALID_STATE;
    *out_path = strdup(store->path);
    return (*out_path == NULL) ? LANA_ERR_OOM : LANA_OK;
}

static char *snapshot_path(const LanaStore *store, uint64_t revision, bool temporary) {
    char name[80];
    (void)snprintf(name, sizeof(name), "snapshot-%llu%s",
                   (unsigned long long)revision, temporary ? ".tmp" : "");
    return store_path(store, name);
}

static size_t hash_key(const char *key, size_t size) {
    size_t hash = 5381u;
    while (*key != '\0') hash = ((hash << 5u) + hash) + (unsigned char)*key++;
    return hash % size;
}

static StoreEntry *index_find(StoreEntry **index, size_t size, const char *key) {
    StoreEntry *entry = index[hash_key(key, size)];
    while (entry != NULL && strcmp(entry->key, key) != 0) entry = entry->next;
    return entry;
}

static LanaError index_put(LanaStore *store, const StoreMutation *mutation) {
    size_t bucket = hash_key(mutation->key, store->index_size);
    StoreEntry *entry = index_find(store->index, store->index_size, mutation->key);
    unsigned char *copy = malloc(mutation->length + 1u);
    if (copy == NULL) return LANA_ERR_OOM;
    memcpy(copy, mutation->data, mutation->length);
    copy[mutation->length] = '\0';
    if (entry == NULL) {
        entry = calloc(1u, sizeof(*entry));
        if (entry == NULL) { free(copy); return LANA_ERR_OOM; }
        entry->key = strdup(mutation->key);
        if (entry->key == NULL) { free(copy); free(entry); return LANA_ERR_OOM; }
        entry->next = store->index[bucket];
        store->index[bucket] = entry;
    } else {
        free(entry->data);
    }
    entry->data = copy;
    entry->length = mutation->length;
    return LANA_OK;
}

static void index_delete(LanaStore *store, const char *key) {
    size_t bucket = hash_key(key, store->index_size);
    StoreEntry **entry = &store->index[bucket];
    while (*entry != NULL) {
        if (strcmp((*entry)->key, key) == 0) {
            StoreEntry *removed = *entry;
            *entry = removed->next;
            free(removed->key); free(removed->data); free(removed);
            return;
        }
        entry = &(*entry)->next;
    }
}

static void mutation_free(StoreMutation *mutation) {
    free(mutation->key);
    free(mutation->data);
    memset(mutation, 0, sizeof(*mutation));
}

static LanaError mutation_copy(StoreMutation *destination,
                               const StoreMutation *source) {
    memset(destination, 0, sizeof(*destination));
    destination->key = strdup(source->key);
    if (destination->key == NULL) return LANA_ERR_OOM;
    destination->deleted = source->deleted;
    destination->length = source->length;
    if (!source->deleted) {
        destination->data = malloc(source->length + 1u);
        if (destination->data == NULL) { mutation_free(destination); return LANA_ERR_OOM; }
        memcpy(destination->data, source->data, source->length);
        destination->data[source->length] = '\0';
    }
    return LANA_OK;
}

static LanaError revision_add(LanaStore *store, uint64_t id, uint64_t previous,
                              const StoreMutation *mutations, size_t count,
                              const unsigned char digest[32]) {
    StoreRevision *revision;
    size_t index;
    if (store->revision_count == store->revision_capacity) {
        size_t capacity = store->revision_capacity == 0u ? 8u : store->revision_capacity * 2u;
        StoreRevision *grown = realloc(store->revisions, capacity * sizeof(*grown));
        if (grown == NULL) return LANA_ERR_OOM;
        store->revisions = grown;
        store->revision_capacity = capacity;
    }
    revision = &store->revisions[store->revision_count];
    memset(revision, 0, sizeof(*revision));
    revision->mutations = calloc(count, sizeof(*revision->mutations));
    if (count != 0u && revision->mutations == NULL) return LANA_ERR_OOM;
    for (index = 0u; index < count; ++index) {
        LanaError error = mutation_copy(&revision->mutations[index], &mutations[index]);
        if (error != LANA_OK) {
            while (index != 0u) mutation_free(&revision->mutations[--index]);
            free(revision->mutations);
            revision->mutations = NULL;
            return error;
        }
    }
    revision->id = id;
    revision->previous = previous;
    revision->count = count;
    memcpy(revision->digest, digest, 32u);
    ++store->revision_count;
    return LANA_OK;
}

static LanaError apply_mutations(LanaStore *store, const StoreMutation *mutations,
                                 size_t count) {
    size_t index;
    for (index = 0u; index < count; ++index) {
        LanaError error;
        if (mutations[index].deleted) index_delete(store, mutations[index].key);
        else if ((error = index_put(store, &mutations[index])) != LANA_OK) return error;
    }
    return LANA_OK;
}

static LanaError build_payload(const StoreMutation *mutations, size_t count,
                               Bytes *payload) {
    size_t index;
    for (index = 0u; index < count; ++index) {
        uint32_t key_length;
        unsigned char value_digest[32];
        LanaError error;
        if (strlen(mutations[index].key) > UINT32_MAX) return LANA_ERR_LIMIT;
        key_length = (uint32_t)strlen(mutations[index].key);
        if ((error = bytes_add(payload, mutations[index].deleted ? "D" : "P", 1u)) != LANA_OK ||
            (error = bytes_u32(payload, key_length)) != LANA_OK ||
            (error = bytes_add(payload, mutations[index].key, key_length)) != LANA_OK)
            return error;
        if (!mutations[index].deleted) {
            lana_sha256(mutations[index].data, mutations[index].length, value_digest);
            if ((error = bytes_add(payload, value_digest, 32u)) != LANA_OK ||
                (error = bytes_u64(payload, mutations[index].length)) != LANA_OK ||
                (error = bytes_add(payload, mutations[index].data, mutations[index].length)) != LANA_OK)
                return error;
        }
    }
    return LANA_OK;
}

static void revision_digest(uint64_t revision, uint64_t previous, uint32_t count,
                            const Bytes *payload, unsigned char digest[32]) {
    Bytes canonical = {0};
    (void)bytes_u64(&canonical, revision);
    (void)bytes_u64(&canonical, previous);
    (void)bytes_u32(&canonical, count);
    (void)bytes_add(&canonical, payload->data, payload->length);
    lana_sha256(canonical.data, canonical.length, digest);
    free(canonical.data);
}

static LanaError parse_payload(const unsigned char *data, size_t length,
                               uint32_t count, StoreMutation **out) {
    const unsigned char *cursor = data, *end = data + length;
    StoreMutation *mutations = calloc(count, sizeof(*mutations));
    uint32_t index;
    if (count != 0u && mutations == NULL) return LANA_ERR_OOM;
    for (index = 0u; index < count; ++index) {
        uint32_t key_length;
        if (cursor == end || (*cursor != 'P' && *cursor != 'D')) goto corrupt;
        mutations[index].deleted = *cursor++ == 'D';
        if (!read_u32(&cursor, end, &key_length) || key_length == 0u ||
            (size_t)(end - cursor) < key_length) goto corrupt;
        mutations[index].key = malloc((size_t)key_length + 1u);
        if (mutations[index].key == NULL) goto oom;
        memcpy(mutations[index].key, cursor, key_length);
        mutations[index].key[key_length] = '\0'; cursor += key_length;
        if (!mutations[index].deleted) {
            uint64_t value_length;
            unsigned char actual[32];
            if ((size_t)(end - cursor) < 32u) goto corrupt;
            const unsigned char *expected = cursor; cursor += 32u;
            if (!read_u64(&cursor, end, &value_length) || value_length > STORE_LIMIT ||
                value_length > (uint64_t)(end - cursor)) goto corrupt;
            lana_sha256(cursor, (size_t)value_length, actual);
            if (memcmp(actual, expected, 32u) != 0) goto corrupt;
            mutations[index].data = malloc((size_t)value_length + 1u);
            if (mutations[index].data == NULL) goto oom;
            memcpy(mutations[index].data, cursor, (size_t)value_length);
            mutations[index].data[value_length] = '\0';
            mutations[index].length = (size_t)value_length;
            cursor += value_length;
        }
    }
    if (cursor != end) goto corrupt;
    *out = mutations;
    return LANA_OK;
oom:
    for (index = 0u; index < count; ++index) mutation_free(&mutations[index]);
    free(mutations); return LANA_ERR_OOM;
corrupt:
    for (index = 0u; index < count; ++index) mutation_free(&mutations[index]);
    free(mutations); return LANA_ERR_CORRUPTION;
}

static void decoded_value_free(Value value) {
    size_t index;
    if (value.type == VAL_STRING) free((void *)value.as.string);
    else if (value.type == VAL_ARRAY && value.as.array != NULL) {
        for (index = 0u; index < value.as.array->count; ++index)
            decoded_value_free(value.as.array->items[index]);
        free(value.as.array->items); free(value.as.array);
    } else if (value.type == VAL_MAP && value.as.map != NULL) {
        for (index = 0u; index < value.as.map->count; ++index) {
            free((void *)value.as.map->entries[index].key);
            if (value.as.map->entries[index].value != NULL) {
                decoded_value_free(*value.as.map->entries[index].value);
                free(value.as.map->entries[index].value);
            }
        }
        free(value.as.map->entries); free(value.as.map);
    }
}

static LanaError read_exact(FILE *file, void *data, size_t length) {
    return fread(data, 1u, length, file) == length ? LANA_OK : LANA_ERR_CORRUPTION;
}

static LanaError load_snapshot(LanaStore *store) {
    unsigned char magic[4], header[48], actual[32];
    const unsigned char *cursor, *end;
    uint64_t revision, length;
    unsigned char *payload = NULL;
    char *path = snapshot_path(store, store->snapshot_rev, false);
    FILE *file;
    LanaBuffer buffer;
    Value value = lana_value_null();
    LanaError error;
    size_t index;

    if (path == NULL) return LANA_ERR_OOM;
    file = fopen(path, "rb"); free(path);
    if (file == NULL) return errno == ENOENT ? LANA_ERR_CORRUPTION : LANA_ERR_IO;
    error = read_exact(file, magic, sizeof(magic));
    if (error == LANA_OK && memcmp(magic, "LSNP", 4u) != 0) error = LANA_ERR_CORRUPTION;
    if (error == LANA_OK) error = read_exact(file, header, sizeof(header));
    cursor = header; end = header + sizeof(header);
    if (error == LANA_OK && (!read_u64(&cursor, end, &revision) ||
                             !read_u64(&cursor, end, &length) ||
                             revision != store->snapshot_rev || length > STORE_LIMIT))
        error = LANA_ERR_CORRUPTION;
    if (error == LANA_OK) {
        payload = malloc((size_t)length);
        if (length != 0u && payload == NULL) error = LANA_ERR_OOM;
    }
    if (error == LANA_OK) error = read_exact(file, payload, (size_t)length);
    if (error == LANA_OK && fgetc(file) != EOF) error = LANA_ERR_CORRUPTION;
    if (fclose(file) != 0 && error == LANA_OK) error = LANA_ERR_IO;
    if (error != LANA_OK) { free(payload); return error; }
    lana_sha256(payload, (size_t)length, actual);
    if (memcmp(actual, header + 16u, sizeof(actual)) != 0) { free(payload); return LANA_ERR_CORRUPTION; }
    buffer = (LanaBuffer){payload, (size_t)length, (size_t)length};
    error = lana_codec_decode_document(&buffer, &value);
    if (error != LANA_OK || value.type != VAL_MAP) {
        free(payload); decoded_value_free(value); return error == LANA_OK ? LANA_ERR_CORRUPTION : error;
    }
    for (index = 0u; index < value.as.map->count && error == LANA_OK; ++index) {
        LanaBuffer encoded = {0};
        StoreMutation mutation = {0};
        mutation.key = (char *)value.as.map->entries[index].key;
        error = lana_codec_encode_value(&encoded, *value.as.map->entries[index].value);
        if (error == LANA_OK) {
            mutation.data = encoded.data; mutation.length = encoded.length;
            error = index_put(store, &mutation);
        }
        free(encoded.data);
    }
    decoded_value_free(value); free(payload);
    if (error == LANA_OK) store->current_rev = store->snapshot_rev;
    return error;
}

static LanaError replay(LanaStore *store) {
    long start;
    uint64_t journal_revision = 0u;
    bool saw_record = false;
    for (;;) {
        unsigned char magic[4], digest[32], commit_digest[32];
        uint64_t revision, previous, payload_length, commit_revision;
        uint32_t count;
        unsigned char *payload;
        StoreMutation *mutations;
        Bytes payload_view;
        unsigned char actual[32];
        start = ftell(store->journal);
        size_t read = fread(magic, 1u, 4u, store->journal);
        if (read == 0u && feof(store->journal)) {
            clearerr(store->journal);
            return (!saw_record && store->snapshot_rev != 0u) ||
                   journal_revision >= store->snapshot_rev ? LANA_OK : LANA_ERR_CORRUPTION;
        }
        if (read != 4u) { clearerr(store->journal); (void)fseek(store->journal, start, SEEK_SET); return LANA_OK; }
        if (memcmp(magic, "LREV", 4u) != 0) return LANA_ERR_CORRUPTION;
        unsigned char header[28];
        if (fread(header, 1u, sizeof(header), store->journal) != sizeof(header) ||
            fread(digest, 1u, 32u, store->journal) != 32u) goto partial;
        const unsigned char *cursor = header, *end = header + sizeof(header);
        if (!read_u64(&cursor, end, &revision) || !read_u64(&cursor, end, &previous) ||
            !read_u32(&cursor, end, &count) || !read_u64(&cursor, end, &payload_length) ||
            payload_length > STORE_LIMIT) return LANA_ERR_CORRUPTION;
        payload = malloc((size_t)payload_length);
        if (payload_length != 0u && payload == NULL) return LANA_ERR_OOM;
        if (fread(payload, 1u, (size_t)payload_length, store->journal) != payload_length ||
            fread(magic, 1u, 4u, store->journal) != 4u) { free(payload); goto partial; }
        unsigned char trailer[8];
        if (fread(trailer, 1u, 8u, store->journal) != 8u ||
            fread(commit_digest, 1u, 32u, store->journal) != 32u) { free(payload); goto partial; }
        cursor = trailer; end = trailer + 8u;
        if (journal_revision == 0u && store->snapshot_rev != 0u &&
            revision == store->snapshot_rev + 1u && previous == store->snapshot_rev)
            journal_revision = store->snapshot_rev;
        if (memcmp(magic, "LCMT", 4u) != 0 || !read_u64(&cursor, end, &commit_revision) ||
            commit_revision != revision || previous != journal_revision ||
            revision != previous + 1u || memcmp(digest, commit_digest, 32u) != 0) {
            free(payload); return LANA_ERR_CORRUPTION;
        }
        payload_view = (Bytes){payload, (size_t)payload_length, (size_t)payload_length};
        revision_digest(revision, previous, count, &payload_view, actual);
        if (memcmp(actual, digest, 32u) != 0) { free(payload); return LANA_ERR_CORRUPTION; }
        LanaError error = parse_payload(payload, (size_t)payload_length, count, &mutations);
        free(payload);
        if (error != LANA_OK) return error;
        error = revision_add(store, revision, previous, mutations, count, digest);
        if (error == LANA_OK && revision > store->snapshot_rev)
            error = apply_mutations(store, mutations, count);
        for (uint32_t index = 0u; index < count; ++index) mutation_free(&mutations[index]);
        free(mutations);
        if (error != LANA_OK) return error;
        journal_revision = revision;
        saw_record = true;
        if (revision > store->snapshot_rev) store->current_rev = revision;
        store->journal_offset = ftell(store->journal);
    }
partial:
        clearerr(store->journal);
        (void)fseek(store->journal, start, SEEK_SET);
        return LANA_OK;
}

static LanaError write_manifest(LanaStore *store) {
    char *path = store_path(store, "manifest");
    FILE *file;
    int result;
    if (path == NULL) return LANA_ERR_OOM;
    file = fopen(path, "wb"); free(path);
    if (file == NULL) return LANA_ERR_IO;
    result = fprintf(file, "LANA_STORE %u\ncurrent %llu\nsnapshot %llu\nretention %llu\n",
                     STORE_SCHEMA, (unsigned long long)store->current_rev,
                     (unsigned long long)store->snapshot_rev,
                     (unsigned long long)store->retention_boundary);
    if (result < 0 || fflush(file) != 0 || fsync(fileno(file)) != 0) result = -1;
    if (fclose(file) != 0) result = -1;
    return result < 0 ? LANA_ERR_IO : LANA_OK;
}

static LanaError read_manifest(LanaStore *store) {
    char *path = store_path(store, "manifest");
    FILE *file;
    unsigned int schema;
    unsigned long long current, snapshot, retention;
    int result;
    if (path == NULL) return LANA_ERR_OOM;
    file = fopen(path, "rb"); free(path);
    if (file == NULL) return errno == ENOENT ? LANA_OK : LANA_ERR_IO;
    result = fscanf(file, "LANA_STORE %u\ncurrent %llu\nsnapshot %llu\nretention %llu",
                    &schema, &current, &snapshot, &retention);
    if (fclose(file) != 0) return LANA_ERR_IO;
    if (result != 4 || schema != STORE_SCHEMA || snapshot > current || retention > current)
        return LANA_ERR_CORRUPTION;
    store->snapshot_rev = (uint64_t)snapshot;
    store->retention_boundary = (uint64_t)retention;
    return LANA_OK;
}

LanaError lana_store_open(const LanaStoreOptions *options, LanaStore **out_store) {
    LanaStore *store;
    char *journal_path;
    LanaError error;
    if (options == NULL || out_store == NULL || options->path == NULL) return LANA_ERR_INVALID_STATE;
    *out_store = NULL;
    if (options->struct_size < sizeof(*options) || options->schema_version != STORE_SCHEMA)
        return LANA_ERR_INCOMPATIBLE_FORMAT;
    if (mkdir(options->path, 0755) != 0 && errno != EEXIST) return LANA_ERR_IO;
    store = calloc(1u, sizeof(*store));
    if (store == NULL) return LANA_ERR_OOM;
    store->path = strdup(options->path); store->index_size = 1024u;
    store->index = calloc(store->index_size, sizeof(*store->index));
    journal_path = store_path(store, "journal");
    if (store->path == NULL || store->index == NULL || journal_path == NULL) { free(journal_path); return LANA_ERR_OOM; }
    store->journal = fopen(journal_path, "a+b"); free(journal_path);
    if (store->journal == NULL) return LANA_ERR_IO;
    store->journal_fd = fileno(store->journal);
    if (pthread_mutex_init(&store->lock, NULL) != 0) return LANA_ERR_IO;
    if (flock(store->journal_fd, LOCK_EX) != 0) return LANA_ERR_IO;
    error = read_manifest(store);
    if (error == LANA_OK && store->snapshot_rev != 0u) error = load_snapshot(store);
    rewind(store->journal);
    if (error == LANA_OK) error = replay(store);
    if (error != LANA_OK) { (void)lana_store_close(store); return error; }
    store->journal_offset = ftell(store->journal);
    error = write_manifest(store);
    if (error != LANA_OK) { (void)lana_store_close(store); return error; }
    *out_store = store;
    return LANA_OK;
}

static LanaError stage(LanaStore *store, const char *key, const Value *value,
                       bool deleted) {
    StoreMutation mutation = {0};
    LanaBuffer encoded = {0};
    StoreMutation *grown;
    LanaError error;
    if (store == NULL || key == NULL || key[0] == '\0') return LANA_ERR_KEY;
    mutation.key = strdup(key);
    if (mutation.key == NULL) return LANA_ERR_OOM;
    mutation.deleted = deleted;
    if (!deleted) {
        error = lana_codec_encode_value(&encoded, *value);
        if (error != LANA_OK) { mutation_free(&mutation); return error; }
        mutation.data = encoded.data; mutation.length = encoded.length;
    }
    pthread_mutex_lock(&store->lock);
    if (store->staged_count == store->staged_capacity) {
        size_t capacity = store->staged_capacity == 0u ? 8u : store->staged_capacity * 2u;
        grown = realloc(store->staged, capacity * sizeof(*grown));
        if (grown == NULL) { pthread_mutex_unlock(&store->lock); mutation_free(&mutation); return LANA_ERR_OOM; }
        store->staged = grown; store->staged_capacity = capacity;
    }
    store->staged[store->staged_count++] = mutation;
    pthread_mutex_unlock(&store->lock);
    return LANA_OK;
}

LanaError lana_store_put(LanaStore *store, const char *key, Value value) {
    return stage(store, key, &value, false);
}

LanaError lana_store_delete(LanaStore *store, const char *key) {
    return stage(store, key, NULL, true);
}

LanaError lana_store_commit(LanaStore *store, LanaStoreRevisionInfo *out_revision) {
    Bytes payload = {0}, header = {0};
    unsigned char digest[32];
    uint64_t revision;
    LanaError error;
    size_t index;
    if (store == NULL) return LANA_ERR_INVALID_STATE;
    pthread_mutex_lock(&store->lock);
    if (store->staged_count == 0u) { pthread_mutex_unlock(&store->lock); return LANA_ERR_UNSUPPORTED_OPERATION; }
    error = build_payload(store->staged, store->staged_count, &payload);
    if (error != LANA_OK) goto done;
    revision = store->current_rev + 1u;
    revision_digest(revision, store->current_rev, (uint32_t)store->staged_count, &payload, digest);
    if ((error = bytes_add(&header, "LREV", 4u)) != LANA_OK ||
        (error = bytes_u64(&header, revision)) != LANA_OK ||
        (error = bytes_u64(&header, store->current_rev)) != LANA_OK ||
        (error = bytes_u32(&header, (uint32_t)store->staged_count)) != LANA_OK ||
        (error = bytes_u64(&header, payload.length)) != LANA_OK ||
        (error = bytes_add(&header, digest, sizeof(digest))) != LANA_OK) goto done;
    if (ftruncate(store->journal_fd, store->journal_offset) != 0 ||
        fseek(store->journal, store->journal_offset, SEEK_SET) != 0 ||
        fwrite(header.data, 1u, header.length, store->journal) != header.length ||
        fwrite(payload.data, 1u, payload.length, store->journal) != payload.length ||
        fwrite("LCMT", 1u, 4u, store->journal) != 4u) error = LANA_ERR_IO;
    Bytes trailer = {0};
    if (error == LANA_OK) error = bytes_u64(&trailer, revision);
    if (error == LANA_OK) error = bytes_add(&trailer, digest, 32u);
    if (error == LANA_OK && fwrite(trailer.data, 1u, trailer.length, store->journal) != trailer.length) error = LANA_ERR_IO;
    if (error == LANA_OK && (fflush(store->journal) != 0 || fsync(store->journal_fd) != 0)) error = LANA_ERR_IO;
    free(trailer.data);
    if (error != LANA_OK) goto done;
    error = revision_add(store, revision, store->current_rev, store->staged,
                         store->staged_count, digest);
    if (error == LANA_OK) error = apply_mutations(store, store->staged, store->staged_count);
    if (error != LANA_OK) goto done;
    store->current_rev = revision; store->journal_offset = ftell(store->journal);
    for (index = 0u; index < store->staged_count; ++index) mutation_free(&store->staged[index]);
    store->staged_count = 0u;
    if (out_revision != NULL) {
        memset(out_revision, 0, sizeof(*out_revision));
        out_revision->revision_id = revision; out_revision->schema_version = STORE_SCHEMA;
        memcpy(out_revision->digest, digest, 32u);
    }
    error = write_manifest(store);
done:
    free(header.data); free(payload.data);
    pthread_mutex_unlock(&store->lock);
    return error;
}

static LanaError decode_bytes(LanaVM *vm, const unsigned char *data, size_t length,
                              Value *out) {
    char *text;
    LanaError error;
    if (vm == NULL || out == NULL) return LANA_ERR_INVALID_STATE;
    text = malloc(length + 1u);
    if (text == NULL) return LANA_ERR_OOM;
    memcpy(text, data, length); text[length] = '\0';
    error = lana_json_parse(vm, text, out);
    free(text);
    return error;
}

LanaError lana_store_get(LanaStore *store, LanaVM *vm, const char *key, Value *out_value) {
    StoreEntry *entry;
    LanaError error;
    if (store == NULL || key == NULL || out_value == NULL) return LANA_ERR_INVALID_STATE;
    pthread_mutex_lock(&store->lock);
    entry = index_find(store->index, store->index_size, key);
    error = entry == NULL ? LANA_ERR_NOT_FOUND : decode_bytes(vm, entry->data, entry->length, out_value);
    pthread_mutex_unlock(&store->lock);
    return error;
}

LanaError lana_store_current_revision(LanaStore *store, LanaStoreRevisionInfo *out_revision) {
    if (store == NULL || out_revision == NULL) return LANA_ERR_INVALID_STATE;
    pthread_mutex_lock(&store->lock);
    memset(out_revision, 0, sizeof(*out_revision));
    out_revision->revision_id = store->current_rev; out_revision->schema_version = STORE_SCHEMA;
    if (store->revision_count != 0u) memcpy(out_revision->digest, store->revisions[store->revision_count - 1u].digest, 32u);
    pthread_mutex_unlock(&store->lock);
    return LANA_OK;
}

LanaError lana_store_get_at(LanaStore *store, LanaVM *vm, uint64_t revision,
                            const char *key, Value *out_value) {
    const StoreMutation *latest = NULL;
    size_t i, j;
    if (store == NULL || vm == NULL || key == NULL || out_value == NULL) return LANA_ERR_INVALID_STATE;
    pthread_mutex_lock(&store->lock);
    if (revision > store->current_rev) { pthread_mutex_unlock(&store->lock); return LANA_ERR_NOT_FOUND; }
    if (revision < store->retention_boundary) { pthread_mutex_unlock(&store->lock); return LANA_ERR_COMPACTED_HISTORY; }
    if (revision == store->current_rev) {
        StoreEntry *entry = index_find(store->index, store->index_size, key);
        LanaError error = entry == NULL ? LANA_ERR_NOT_FOUND : decode_bytes(vm, entry->data, entry->length, out_value);
        pthread_mutex_unlock(&store->lock);
        return error;
    }
    for (i = 0u; i < store->revision_count && store->revisions[i].id <= revision; ++i)
        for (j = 0u; j < store->revisions[i].count; ++j)
            if (strcmp(store->revisions[i].mutations[j].key, key) == 0) latest = &store->revisions[i].mutations[j];
    LanaError error = latest == NULL || latest->deleted ? LANA_ERR_NOT_FOUND : decode_bytes(vm, latest->data, latest->length, out_value);
    pthread_mutex_unlock(&store->lock);
    return error;
}

LanaError lana_store_put_persistent_state(LanaStore *store, const char *key,
                                          const LanaPersistentState *state) {
    void *encoded = NULL;
    size_t length = 0u;
    LanaBuffer buffer;
    Value value = lana_value_null();
    LanaError error = lana_persistent_state_encode(state, &encoded, &length);
    if (error != LANA_OK) return error;
    buffer = (LanaBuffer){encoded, length, length};
    error = lana_codec_decode_document(&buffer, &value);
    if (error == LANA_OK) error = lana_store_put(store, key, value);
    decoded_value_free(value); free(encoded);
    return error;
}

static LanaError persistent_state_from_value(Value value, LanaPersistentState *out_state) {
    LanaBuffer encoded = {0};
    LanaError error;
    if (out_state == NULL) return LANA_ERR_INVALID_STATE;
    error = lana_codec_encode_value(&encoded, value);
    if (error == LANA_OK)
        error = lana_persistent_state_decode(encoded.data, encoded.length, out_state);
    free(encoded.data);
    return error;
}

LanaError lana_store_get_persistent_state(LanaStore *store, LanaVM *vm,
                                          const char *key, LanaPersistentState *out_state) {
    Value value;
    LanaError error = lana_store_get(store, vm, key, &value);
    return error == LANA_OK ? persistent_state_from_value(value, out_state) : error;
}

LanaError lana_store_get_persistent_state_at(LanaStore *store, LanaVM *vm,
                                             uint64_t revision, const char *key,
                                             LanaPersistentState *out_state) {
    Value value;
    LanaError error = lana_store_get_at(store, vm, revision, key, &value);
    return error == LANA_OK ? persistent_state_from_value(value, out_state) : error;
}

LanaError lana_store_history(LanaStore *store, const char *key,
                             LanaStoreHistory *out_history) {
    size_t i, j, count = 0u;
    if (store == NULL || key == NULL || out_history == NULL) return LANA_ERR_INVALID_STATE;
    memset(out_history, 0, sizeof(*out_history));
    pthread_mutex_lock(&store->lock);
    for (i = 0u; i < store->revision_count; ++i)
        for (j = 0u; j < store->revisions[i].count; ++j)
            if (strcmp(store->revisions[i].mutations[j].key, key) == 0) ++count;
    if (count == 0u) { pthread_mutex_unlock(&store->lock); return LANA_ERR_NOT_FOUND; }
    out_history->records = calloc(count, sizeof(*out_history->records));
    if (out_history->records == NULL) { pthread_mutex_unlock(&store->lock); return LANA_ERR_OOM; }
    count = 0u;
    for (i = 0u; i < store->revision_count; ++i) {
        for (j = 0u; j < store->revisions[i].count; ++j) {
            StoreMutation *mutation = &store->revisions[i].mutations[j];
            if (strcmp(mutation->key, key) != 0) continue;
            out_history->records[count].revision = store->revisions[i].id;
            out_history->records[count].key = strdup(key);
            if (mutation->deleted) out_history->records[count].value = lana_value_null();
            else {
                LanaBuffer buffer = {mutation->data, mutation->length, mutation->length};
                if (lana_codec_decode_document(&buffer, &out_history->records[count].value) != LANA_OK) {
                    pthread_mutex_unlock(&store->lock); return LANA_ERR_CORRUPTION;
                }
            }
            ++count;
        }
    }
    out_history->count = count;
    pthread_mutex_unlock(&store->lock);
    return LANA_OK;
}

static int scan_record_compare(const void *left, const void *right) {
    const LanaStoreScanRecord *a = left, *b = right;
    return strcmp(a->key, b->key);
}

LanaError lana_store_scan(LanaStore *store, LanaVM *vm, const char *prefix,
                          LanaStoreScanRecord **out_records, size_t *out_count) {
    LanaStoreScanRecord *records = NULL;
    size_t count = 0u, capacity = 0u, bucket, prefix_length;
    LanaError error = LANA_OK;
    if (store == NULL || vm == NULL || prefix == NULL || out_records == NULL || out_count == NULL)
        return LANA_ERR_INVALID_STATE;
    prefix_length = strlen(prefix);
    pthread_mutex_lock(&store->lock);
    for (bucket = 0u; bucket < store->index_size; ++bucket) {
        StoreEntry *entry;
        for (entry = store->index[bucket]; entry != NULL; entry = entry->next) {
            if (strncmp(entry->key, prefix, prefix_length) != 0) continue;
            if (count == capacity) {
                size_t new_capacity = capacity == 0u ? 8u : capacity * 2u;
                LanaStoreScanRecord *grown = realloc(records, new_capacity * sizeof(*grown));
                if (grown == NULL) { error = LANA_ERR_OOM; goto done; }
                records = grown; capacity = new_capacity;
            }
            records[count].key = strdup(entry->key);
            if (records[count].key == NULL) { error = LANA_ERR_OOM; goto done; }
            error = decode_bytes(vm, entry->data, entry->length, &records[count].value);
            if (error != LANA_OK) { free(records[count].key); goto done; }
            ++count;
        }
    }
    qsort(records, count, sizeof(*records), scan_record_compare);
done:
    pthread_mutex_unlock(&store->lock);
    if (error != LANA_OK) {
        lana_store_scan_free(records, count);
        return error;
    }
    *out_records = records; *out_count = count;
    return LANA_OK;
}

void lana_store_scan_free(LanaStoreScanRecord *records, size_t count) {
    size_t index;
    if (records == NULL) return;
    /* Values are VM-owned (decoded via lana_json_parse); only keys are freed. */
    for (index = 0u; index < count; ++index) free(records[index].key);
    free(records);
}

LanaError lana_store_snapshot(LanaStore *store, LanaVM *vm, Value *out_value,
                              LanaStoreRevisionInfo *out_revision) {
    LanaMap *map;
    LanaBuffer encoded = {0};
    Bytes header = {0};
    unsigned char digest[32];
    char *temporary = NULL, *published = NULL;
    FILE *file = NULL;
    LanaError error;
    size_t bucket;
    if (store == NULL || vm == NULL || out_value == NULL) return LANA_ERR_INVALID_STATE;
    pthread_mutex_lock(&store->lock);
    error = lana_map_new(vm, store->index_size, &map);
    for (bucket = 0u; error == LANA_OK && bucket < store->index_size; ++bucket) {
        StoreEntry *entry;
        for (entry = store->index[bucket]; error == LANA_OK && entry != NULL; entry = entry->next) {
            Value value;
            error = decode_bytes(vm, entry->data, entry->length, &value);
            if (error == LANA_OK) error = lana_map_set(vm, map, entry->key, &value, true);
        }
    }
    if (error == LANA_OK) {
        *out_value = lana_value_map(map);
        error = lana_codec_encode_value(&encoded, *out_value);
    }
    if (error == LANA_OK) {
        temporary = snapshot_path(store, store->current_rev, true);
        published = snapshot_path(store, store->current_rev, false);
        if (temporary == NULL || published == NULL) error = LANA_ERR_OOM;
    }
    if (error == LANA_OK) {
        lana_sha256(encoded.data, encoded.length, digest);
        if ((error = bytes_add(&header, "LSNP", 4u)) != LANA_OK ||
            (error = bytes_u64(&header, store->current_rev)) != LANA_OK ||
            (error = bytes_u64(&header, encoded.length)) != LANA_OK ||
            (error = bytes_add(&header, digest, sizeof(digest))) != LANA_OK) goto done;
    }
    if (error == LANA_OK) {
        file = fopen(temporary, "wb");
        if (file == NULL || fwrite(header.data, 1u, header.length, file) != header.length ||
            fwrite(encoded.data, 1u, encoded.length, file) != encoded.length ||
            fflush(file) != 0 || fsync(fileno(file)) != 0) error = LANA_ERR_IO;
    }
    if (file != NULL && fclose(file) != 0) error = LANA_ERR_IO;
    if (error == LANA_OK && rename(temporary, published) != 0) error = LANA_ERR_IO;
    if (error == LANA_OK) {
        store->snapshot_rev = store->current_rev;
        error = write_manifest(store);
    }
    if (error == LANA_OK && out_revision != NULL) {
        memset(out_revision, 0, sizeof(*out_revision));
        out_revision->revision_id = store->current_rev;
        out_revision->schema_version = STORE_SCHEMA;
        memcpy(out_revision->digest, digest, sizeof(digest));
    }
done:
    if (error != LANA_OK && temporary != NULL) (void)unlink(temporary);
    free(temporary); free(published); free(header.data); free(encoded.data);
    pthread_mutex_unlock(&store->lock);
    return error;
}

LanaError lana_store_compact(LanaStore *store, uint64_t retention,
                             LanaStoreRevisionInfo *out_revision) {
    uint64_t boundary;
    LanaError error;
    if (store == NULL) return LANA_ERR_INVALID_STATE;
    if (retention == 0u) {
        LanaVM vm;
        Value snapshot;
        lana_vm_init(&vm, NULL);
        error = lana_store_snapshot(store, &vm, &snapshot, NULL);
        lana_vm_free(&vm);
        if (error != LANA_OK) return error;
    }
    pthread_mutex_lock(&store->lock);
    boundary = retention >= store->current_rev ? 0u : store->current_rev - retention;
    store->retention_boundary = boundary;
    if (retention == 0u) {
        size_t index, mutation;
        if (fflush(store->journal) != 0 || ftruncate(store->journal_fd, 0) != 0 ||
            fseek(store->journal, 0, SEEK_SET) != 0 || fsync(store->journal_fd) != 0) {
            pthread_mutex_unlock(&store->lock);
            return LANA_ERR_IO;
        }
        store->journal_offset = 0;
        for (index = 0u; index < store->revision_count; ++index) {
            for (mutation = 0u; mutation < store->revisions[index].count; ++mutation)
                mutation_free(&store->revisions[index].mutations[mutation]);
            free(store->revisions[index].mutations);
        }
        store->revision_count = 0u;
    }
    error = write_manifest(store);
    if (error == LANA_OK && out_revision != NULL) {
        memset(out_revision, 0, sizeof(*out_revision));
        out_revision->revision_id = store->current_rev;
        out_revision->schema_version = STORE_SCHEMA;
        if (store->revision_count != 0u)
            memcpy(out_revision->digest, store->revisions[store->revision_count - 1u].digest, 32u);
    }
    pthread_mutex_unlock(&store->lock);
    return error;
}

LanaError lana_store_close(LanaStore *store) {
    size_t bucket, index, mutation;
    if (store == NULL) return LANA_OK;
    for (bucket = 0u; bucket < store->index_size; ++bucket) {
        StoreEntry *entry = store->index[bucket];
        while (entry != NULL) { StoreEntry *next = entry->next; free(entry->key); free(entry->data); free(entry); entry = next; }
    }
    for (index = 0u; index < store->staged_count; ++index) mutation_free(&store->staged[index]);
    for (index = 0u; index < store->revision_count; ++index) {
        for (mutation = 0u; mutation < store->revisions[index].count; ++mutation)
            mutation_free(&store->revisions[index].mutations[mutation]);
        free(store->revisions[index].mutations);
    }
    if (store->journal != NULL) {
        (void)flock(store->journal_fd, LOCK_UN);
        fclose(store->journal);
    }
    pthread_mutex_destroy(&store->lock);
    free(store->revisions); free(store->staged); free(store->index); free(store->path); free(store);
    return LANA_OK;
}
