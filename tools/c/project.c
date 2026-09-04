#define _POSIX_C_SOURCE 200809L

#include "project.h"

#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static int path_join(char *out, size_t size, const char *left,
                     const char *right) {
    int written = snprintf(out, size, "%s/%s", left, right);
    return written > 0 && (size_t)written < size ? 0 : 1;
}

static int make_directory(const char *path) {
    return mkdir(path, 0755) == 0 || errno == EEXIST ? 0 : 1;
}

static int write_text(const char *path, const char *text) {
    FILE *file = fopen(path, "wb");
    size_t length = strlen(text);
    if (file == NULL) return 1;
    if (fwrite(text, 1u, length, file) != length || fclose(file) != 0)
        return 1;
    return 0;
}

static char *read_text(const char *path, size_t *length) {
    FILE *file = fopen(path, "rb");
    long size;
    char *text;
    if (file == NULL || fseek(file, 0, SEEK_END) != 0 ||
        (size = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        if (file != NULL) (void)fclose(file);
        return NULL;
    }
    text = malloc((size_t)size + 1u);
    if (text == NULL || fread(text, 1u, (size_t)size, file) != (size_t)size) {
        free(text); (void)fclose(file); return NULL;
    }
    text[size] = '\0';
    (void)fclose(file);
    if (length != NULL) *length = (size_t)size;
    return text;
}

static uint64_t hash_bytes(uint64_t hash, const void *data, size_t length) {
    const unsigned char *bytes = data;
    size_t index;
    for (index = 0u; index < length; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

typedef struct { char **items; size_t count; size_t capacity; } PathList;

static int path_compare(const void *left, const void *right) {
    return strcmp(*(const char *const *)left, *(const char *const *)right);
}

static int collect_sources(const char *directory, PathList *paths) {
    DIR *stream = opendir(directory); struct dirent *entry;
    if (stream == NULL) return 1;
    while ((entry = readdir(stream)) != NULL) {
        char path[2048]; struct stat metadata; size_t length = strlen(entry->d_name);
        if (entry->d_name[0] == '.') continue;
        if (path_join(path, sizeof(path), directory, entry->d_name) != 0 ||
            stat(path, &metadata) != 0) { (void)closedir(stream); return 1; }
        if (S_ISDIR(metadata.st_mode)) {
            if (strcmp(entry->d_name, "build") != 0 &&
                strcmp(entry->d_name, "docs") != 0 &&
                collect_sources(path, paths) != 0) { (void)closedir(stream); return 1; }
        } else if (length >= 5u && strcmp(entry->d_name + length - 5u, ".lana") == 0) {
            char **items;
            if (paths->count == paths->capacity) {
                size_t capacity = paths->capacity == 0u ? 8u : paths->capacity * 2u;
                items = realloc(paths->items, capacity * sizeof(*items));
                if (items == NULL) { (void)closedir(stream); return 1; }
                paths->items = items; paths->capacity = capacity;
            }
            paths->items[paths->count] = strdup(path);
            if (paths->items[paths->count++] == NULL) { (void)closedir(stream); return 1; }
        }
    }
    (void)closedir(stream); return 0;
}

static int hash_sources(const char *directory, uint64_t *hash) {
    PathList paths = {0}; size_t index;
    if (collect_sources(directory, &paths) != 0) return 1;
    qsort(paths.items, paths.count, sizeof(*paths.items), path_compare);
    for (index = 0u; index < paths.count; ++index) {
        size_t length; char *source = read_text(paths.items[index], &length);
        if (source == NULL) return 1;
        *hash = hash_bytes(*hash, paths.items[index], strlen(paths.items[index]));
        *hash = hash_bytes(*hash, source, length);
        free(source); free(paths.items[index]);
    }
    free(paths.items); return 0;
}

static int hash_dependencies(const char *directory, const char *manifest,
                             uint64_t *hash, char *locked,
                             size_t locked_size) {
    char *copy = strdup(manifest); char *line; bool dependencies = false;
    size_t used = 0u;
    if (copy == NULL) return 1;
    line = strtok(copy, "\n");
    while (line != NULL) {
        if (strcmp(line, "[dependencies]") == 0) dependencies = true;
        else if (line[0] == '[') dependencies = false;
        else if (dependencies && strchr(line, '=') != NULL) {
            char name[128]; char relative[1024]; char path[2048];
            char *equals = strchr(line, '='); char *quote = strchr(equals, '"');
            char *end = quote == NULL ? NULL : strchr(quote + 1, '"');
            size_t name_length = (size_t)(equals - line);
            while (name_length > 0u && line[name_length - 1u] == ' ') --name_length;
            if (quote == NULL || end == NULL || name_length == 0u ||
                name_length >= sizeof(name) || (size_t)(end - quote - 1) >= sizeof(relative)) {
                free(copy); return 1;
            }
            memcpy(name, line, name_length); name[name_length] = '\0';
            memcpy(relative, quote + 1, (size_t)(end - quote - 1));
            relative[end - quote - 1] = '\0';
            if (relative[0] == '/') (void)snprintf(path, sizeof(path), "%s", relative);
            else if (path_join(path, sizeof(path), directory, relative) != 0) { free(copy); return 1; }
            {
                LanaProject dependency; uint64_t before = *hash; int written;
                if (lana_project_load(path, &dependency) != 0 ||
                    hash_sources(path, hash) != 0) { free(copy); return 1; }
                *hash = hash_bytes(*hash, dependency.name, strlen(dependency.name));
                written = snprintf(locked + used, locked_size - used,
                    "dependency.%s = \"%s:%016llx\"\n", name,
                    dependency.version, (unsigned long long)(*hash ^ before));
                if (written < 0 || (size_t)written >= locked_size - used) { free(copy); return 1; }
                used += (size_t)written;
            }
        }
        line = strtok(NULL, "\n");
    }
    free(copy); return 0;
}

static int quoted_value(const char *text, const char *key, char *out,
                        size_t out_size) {
    char prefix[128];
    const char *start;
    const char *end;
    size_t length;
    (void)snprintf(prefix, sizeof(prefix), "%s = \"", key);
    start = strstr(text, prefix);
    if (start == NULL) return 1;
    start += strlen(prefix);
    end = strchr(start, '"');
    if (end == NULL || (length = (size_t)(end - start)) >= out_size) return 1;
    memcpy(out, start, length); out[length] = '\0';
    return 0;
}

int lana_project_new(const char *directory) {
    char path[2048];
    const char *name = strrchr(directory, '/');
    if (access(directory, F_OK) == 0) {
        (void)fprintf(stderr, "project directory already exists: %s\n", directory);
        return 1;
    }
    name = name == NULL ? directory : name + 1;
    if (*name == '\0' || make_directory(directory) != 0) return 1;
    if (path_join(path, sizeof(path), directory, "src") != 0 ||
        make_directory(path) != 0) return 1;
    if (path_join(path, sizeof(path), directory, "tests") != 0 ||
        make_directory(path) != 0) return 1;
    if (path_join(path, sizeof(path), directory, "lana.toml") != 0) return 1;
    {
        char manifest[512];
        int written = snprintf(manifest, sizeof(manifest),
            "schema = 1\nname = \"%s\"\nversion = \"0.1.0\"\nentry = \"src/main.lana\"\n\n[dependencies]\n",
            name);
        if (written < 0 || (size_t)written >= sizeof(manifest) ||
            write_text(path, manifest) != 0) return 1;
    }
    if (path_join(path, sizeof(path), directory, "src/belief.lana") != 0 ||
        write_text(path,
            "fn probability() {\n"
            "    state belief = state(p: 0.75, d: 0.25);\n"
            "    return measure belief as probability;\n"
            "}\n") != 0)
        return 1;
    if (path_join(path, sizeof(path), directory, "src/main.lana") != 0 ||
        write_text(path,
            "import \"./belief.lana\" as belief;\n\n"
            "fn main() { print(belief.probability()); }\n"
            "main();\n") != 0)
        return 1;
    if (path_join(path, sizeof(path), directory, "tests/main_test.lana") != 0 ||
        write_text(path,
            "import \"../src/belief.lana\" as belief;\n\n"
            "assert(belief.probability() == 0.75, \"first program probability\");\n") != 0)
        return 1;
    (void)printf("created %s\n", directory);
    return 0;
}

int lana_project_load(const char *directory, LanaProject *project) {
    char path[2048];
    char *text;
    if (path_join(path, sizeof(path), directory, "lana.toml") != 0 ||
        (text = read_text(path, NULL)) == NULL) {
        (void)fprintf(stderr, "lana.toml not found\n"); return 1;
    }
    if (strstr(text, "schema = 1") == NULL ||
        quoted_value(text, "name", project->name, sizeof(project->name)) != 0 ||
        quoted_value(text, "version", project->version,
                     sizeof(project->version)) != 0 ||
        quoted_value(text, "entry", project->entry,
                     sizeof(project->entry)) != 0) {
        free(text); (void)fprintf(stderr, "invalid lana.toml schema\n"); return 1;
    }
    free(text); return 0;
}

static int copy_file(const char *source, const char *destination) {
    size_t length;
    char *bytes = read_text(source, &length);
    FILE *file;
    if (bytes == NULL || (file = fopen(destination, "wb")) == NULL) {
        free(bytes); return 1;
    }
    if (fwrite(bytes, 1u, length, file) != length || fclose(file) != 0) {
        free(bytes); return 1;
    }
    free(bytes); return 0;
}

static int project_finish_build(const char *directory, const LanaProject *project,
                                uint64_t hash, const char *locked_dependencies,
                                LanaProjectCompileFn compile, void *context,
                                char *output, size_t output_size) {
    char source_path[2048], lana_dir[2048], cache_dir[2048];
    char build_dir[2048], cache_path[2048], lock_path[2048];
    if (project == NULL || path_join(source_path, sizeof(source_path), directory,
                                     project->entry) != 0) return 1;
    if (path_join(lana_dir, sizeof(lana_dir), directory, ".lana") ||
        make_directory(lana_dir) ||
        path_join(cache_dir, sizeof(cache_dir), lana_dir, "cache") ||
        make_directory(cache_dir) ||
        path_join(build_dir, sizeof(build_dir), directory, "build") ||
        make_directory(build_dir)) return 1;
    if (snprintf(cache_path, sizeof(cache_path), "%s/%016llx.labc", cache_dir,
                 (unsigned long long)hash) <= 0 ||
        snprintf(output, output_size, "%s/%s.labc", build_dir, project->name) <= 0)
        return 1;
    if (access(cache_path, R_OK) != 0 && compile(source_path, cache_path, context) != 0)
        return 1;
    if (copy_file(cache_path, output) != 0) return 1;
    if (path_join(lock_path, sizeof(lock_path), directory, "lana.lock") != 0)
        return 1;
    {
        char lock[4608];
        (void)snprintf(lock, sizeof(lock),
            "schema = 1\nproject = \"%s\"\ncontent = \"%016llx\"\n%s",
            project->name, (unsigned long long)hash, locked_dependencies);
        if (write_text(lock_path, lock) != 0) return 1;
    }
    (void)printf("built %s\n", output);
    return 0;
}

int lana_project_build(const char *directory, LanaProjectCompileFn compile,
                       void *context, char *output, size_t output_size) {
    LanaProject project;
    char manifest_path[2048], source_path[2048];
    char locked_dependencies[4096] = "";
    char *manifest; char *source; size_t manifest_length, source_length;
    uint64_t hash = UINT64_C(1469598103934665603);
    if (lana_project_load(directory, &project) != 0 ||
        path_join(manifest_path, sizeof(manifest_path), directory, "lana.toml") ||
        path_join(source_path, sizeof(source_path), directory, project.entry)) return 1;
    manifest = read_text(manifest_path, &manifest_length);
    source = read_text(source_path, &source_length);
    if (manifest == NULL || source == NULL) { free(manifest); free(source); return 1; }
    hash = hash_bytes(hash, manifest, manifest_length);
    hash = hash_bytes(hash, source, source_length);
    if (hash_sources(directory, &hash) != 0 ||
        hash_dependencies(directory, manifest, &hash, locked_dependencies,
                          sizeof(locked_dependencies)) != 0) {
        free(manifest); free(source); return 1;
    }
    free(manifest); free(source);
    return project_finish_build(directory, &project, hash, locked_dependencies,
                                compile, context, output, output_size);
}

int lana_project_build_with_plan(const char *directory, LanaProjectPlanFn plan,
                                 LanaProjectCompileFn compile, void *context,
                                 char *output, size_t output_size) {
    LanaProject project;
    char plan_path[] = "/tmp/lana-project-plan-XXXXXX";
    char content[32], locked_dependencies[4096] = "";
    char *text; char *dependency; char *end;
    unsigned long long parsed; int descriptor, result;
    if (plan == NULL || lana_project_load(directory, &project) != 0) return 1;
    descriptor = mkstemp(plan_path);
    if (descriptor < 0) return 1;
    (void)close(descriptor);
    result = plan(directory, plan_path, context);
    text = result == 0 ? read_text(plan_path, NULL) : NULL;
    (void)unlink(plan_path);
    if (text == NULL || quoted_value(text, "content", content, sizeof(content)) != 0 ||
        strlen(content) != 16u) { free(text); return 1; }
    errno = 0; parsed = strtoull(content, &end, 16);
    if (errno != 0 || *end != '\0') { free(text); return 1; }
    dependency = strstr(text, "\ndependency.");
    if (dependency != NULL && strlen(dependency + 1u) >= sizeof(locked_dependencies)) {
        free(text); return 1;
    }
    if (dependency != NULL) (void)snprintf(locked_dependencies,
                                            sizeof(locked_dependencies), "%s",
                                            dependency + 1u);
    free(text);
    return project_finish_build(directory, &project, (uint64_t)parsed,
                                locked_dependencies, compile, context, output,
                                output_size);
}

int lana_project_check(const char *directory, LanaProjectCompileFn compile,
                       void *context) {
    char output[2048];
    return lana_project_build(directory, compile, context, output,
                              sizeof(output));
}

int lana_project_test(const char *directory, const char *executable) {
    char tests_path[2048]; DIR *tests; struct dirent *entry; size_t count = 0u;
    char cmake_path[2048];
    if (path_join(cmake_path, sizeof(cmake_path), directory, "CMakeLists.txt") == 0 &&
        access(cmake_path, R_OK) == 0) {
        pid_t child = fork(); int status;
        if (child == 0) { execlp("ctest", "ctest", "--test-dir", "build", "--output-on-failure", (char *)NULL); _exit(127); }
        if (child < 0 || waitpid(child, &status, 0) < 0 || !WIFEXITED(status)) return 1;
        return WEXITSTATUS(status);
    }
    if (path_join(tests_path, sizeof(tests_path), directory, "tests") != 0 ||
        (tests = opendir(tests_path)) == NULL) return 1;
    while ((entry = readdir(tests)) != NULL) {
        char source[2048]; pid_t child; int status; size_t length = strlen(entry->d_name);
        if (length < 5u || strcmp(entry->d_name + length - 5u, ".lana") != 0) continue;
        if (path_join(source, sizeof(source), tests_path, entry->d_name) != 0) { (void)closedir(tests); return 1; }
        child = fork();
        if (child == 0) { execl(executable, executable, "run", source, (char *)NULL); _exit(127); }
        if (child < 0 || waitpid(child, &status, 0) < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0) { (void)closedir(tests); return 1; }
        ++count;
    }
    (void)closedir(tests); (void)printf("%zu project tests passed\n", count); return 0;
}

static int format_file(const char *path, int check_only, int *changed) {
    size_t length, read = 0u, written = 0u; char *source = read_text(path, &length); char *output;
    if (source == NULL || (output = malloc(length + 2u)) == NULL) { free(source); return 1; }
    while (read < length) {
        size_t start = read, end;
        while (read < length && source[read] != '\n') ++read;
        end = read;
        while (end > start && (source[end - 1u] == ' ' || source[end - 1u] == '\t' || source[end - 1u] == '\r')) --end;
        memcpy(output + written, source + start, end - start); written += end - start;
        output[written++] = '\n';
        if (read < length) ++read;
    }
    if (length == 0u) output[written++] = '\n';
    output[written] = '\0';
    if (written != length || memcmp(source, output, length) != 0) {
        *changed = 1;
        if (!check_only && write_text(path, output) != 0) { free(source); free(output); return 1; }
    }
    free(source); free(output); return 0;
}

static int visit_lana_files(const char *directory, int check_only, int *changed) {
    DIR *stream = opendir(directory); struct dirent *entry;
    if (stream == NULL) return errno == ENOENT ? 0 : 1;
    while ((entry = readdir(stream)) != NULL) {
        char path[2048]; size_t length = strlen(entry->d_name);
        if (entry->d_name[0] == '.') continue;
        if (path_join(path, sizeof(path), directory, entry->d_name) != 0) { (void)closedir(stream); return 1; }
        {
            struct stat metadata;
            if (stat(path, &metadata) != 0) { (void)closedir(stream); return 1; }
            if (S_ISDIR(metadata.st_mode)) { if (visit_lana_files(path, check_only, changed) != 0) { (void)closedir(stream); return 1; } }
            else if (length >= 5u && strcmp(entry->d_name + length - 5u, ".lana") == 0 && format_file(path, check_only, changed) != 0) { (void)closedir(stream); return 1; }
        }
    }
    (void)closedir(stream); return 0;
}

int lana_project_format(const char *directory, int check_only) {
    char path[2048]; int changed = 0;
    if (path_join(path, sizeof(path), directory, "src") || visit_lana_files(path, check_only, &changed) ||
        path_join(path, sizeof(path), directory, "tests") || visit_lana_files(path, check_only, &changed)) return 1;
    if (check_only && changed) { (void)fprintf(stderr, "format changes required\n"); return 1; }
    (void)printf(changed ? "formatted\n" : "already formatted\n"); return 0;
}

int lana_project_document(const char *directory) {
    LanaProject project; char docs[2048], output[2048], source_path[2048];
    char *source; size_t length; FILE *file; char *line;
    if (lana_project_load(directory, &project) || path_join(docs, sizeof(docs), directory, "docs") || make_directory(docs) ||
        path_join(output, sizeof(output), docs, "api.md") || path_join(source_path, sizeof(source_path), directory, project.entry)) return 1;
    source = read_text(source_path, &length); if (source == NULL) return 1;
    file = fopen(output, "wb"); if (file == NULL) { free(source); return 1; }
    (void)fprintf(file, "# %s API\n\nVersion %s.\n\n", project.name, project.version);
    (void)fprintf(file,
        "## Language metadata\n\n"
        "Types include `Information<T>`, `Claim<T, Proposition>`, `Sample<T>`, "
        "`PlannedEffect<T, Kind>`, `Result<T, E>`, task handles, and "
        "`SharedCapability<T, read|observe|admin>`. Effects are pure, observation, "
        "stochastic, I/O, mutation, task, and external-call. Compiler diagnostics "
        "use stable `LANA_ERR_*` codes and source spans.\n\n"
        "```lana\nlet source = information(possibility([1, 2]));\n"
        "let sample_record = sample(source);\n"
        "let exact_value = sample_value(sample_record);\n```\n\n");
    line = strtok(source, "\n");
    while (line != NULL) { if (strncmp(line, "fn ", 3u) == 0) (void)fprintf(file, "## `%.*s`\n\n```lana\n%s\n```\n\n", (int)(strchr(line, '{') == NULL ? strlen(line + 3u) : (size_t)(strchr(line, '{') - line - 3)), line + 3u, line); line = strtok(NULL, "\n"); }
    free(source); if (fclose(file) != 0) return 1;
    (void)printf("generated %s\n", output); return 0;
}
