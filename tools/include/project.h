#ifndef LANA_PROJECT_H
#define LANA_PROJECT_H

#include <stddef.h>

typedef struct {
    char name[128];
    char version[32];
    char entry[1024];
} LanaProject;

typedef int (*LanaProjectCompileFn)(const char *source, const char *output,
                                    void *context);
typedef int (*LanaProjectPlanFn)(const char *directory, const char *output,
                                 void *context);

int lana_project_new(const char *directory);
int lana_project_load(const char *directory, LanaProject *project);
int lana_project_build(const char *directory, LanaProjectCompileFn compile,
                       void *context, char *output, size_t output_size);
int lana_project_build_with_plan(const char *directory, LanaProjectPlanFn plan,
                                 LanaProjectCompileFn compile, void *context,
                                 char *output, size_t output_size);
int lana_project_check(const char *directory, LanaProjectCompileFn compile,
                       void *context);
int lana_project_test(const char *directory, const char *executable);
int lana_project_format(const char *directory, int check_only);
int lana_project_document(const char *directory);

#endif
