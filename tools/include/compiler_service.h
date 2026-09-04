#ifndef LANA_COMPILER_SERVICE_H
#define LANA_COMPILER_SERVICE_H

#include "error.h"

#include <stdbool.h>
#include <stddef.h>

/* Compiler service: hosts the self-hosted Lana compiler bytecode in-process and
 * returns structured results. The LSP links this instead of forking `lana
 * check` and inspecting the exit code. */

/* Locate the compiler bytecode (lana-compiler.labc). Returns true and fills
 * `out` when found. */
bool lana_compiler_find(const char *argv0, char *out, size_t out_size);

/* Run the compiler program in-process. Returns 0 on success; on failure fills
 * `error_out` with the structured error. */
int lana_compiler_run(const char *compiler_path, size_t argument_count,
                      const char **arguments, LanaErrorInfo *error_out);

/* Extract the source span and classify a compiler error message. */
void lana_compiler_error_context(LanaErrorInfo *error, const char *source_path);

/* Check source text (which may be an unsaved buffer) and return structured
 * diagnostics. Returns 0 on success; on failure fills `error_out`. */
int lana_compiler_check(const char *compiler_path, const char *source_text,
                        const char *source_path, LanaErrorInfo *error_out);

/* Resolve source text and return the symbol table as a JSON string (malloc'd;
 * the caller frees it with free()). Returns 0 on success. The JSON object has
 * "definitions" (name/kind/type/line/column) and "references" (name/line/column)
 * arrays; line/column are 1-based. */
int lana_compiler_symbols(const char *compiler_path, const char *source_text,
                          const char *source_path, char **json_out);

/* Serialize a LanaErrorInfo to a single LSP diagnostic JSON object. Returns a
 * malloc'd string; the caller frees it with free(). */
char *lana_compiler_diagnostic_json(const LanaErrorInfo *error);

#endif
