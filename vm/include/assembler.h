#ifndef LANA_ASSEMBLER_H
#define LANA_ASSEMBLER_H

#include "bytecode.h"

LanaError lana_assemble_file(const char *path, LanaChunk *chunk, LanaErrorInfo *error);

#endif
