#ifndef SS_ASSEMBLER_H
#define SS_ASSEMBLER_H

#include "ss/bytecode.h"

SSError ss_assemble_file(const char *path, SSChunk *chunk, SSErrorInfo *error);

#endif
