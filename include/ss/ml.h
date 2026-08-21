#ifndef SS_ML_H
#define SS_ML_H

#include <stddef.h>

typedef enum {
    SS_ML_RIDGE = 0,
    SS_ML_LOGISTIC = 1
} SSMLModelKind;

typedef struct {
    SSMLModelKind kind;
    size_t feature_count;
    double *coefficients;
    double *means;
    double *scales;
    double regularization;
} SSMLModel;

#endif
