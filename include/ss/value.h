#ifndef SS_VALUE_H
#define SS_VALUE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "ss/state.h"
#include "ss/ml.h"

typedef enum {
    VAL_NULL = 0,
    VAL_NUMBER,
    VAL_BOOL,
    VAL_STRING,
    VAL_STATE,
    VAL_DISTRIBUTION,
    VAL_SAMPLE,
    VAL_JOINT_STATE,
    VAL_ARRAY,
    VAL_FUNCTION,
    VAL_TASK,
    VAL_MODEL,
    VAL_STATE_DIST,
    VAL_MAP,
    VAL_POSSIBILITY,
    VAL_PATH_SET
} ValueType;

typedef struct {
    double p0;
    double p1;
} SSDistribution;

typedef struct Value Value;
typedef struct SSTask SSTask;
typedef struct SSStateDist SSStateDist;
typedef struct SSMap SSMap;
typedef struct SSPossibility SSPossibility;
typedef struct SSPathSet SSPathSet;

typedef enum {
    SS_JOINT_INDEPENDENT = 0,
    SS_JOINT_FINITE_LAW,
    SS_JOINT_CONDITIONAL,
    SS_JOINT_PROJECTED
} SSJointKind;

typedef enum {
    SS_JOINT_CAN_PROJECT = 1u << 0,
    SS_JOINT_CAN_CONDITION = 1u << 1,
    SS_JOINT_CAN_SAMPLE = 1u << 2,
    SS_JOINT_CAN_RESOLVE = 1u << 3
} SSJointCapability;

typedef struct {
    ValueType type;
} SSJointDomain;

typedef struct {
    Value *values;
    double weight;
} SSJointRow;

/* An immutable named product-space law/view.  The legacy two-state opcode is
 * retained separately and is not reinterpreted as this representation. */
typedef struct {
    size_t count;
    char **names;
    SSJointDomain *domains;
    /* Independent marginals. NULL for a finite correlated law. */
    Value *values;
    size_t row_count;
    SSJointRow *rows;
    SSJointKind kind;
    uint32_t capabilities;
} SSJointState;

typedef enum {
    SS_DIST_DIRAC = 0,
    SS_DIST_APPEND,
    SS_DIST_TRANSFORM
} SSStateDistKind;

struct SSStateDist {
    SSStateDistKind kind;
    union {
        SSStateValue dirac;
        struct {
            SSStateDist *left;
            SSStateDist *right;
            bool has_cached_parameters;
            double p;
            double m_re;
            double m_im;
            double sigma;
        } append;
        struct {
            SSStateDist *child;
            uint32_t transform_id;
        } transform;
    } as;
};

typedef struct {
    size_t count;
    size_t capacity;
    Value *items;
} SSArray;

typedef struct {
    const char *key;
    Value *value;
} SSMapEntry;

struct SSMap {
    size_t count;
    size_t capacity;
    SSMapEntry *entries;
};

struct SSPossibility {
    size_t count;
    Value *values;
    double *weights; /* NULL means non-probabilistic, equipossible support. */
    uint64_t dependency_id;
};

typedef struct {
    bool guard;
    double weight;
    Value *result;
} SSPathAlternative;

struct SSPathSet {
    size_t count;
    SSPathAlternative *alternatives;
    uint64_t dependency_id;
};

struct Value {
    ValueType type;
    union {
        double number;
        bool boolean;
        const char *string;
        SSStateValue state;
        SSDistribution distribution;
        int sample;
        SSJointState *joint;
        SSArray *array;
        uint32_t function;
        SSTask *task;
        SSMLModel *model;
        SSStateDist *state_dist;
        SSMap *map;
        SSPossibility *possibility;
        SSPathSet *paths;
    } as;
};

Value ss_value_null(void);
Value ss_value_number(double number);
Value ss_value_bool(bool boolean);
Value ss_value_string(const char *string);
Value ss_value_state(SSState state);
Value ss_value_distribution(double p0, double p1);
Value ss_value_sample(int sample);
Value ss_value_model(SSMLModel *model);
Value ss_value_state_dist(SSStateDist *distribution);
Value ss_value_map(SSMap *map);
Value ss_value_array(SSArray *array);
Value ss_value_possibility(SSPossibility *possibility);
Value ss_value_paths(SSPathSet *paths);
const char *ss_value_type_name(ValueType type);
void ss_value_print(const Value *value);

#endif
