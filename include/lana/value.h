#ifndef LANA_VALUE_H
#define LANA_VALUE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "lana/state.h"

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
    VAL_STATE_DIST,
    VAL_MAP,
    VAL_POSSIBILITY,
    VAL_PATH_SET,
    VAL_SHARED_CAPABILITY
} ValueType;

typedef struct {
    double p0;
    double p1;
} LanaDistribution;

typedef struct Value Value;
typedef struct LanaTask LanaTask;
typedef struct LanaStateDist LanaStateDist;
typedef struct LanaMap LanaMap;
typedef struct LanaPossibility LanaPossibility;
typedef struct LanaPathSet LanaPathSet;
typedef struct LanaDerivation LanaDerivation;
typedef struct LanaReactive LanaReactive;
typedef struct LanaClaim LanaClaim;
typedef struct LanaPlannedEffect LanaPlannedEffect;
typedef struct LanaEffectReceipt LanaEffectReceipt;
typedef struct LanaCapabilityToken LanaCapabilityToken;

typedef enum {
    LANA_DERIVATION_EVIDENCE = 0,
    LANA_DERIVATION_ASSUMPTION,
    LANA_DERIVATION_OPERATION,
    LANA_DERIVATION_OBSERVATION,
    LANA_DERIVATION_PATH,
    LANA_DERIVATION_SAMPLE,
    LANA_DERIVATION_APPROXIMATION,
    LANA_DERIVATION_RESOLUTION
} LanaDerivationKind;

typedef enum {
    LANA_EXACTNESS_EXACT = 0,
    LANA_EXACTNESS_SAMPLE,
    LANA_EXACTNESS_APPROXIMATE
} LanaDerivationExactness;

typedef enum {
    LANA_DERIVATION_SUCCESS = 0,
    LANA_DERIVATION_UNRESOLVED,
    LANA_DERIVATION_UNSUPPORTED,
    LANA_DERIVATION_ERROR
} LanaDerivationOutcome;

struct LanaDerivation {
    uint64_t task_lineage;
    uint64_t local_sequence;
    uint64_t revision;
    LanaDerivationKind kind;
    const char *operation;
    LanaDerivation **inputs;
    size_t input_count;
    const char *label;
    const char *function;
    uint32_t line;
    LanaDerivationExactness exactness;
    const char *details;
    LanaDerivationOutcome outcome;
    const char *reason;
};

typedef enum {
    LANA_REACTIVE_ROOT = 0,
    LANA_REACTIVE_BINARY,
    LANA_REACTIVE_COMPARE,
    LANA_REACTIVE_UNARY
} LanaReactiveKind;

typedef enum {
    LANA_RELATION_EXACT = 0,
    LANA_RELATION_SAME_DEPENDENCY,
    LANA_RELATION_EXPLICIT_JOINT
} LanaRelationshipKind;

typedef struct {
    uint64_t revision;
    Value *value;
} LanaReactiveVersion;

struct LanaReactive {
    uint64_t id;
    uint64_t dependency_id;
    uint64_t revision;
    LanaReactiveKind kind;
    LanaRelationshipKind relationship;
    LanaDerivationExactness exactness;
    uint32_t operation;
    LanaReactive *inputs[2];
    Value *constants[2];
    Value *current;
    LanaReactiveVersion *history;
    size_t history_count;
};

struct LanaClaim {
    Value *value;
    const char *proposition;
    LanaDerivationExactness exactness;
    double tolerance;
    bool source_valid;
};

struct LanaEffectReceipt {
    uint64_t revision;
    Value *result;
    LanaEffectReceipt *next;
};

struct LanaPlannedEffect {
    uint64_t id;
    const char *kind;
    Value *payload;
    LanaEffectReceipt *receipts;
    size_t execution_count;
};

typedef enum {
    LANA_JOINT_INDEPENDENT = 0,
    LANA_JOINT_FINITE_LAW,
    LANA_JOINT_CONDITIONAL,
    LANA_JOINT_PROJECTED
} LanaJointKind;

typedef enum {
    LANA_JOINT_CAN_PROJECT = 1u << 0,
    LANA_JOINT_CAN_CONDITION = 1u << 1,
    LANA_JOINT_CAN_SAMPLE = 1u << 2,
    LANA_JOINT_CAN_RESOLVE = 1u << 3
} LanaJointCapability;

typedef struct {
    ValueType type;
} LanaJointDomain;

typedef struct {
    Value *values;
    double weight;
} LanaJointRow;

/* An immutable named product-space law or view. */
typedef struct {
    size_t count;
    char **names;
    LanaJointDomain *domains;
    /* Independent marginals. NULL for a finite correlated law. */
    Value *values;
    size_t row_count;
    LanaJointRow *rows;
    LanaJointKind kind;
    uint32_t capabilities;
} LanaJointState;

typedef enum {
    LANA_DIST_DIRAC = 0,
    LANA_DIST_APPEND,
    LANA_DIST_TRANSFORM
} LanaStateDistKind;

struct LanaStateDist {
    LanaStateDistKind kind;
    union {
        LanaStateValue dirac;
        struct {
            LanaStateDist *left;
            LanaStateDist *right;
            bool has_cached_parameters;
            double p;
            double m_re;
            double m_im;
            double sigma;
        } append;
        struct {
            LanaStateDist *child;
            uint32_t transform_id;
        } transform;
    } as;
};

typedef struct {
    size_t count;
    size_t capacity;
    Value *items;
} LanaArray;

typedef struct {
    const char *key;
    Value *value;
} LanaMapEntry;

struct LanaMap {
    size_t count;
    size_t capacity;
    LanaMapEntry *entries;
};

struct LanaPossibility {
    size_t count;
    Value *values;
    double *weights; /* NULL means non-probabilistic, equipossible support. */
    uint64_t dependency_id;
};

typedef struct {
    bool guard;
    double weight;
    Value *result;
} LanaPathAlternative;

struct LanaPathSet {
    size_t count;
    LanaPathAlternative *alternatives;
    uint64_t dependency_id;
};

struct Value {
    ValueType type;
    LanaDerivation *derivation;
    LanaReactive *reactive;
    LanaClaim *claim;
    LanaPlannedEffect *planned_effect;
    union {
        double number;
        bool boolean;
        const char *string;
        LanaStateValue state;
        LanaDistribution distribution;
        int sample;
        LanaJointState *joint;
        LanaArray *array;
        uint32_t function;
        LanaTask *task;
        LanaStateDist *state_dist;
        LanaMap *map;
        LanaPossibility *possibility;
        LanaPathSet *paths;
        LanaCapabilityToken *capability;
    } as;
};

Value lana_value_null(void);
Value lana_value_number(double number);
Value lana_value_bool(bool boolean);
Value lana_value_string(const char *string);
Value lana_value_state(LanaState state);
Value lana_value_distribution(double p0, double p1);
Value lana_value_sample(int sample);
Value lana_value_state_dist(LanaStateDist *distribution);
Value lana_value_map(LanaMap *map);
Value lana_value_array(LanaArray *array);
Value lana_value_possibility(LanaPossibility *possibility);
Value lana_value_paths(LanaPathSet *paths);
Value lana_value_shared_capability(LanaCapabilityToken *capability);
const char *lana_value_type_name(ValueType type);
void lana_value_print(const Value *value);

#endif
