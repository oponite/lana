#ifndef LANA_VALUE_H
#define LANA_VALUE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "state.h"

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
    VAL_SHARED_CAPABILITY,
    VAL_ADT,
    VAL_TENSOR,   // First‑class tensor value (LIP‑004)
    VAL_NQUBIT_STATE,  // N‑qubit density operator (LIP‑005 §1)
    VAL_POVM,          // finite set of PSD operators summing to I (LIP‑005 §3)
    VAL_CHANNEL,       // CPTP map in Kraus form (LIP‑005 §4)
    VAL_OBSERVABLE,    // Hermitian operator (LIP‑005 §6)
    VAL_LAZY,
    VAL_GENERATOR,
    VAL_FUTURE,
    VAL_SET,
    VAL_REGEX,
    VAL_OPTIMIZER,        // LIP-006 optimizer descriptor (name + hyperparameters)
    VAL_TRAINING_RESULT,  // LIP-006 trained parameters + step history
    VAL_INFERENCE_ALGORITHM,  // LIP-009 inference algorithm descriptor
    VAL_POSTERIOR,            // LIP-009 posterior distribution over parameters
    VAL_DATASET               // LIP-015 lazy relational-algebra plan node
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
typedef struct LanaAdt LanaAdt;
typedef struct LanaLazy LanaLazy;
typedef struct LanaGenerator LanaGenerator;
typedef struct LanaFuture LanaFuture;
typedef struct LanaSet LanaSet;
typedef struct LanaRegex LanaRegex;
typedef struct LanaOptimizer LanaOptimizer;
typedef struct LanaTrainingResult LanaTrainingResult;
typedef struct LanaInferenceAlgorithm LanaInferenceAlgorithm;
typedef struct LanaPosterior LanaPosterior;
typedef struct LanaDataset LanaDataset;

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

/* The evidence status of a derivation, ordered least to most certain. It is a
 * derived label over (kind, exactness, outcome), not a stored field. */
typedef enum {
    LANA_EVIDENCE_UNKNOWN = 0,
    LANA_EVIDENCE_SAMPLED,
    LANA_EVIDENCE_MODELED,
    LANA_EVIDENCE_EXACT,
    LANA_EVIDENCE_OBSERVED
} LanaEvidenceStatus;

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
    /* LIP-011 reverse-mode autodiff fields. `ad_op` is -1 for a non-
     * differentiable node (including the input leaf); otherwise it is the
     * differentiable primitive: 0=add 1=sub 2=mul 3=div 4=matmul 5=sum
     * 6=mean. `ad_a`/`ad_b` are the saved forward-pass operands (the input
     * tensor for reductions, NULL for the absent right operand). The input
     * derivations are stored separately from `inputs[]` so a constant operand
     * (no derivation) still maps to its saved tensor. `ad_grad` is the
     * accumulated gradient tensor, allocated lazily by the backward pass. */
    int ad_op;
    struct LanaTensor *ad_a;
    struct LanaTensor *ad_b;
    LanaDerivation *ad_a_deriv;
    LanaDerivation *ad_b_deriv;
    struct LanaTensor *ad_grad;
    int ad_axis; /* reduction axis, -1 for a full reduction */
};

typedef enum {
    LANA_REACTIVE_ROOT = 0,
    LANA_REACTIVE_BINARY,
    LANA_REACTIVE_COMPARE,
    LANA_REACTIVE_UNARY,
    LANA_REACTIVE_TRAIN
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
    /* LIP-010: a training data root accepts a single [x, target] observation
     * on `observe` (its support is structural, not a membership test). */
    bool is_training_data;
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
    LANA_DIST_TRANSFORM,
    LANA_DIST_ATTENUATE
} LanaStateDistKind;

#define LANA_STATE_DIST_DEPTH_LIMIT 1024u

typedef struct {
    bool is_inline;
    union {
        LanaStateValue state;
        LanaStateDist *node;
    } as;
} LanaDistOperand;

struct LanaStateDist {
    LanaStateDistKind kind;
    union {
        LanaStateValue dirac;
        struct {
            LanaDistOperand left;
            LanaDistOperand right;
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
        struct {
            LanaStateDist *child;
            double factor;
        } attenuate;
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

/* An algebraic data type value: a variant tag plus its fields. The reserved
 * variant 0xFFFFFFFF is the built-in `unknown` value available to every ADT. */
struct LanaAdt {
    uint32_t variant;
    Value *fields;
    size_t field_count;
};

/* A lazy bounded dataset: a generator function (a VAL_FUNCTION index) plus a
 * materialization bound. The dataset occupies constant space until forced. */
struct LanaLazy {
    uint32_t function;
    size_t bound;
};

/* A suspended generator frame (LIP-022 §2): the function it runs, the saved
 * instruction pointer, a snapshot of its registers, and whether it has run to
 * completion. `registers` is GC-traced like a live frame's register array. */
struct LanaGenerator {
    uint32_t function;
    size_t ip;
    Value *registers;
    size_t register_count;
    bool exhausted;
};

/* A suspended async frame (LIP-024 §5): the async function it runs, the saved
 * instruction pointer, a snapshot of its registers, whether it has run to
 * completion, and whether it is runnable (on the event loop's ready queue).
 * `registers` is GC-traced like a live frame's register array. Composite
 * futures (future_all/future_race/sleep) carry their inputs and wake time;
 * for a composite future `function` is UINT32_MAX and `registers` is a
 * single-element array holding the result once exhausted. */
struct LanaFuture {
    uint32_t function;       /* async function index, or UINT32_MAX for composite */
    size_t ip;
    Value *registers;
    size_t register_count;
    bool exhausted;
    bool ready;              /* true while on the event loop's ready queue */
    bool is_composite;
    uint32_t composite_kind; /* LANA_FUTURE_ALL / LANA_FUTURE_RACE / LANA_FUTURE_SLEEP */
    LanaFuture **inputs;     /* input futures (future_all/future_race) */
    size_t input_count;
    double wake_time;        /* sleep: monotonic completion time (seconds) */
};

typedef enum {
    LANA_FUTURE_ALL = 0,
    LANA_FUTURE_RACE,
    LANA_FUTURE_SLEEP
} LanaFutureKind;

/* An immutable set of ordinary values (LIP-022 §1). Membership is linear over
 * `items`, mirroring `LanaMap`; there is no hash function. `STATE`,
 * `STATE_DIST`, and `Information` are not set members. */
struct LanaSet {
    size_t count;
    size_t capacity;
    Value *items;
};

/* A compiled regular expression (LIP-021 §2): a Thompson NFA program plus its
 * character classes. No `Value` references inside, so GC tracing marks the
 * struct and its arrays as opaque leaves. */
typedef enum {
    LANA_REGEX_CHAR = 0,  /* match literal byte `c`, advance pc+1 */
    LANA_REGEX_ANY,       /* match any byte except '\n', advance pc+1 */
    LANA_REGEX_CLASS,     /* match if byte in class `c`, advance pc+1 */
    LANA_REGEX_BOL,       /* match if at position 0 (zero-width), advance pc+1 */
    LANA_REGEX_EOL,       /* match if at position len (zero-width), advance pc+1 */
    LANA_REGEX_SPLIT,     /* epsilon to `x` and `y` */
    LANA_REGEX_JMP,       /* epsilon to `x` */
    LANA_REGEX_MATCH      /* accept */
} LanaRegexOp;

typedef struct {
    LanaRegexOp op;
    uint32_t c;  /* CHAR: byte; CLASS: class index; SPLIT/JMP: target */
    uint32_t x;  /* SPLIT: first target */
    uint32_t y;  /* SPLIT: second target */
} LanaRegexInst;

typedef struct {
    uint32_t bitmap[8];  /* 256-bit membership set */
    bool negated;
} LanaRegexClass;

struct LanaRegex {
    LanaRegexInst *insts;
    size_t inst_count;
    LanaRegexClass *classes;
    size_t class_count;
};

/* LIP-006 optimizer descriptor: a named optimizer plus its hyperparameters.
 * `name` is "sgd" or "adam". For SGD, `learning_rate` and `momentum` are used
 * and the Adam fields are zero; for Adam, `learning_rate`, `beta1`, `beta2`,
 * and `epsilon` are used and `momentum` is zero. */
struct LanaOptimizer {
    const char *name;
    double learning_rate;
    double momentum;
    double beta1;
    double beta2;
    double epsilon;
};

/* LIP-006 training result: the trained parameters plus the per-step history.
 * `params` is the final parameter tensor; `steps` is an array of step maps,
 * each carrying the post-update params (with provenance), the gradient, and
 * the optimizer state. LIP-010 adds the training configuration so an
 * incremental `update` (or a reactive recomputation) can resume the optimizer
 * from the last step map: `model_function`/`loss_function` are arity-2
 * function indices and `optimizer` is the optimizer descriptor. LIP-014 adds
 * the resolved dataset and the effective batch size so `resume` can continue
 * the run from any step: `data` is the (non-reactive) dataset — an array of
 * [x, target] pairs or a lazy dataset — and `batch_size` is the effective
 * batch size (0-normalized to the full dataset at `train` time). */
struct LanaTrainingResult {
    struct LanaTensor *params;
    LanaArray *steps;
    uint32_t model_function;
    uint32_t loss_function;
    LanaOptimizer *optimizer;
    Value *data;
    size_t batch_size;
};

/* LIP-009 inference algorithm descriptor: a named algorithm plus its
 * hyperparameters. `name` is "mcmc", "vi", or "smc". For MCMC, `samples` and
 * `burn_in` are used; for VI, `family` ("gaussian" or "mean_field") and
 * `iterations` are used; for SMC, `samples` is the particle count. Unused
 * fields are zero (or NULL for `family`). */
struct LanaInferenceAlgorithm {
    const char *name;
    const char *family;
    double samples;
    double burn_in;
    double iterations;
};

/* LIP-009 posterior: a distribution over parameters produced by `infer`.
 * `mean` is the point estimate (posterior mean); `variance` is the per-element
 * uncertainty (posterior variance); `samples` is the sample matrix
 * [n_samples, ...param_shape] for sample-based algorithms (NULL for VI);
 * `steps` is the per-step provenance (an array of step maps); `seed` is the
 * RNG seed the run used, for replay. */
struct LanaPosterior {
    struct LanaTensor *mean;
    struct LanaTensor *variance;
    struct LanaTensor *samples;
    LanaArray *steps;
    uint64_t seed;
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
        LanaAdt *adt;
        struct LanaTensor *tensor;
        LanaLazy lazy;
        LanaGenerator *generator;
        LanaFuture *future;
        LanaSet *set;
        LanaRegex *regex;
        LanaOptimizer *optimizer;
        LanaTrainingResult *training_result;
        LanaInferenceAlgorithm *inference_algorithm;
        LanaPosterior *posterior;
        LanaDataset *dataset;
    } as;
};

/* LIP-015: a lazy relational-algebra plan node. Each operation builds a new
 * node over a source (a VAL_LAZY or another VAL_DATASET); only `materialize`
 * forces evaluation. `function` is the predicate/transform function index for
 * FILTER/MAP. `columns` is an array of column-name strings for SELECT.
 * `key` is the key string for SORT/GROUP_BY/JOIN. `limit` is the cap for
 * LIMIT. `other` is the right-hand dataset for JOIN. `aggregate` is the
 * aggregate descriptor (an array like ["sum","v"] or ["count"]) for
 * AGGREGATE. All Value fields are GC-traced. */
typedef enum {
    LANA_DATASET_SOURCE,   /* wraps a lazy source */
    LANA_DATASET_FILTER,
    LANA_DATASET_MAP,
    LANA_DATASET_SELECT,
    LANA_DATASET_LIMIT,
    LANA_DATASET_SORT,
    LANA_DATASET_GROUP_BY,
    LANA_DATASET_AGGREGATE,
    LANA_DATASET_JOIN
} LanaDatasetOp;

struct LanaDataset {
    LanaDatasetOp op;
    Value source;
    uint32_t function;
    Value columns;
    Value key;
    Value limit;
    Value other;
    Value aggregate;
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
Value lana_value_function(uint32_t function);
Value lana_value_possibility(LanaPossibility *possibility);
Value lana_value_paths(LanaPathSet *paths);
Value lana_value_shared_capability(LanaCapabilityToken *capability);
Value lana_value_adt(LanaAdt *adt);
Value lana_value_lazy(LanaLazy lazy);
Value lana_value_generator(LanaGenerator *generator);
Value lana_value_future(LanaFuture *future);
Value lana_value_set(LanaSet *set);
Value lana_value_regex(LanaRegex *regex);
Value lana_value_optimizer(LanaOptimizer *optimizer);
Value lana_value_training_result(LanaTrainingResult *result);
Value lana_value_inference_algorithm(LanaInferenceAlgorithm *algorithm);
Value lana_value_posterior(LanaPosterior *posterior);
Value lana_value_dataset(LanaDataset *dataset);
Value lana_value_tensor(struct LanaTensor *tensor);
Value lana_value_nqubit_state(struct LanaTensor *tensor);
Value lana_value_povm(struct LanaTensor *tensor);
Value lana_value_channel(struct LanaTensor *tensor);
Value lana_value_observable(struct LanaTensor *tensor);
const char *lana_value_type_name(ValueType type);
void lana_value_print(const Value *value);
void lana_value_free(Value value);

#endif
