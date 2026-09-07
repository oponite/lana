#include "value.h"
#include "vm.h"
#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>

Value lana_value_null(void) {
    Value value = {0};
    value.type = VAL_NULL;
    return value;
}

Value lana_value_number(double number) {
    Value value = {0};
    value.type = VAL_NUMBER;
    value.as.number = number;
    return value;
}

Value lana_value_bool(bool boolean) {
    Value value = {0};
    value.type = VAL_BOOL;
    value.as.boolean = boolean;
    return value;
}

Value lana_value_string(const char *string) {
    Value value = {0};
    value.type = VAL_STRING;
    value.as.string = string;
    return value;
}

Value lana_value_state(LanaState state) {
    Value value = {0};
    value.type = VAL_STATE;
    value.as.state.state = state;
    return value;
}

Value lana_value_distribution(double p0, double p1) {
    Value value = {0};
    value.type = VAL_DISTRIBUTION;
    value.as.distribution.p0 = p0;
    value.as.distribution.p1 = p1;
    return value;
}

Value lana_value_sample(int sample) {
    Value value = {0};
    value.type = VAL_SAMPLE;
    value.as.sample = sample;
    return value;
}

Value lana_value_state_dist(LanaStateDist *distribution) {
    Value value = {0};
    value.type = VAL_STATE_DIST;
    value.as.state_dist = distribution;
    return value;
}

Value lana_value_map(LanaMap *map) {
    Value value = {0};
    value.type = VAL_MAP;
    value.as.map = map;
    return value;
}

Value lana_value_array(LanaArray *array) {
    Value value = {0};
    value.type = VAL_ARRAY;
    value.as.array = array;
    return value;
}

Value lana_value_function(uint32_t function) {
    Value value = {0};
    value.type = VAL_FUNCTION;
    value.as.function = function;
    return value;
}

Value lana_value_possibility(LanaPossibility *possibility) {
    Value value = {0};
    value.type = VAL_POSSIBILITY;
    value.as.possibility = possibility;
    return value;
}

Value lana_value_paths(LanaPathSet *paths) {
    Value value = {0};
    value.type = VAL_PATH_SET;
    value.as.paths = paths;
    return value;
}

Value lana_value_shared_capability(LanaCapabilityToken *capability) {
    Value value = lana_value_null();
    value.type = VAL_SHARED_CAPABILITY;
    value.as.capability = capability;
    return value;
}

Value lana_value_tensor(struct LanaTensor *tensor) {
    Value value = {0};
    value.type = VAL_TENSOR;
    value.as.tensor = tensor;
    return value;
}

Value lana_value_nqubit_state(struct LanaTensor *tensor) {
    Value value = {0};
    value.type = VAL_NQUBIT_STATE;
    value.as.tensor = tensor;
    return value;
}

Value lana_value_povm(struct LanaTensor *tensor) {
    Value value = {0};
    value.type = VAL_POVM;
    value.as.tensor = tensor;
    return value;
}

Value lana_value_channel(struct LanaTensor *tensor) {
    Value value = {0};
    value.type = VAL_CHANNEL;
    value.as.tensor = tensor;
    return value;
}

Value lana_value_observable(struct LanaTensor *tensor) {
    Value value = {0};
    value.type = VAL_OBSERVABLE;
    value.as.tensor = tensor;
    return value;
}

Value lana_value_adt(LanaAdt *adt) {
    Value value = lana_value_null();
    value.type = VAL_ADT;
    value.as.adt = adt;
    return value;
}

Value lana_value_lazy(LanaLazy lazy) {
    Value value = lana_value_null();
    value.type = VAL_LAZY;
    value.as.lazy = lazy;
    return value;
}

Value lana_value_generator(LanaGenerator *generator) {
    Value value = lana_value_null();
    value.type = VAL_GENERATOR;
    value.as.generator = generator;
    return value;
}

Value lana_value_future(LanaFuture *future) {
    Value value = lana_value_null();
    value.type = VAL_FUTURE;
    value.as.future = future;
    return value;
}

Value lana_value_set(LanaSet *set) {
    Value value = lana_value_null();
    value.type = VAL_SET;
    value.as.set = set;
    return value;
}

Value lana_value_regex(LanaRegex *regex) {
    Value value = lana_value_null();
    value.type = VAL_REGEX;
    value.as.regex = regex;
    return value;
}

Value lana_value_optimizer(LanaOptimizer *optimizer) {
    Value value = lana_value_null();
    value.type = VAL_OPTIMIZER;
    value.as.optimizer = optimizer;
    return value;
}

Value lana_value_training_result(LanaTrainingResult *result) {
    Value value = lana_value_null();
    value.type = VAL_TRAINING_RESULT;
    value.as.training_result = result;
    return value;
}

Value lana_value_inference_algorithm(LanaInferenceAlgorithm *algorithm) {
    Value value = lana_value_null();
    value.type = VAL_INFERENCE_ALGORITHM;
    value.as.inference_algorithm = algorithm;
    return value;
}

Value lana_value_dataset(LanaDataset *dataset) {
    Value value;
    value.type = VAL_DATASET;
    value.derivation = NULL;
    value.reactive = NULL;
    value.claim = NULL;
    value.planned_effect = NULL;
    value.as.dataset = dataset;
    return value;
}

Value lana_value_posterior(LanaPosterior *posterior) {
    Value value = lana_value_null();
    value.type = VAL_POSTERIOR;
    value.as.posterior = posterior;
    return value;
}

const char *lana_value_type_name(ValueType type) {
    static const char *names[] = {
        "null", "number", "bool", "string", "state", "distribution",
        "sample", "joint_state", "array", "function", "task",
        "state_dist", "map", "possibility", "paths", "shared_capability", "adt",
        "tensor", "nqubit_state", "povm", "channel", "observable",
        "lazy", "generator", "future", "set", "regex", "optimizer", "training_result",
        "inference_algorithm", "posterior", "dataset"
    };
    if ((size_t)type >= sizeof(names) / sizeof(names[0])) {
        return "unknown";
    }
    return names[type];
}

static void tensor_print_rec(const LanaTensor *t, size_t dim, size_t offset) {
    size_t i;
    if (dim == t->ndim) {
        if (t->is_complex)
            (void)printf("[%.12g, %.12g]", tensor_get_real(t, offset), tensor_get_imag(t, offset));
        else
            (void)printf("%.12g", tensor_get_real(t, offset));
        return;
    }
    (void)printf("[");
    for (i = 0; i < t->shape[dim]; ++i) {
        if (i > 0) (void)printf(", ");
        tensor_print_rec(t, dim + 1, offset + i * t->strides[dim]);
    }
    (void)printf("]");
}

void lana_value_print(const Value *value) {
    size_t index;
    if (value == NULL) {
        (void)printf("null");
        return;
    }
    switch (value->type) {
        case VAL_NULL: (void)printf("null"); break;
        case VAL_NUMBER: (void)printf("%.12g", value->as.number); break;
        case VAL_BOOL: (void)printf("%s", value->as.boolean ? "true" : "false"); break;
        case VAL_STRING: (void)printf("%s", value->as.string); break;
        case VAL_STATE:
            (void)printf("state(p=%.12g, d_re=%.12g, d_im=%.12g)",
                         value->as.state.state.p, value->as.state.state.d_re,
                         value->as.state.state.d_im);
            break;
        case VAL_DISTRIBUTION:
            (void)printf("distribution(p0=%.12g, p1=%.12g)",
                         value->as.distribution.p0, value->as.distribution.p1);
            break;
        case VAL_SAMPLE: (void)printf("%d", value->as.sample); break;
        case VAL_JOINT_STATE:
            (void)printf("joint_state{");
            if (value->as.joint != NULL) {
                for (index = 0; index < value->as.joint->count; ++index) {
                    if (index > 0) (void)printf(", ");
                    (void)printf("%s: ", value->as.joint->names[index]);
                    if (value->as.joint->values != NULL)
                        lana_value_print(&value->as.joint->values[index]);
                    else
                        (void)printf("<finite-law>");
                }
            }
            (void)printf("}");
            break;
        case VAL_ARRAY:
            (void)printf("[");
            for (index = 0; index < value->as.array->count; ++index) {
                if (index > 0) (void)printf(", ");
                lana_value_print(&value->as.array->items[index]);
            }
            (void)printf("]");
            break;
        case VAL_FUNCTION: (void)printf("function(%u)", value->as.function); break;
        case VAL_TASK:
            (void)printf("task(%llu)",
                         (unsigned long long)(value->as.task == NULL ? 0u : value->as.task->id));
            break;
        case VAL_STATE_DIST: (void)printf("state_dist"); break;
        case VAL_MAP:
            (void)printf("{");
            for (index = 0; index < value->as.map->count; ++index) {
                if (index > 0) (void)printf(", ");
                (void)printf("\"%s\": ", value->as.map->entries[index].key);
                lana_value_print(value->as.map->entries[index].value);
            }
            (void)printf("}");
            break;
        case VAL_POSSIBILITY:
            (void)printf("possibility{");
            for (index = 0; index < value->as.possibility->count; ++index) {
                if (index > 0) (void)printf(", ");
                lana_value_print(&value->as.possibility->values[index]);
            }
            (void)printf("}");
            break;
        case VAL_PATH_SET:
            (void)printf("paths{");
            for (index = 0; index < value->as.paths->count; ++index) {
                if (index > 0) (void)printf(", ");
                (void)printf("%s => ", value->as.paths->alternatives[index].guard ? "true" : "false");
                lana_value_print(value->as.paths->alternatives[index].result);
            }
            (void)printf("}");
            break;
        case VAL_SHARED_CAPABILITY:
            (void)printf("shared_capability");
            break;
        case VAL_TENSOR:
        case VAL_NQUBIT_STATE:
        case VAL_POVM:
        case VAL_CHANNEL:
        case VAL_OBSERVABLE:
            if (value->as.tensor != NULL && value->as.tensor->data != NULL)
                tensor_print_rec(value->as.tensor, 0u, value->as.tensor->offset);
            else
                (void)printf("tensor(<empty>)");
            break;
        case VAL_ADT:
            (void)printf("adt(variant=%u){", value->as.adt == NULL ? 0u : value->as.adt->variant);
            if (value->as.adt != NULL) {
                for (index = 0; index < value->as.adt->field_count; ++index) {
                    if (index > 0) (void)printf(", ");
                    lana_value_print(&value->as.adt->fields[index]);
                }
            }
            (void)printf("}");
            break;
        case VAL_LAZY:
            (void)printf("lazy(function=%u, bound=%zu)",
                         value->as.lazy.function, value->as.lazy.bound);
            break;
        case VAL_GENERATOR:
            (void)printf("generator(function=%u, exhausted=%s)",
                         value->as.generator->function,
                         value->as.generator->exhausted ? "true" : "false");
            break;
        case VAL_FUTURE:
            (void)printf("future(function=%u, exhausted=%s, ready=%s)",
                         value->as.future->function,
                         value->as.future->exhausted ? "true" : "false",
                         value->as.future->ready ? "true" : "false");
            break;
        case VAL_SET:
            (void)printf("set{");
            for (index = 0; index < value->as.set->count; ++index) {
                if (index > 0) (void)printf(", ");
                lana_value_print(&value->as.set->items[index]);
            }
            (void)printf("}");
            break;
        case VAL_REGEX:
            (void)printf("regex(insts=%zu)", value->as.regex == NULL ? 0u : value->as.regex->inst_count);
            break;
        case VAL_OPTIMIZER:
            if (value->as.optimizer != NULL)
                (void)printf("optimizer(name=%s, learning_rate=%.12g, momentum=%.12g, beta1=%.12g, beta2=%.12g, epsilon=%.12g)",
                             value->as.optimizer->name,
                             value->as.optimizer->learning_rate,
                             value->as.optimizer->momentum,
                             value->as.optimizer->beta1,
                             value->as.optimizer->beta2,
                             value->as.optimizer->epsilon);
            else
                (void)printf("optimizer(<empty>)");
            break;
        case VAL_TRAINING_RESULT:
            (void)printf("training_result(steps=%zu)",
                         value->as.training_result == NULL ? 0u :
                         value->as.training_result->steps == NULL ? 0u :
                         value->as.training_result->steps->count);
            break;
        case VAL_INFERENCE_ALGORITHM:
            if (value->as.inference_algorithm != NULL)
                (void)printf("inference_algorithm(name=%s, family=%s, samples=%.12g, burn_in=%.12g, iterations=%.12g)",
                             value->as.inference_algorithm->name,
                             value->as.inference_algorithm->family == NULL ? "" :
                             value->as.inference_algorithm->family,
                             value->as.inference_algorithm->samples,
                             value->as.inference_algorithm->burn_in,
                             value->as.inference_algorithm->iterations);
            else
                (void)printf("inference_algorithm(<empty>)");
            break;
        case VAL_POSTERIOR:
            (void)printf("posterior(steps=%zu)",
                         value->as.posterior == NULL ? 0u :
                         value->as.posterior->steps == NULL ? 0u :
                         value->as.posterior->steps->count);
            break;
        case VAL_DATASET:
            (void)printf("dataset(op=%d)", value->as.dataset == NULL ? -1 :
                         (int)value->as.dataset->op);
            break;
        default: (void)printf("<invalid>"); break;
    }
}

void lana_value_free(Value value) {
    size_t index;
    if (value.type == VAL_STRING) {
        free((void *)value.as.string);
    } else if (value.type == VAL_ARRAY && value.as.array != NULL) {
        for (index = 0u; index < value.as.array->count; ++index)
            lana_value_free(value.as.array->items[index]);
        free(value.as.array->items);
        free(value.as.array);
    } else if ((value.type == VAL_TENSOR || value.type == VAL_NQUBIT_STATE ||
                value.type == VAL_POVM || value.type == VAL_CHANNEL ||
                value.type == VAL_OBSERVABLE) && value.as.tensor != NULL) {
        // Free shape, strides, data, then tensor struct. A view shares its
        // buffer with the source tensor and owns neither the buffer nor the
        // base, so only base tensors free data here.
        if (value.as.tensor->shape != NULL) free((void *)value.as.tensor->shape);
        if (value.as.tensor->strides != NULL) free((void *)value.as.tensor->strides);
        if (value.as.tensor->base == NULL && value.as.tensor->data != NULL)
            free((void *)value.as.tensor->data);
        free(value.as.tensor);
    } else if (value.type == VAL_MAP && value.as.map != NULL) {
        for (index = 0u; index < value.as.map->count; ++index) {
            free((void *)value.as.map->entries[index].key);
            if (value.as.map->entries[index].value != NULL) {
                lana_value_free(*value.as.map->entries[index].value);
                free(value.as.map->entries[index].value);
            }
        }
        free(value.as.map->entries);
        free(value.as.map);
    } else if (value.type == VAL_SET && value.as.set != NULL) {
        for (index = 0u; index < value.as.set->count; ++index)
            lana_value_free(value.as.set->items[index]);
        free(value.as.set->items);
        free(value.as.set);
    } else if (value.type == VAL_DATASET && value.as.dataset != NULL) {
        lana_value_free(value.as.dataset->source);
        lana_value_free(value.as.dataset->columns);
        lana_value_free(value.as.dataset->key);
        lana_value_free(value.as.dataset->limit);
        lana_value_free(value.as.dataset->other);
        lana_value_free(value.as.dataset->aggregate);
        free(value.as.dataset);
    } else if (value.type == VAL_REGEX && value.as.regex != NULL) {
        free(value.as.regex->insts);
        free(value.as.regex->classes);
        free(value.as.regex);
    } else if (value.type == VAL_OPTIMIZER && value.as.optimizer != NULL) {
        free((void *)value.as.optimizer->name);
        free(value.as.optimizer);
    } else if (value.type == VAL_TRAINING_RESULT && value.as.training_result != NULL) {
        if (value.as.training_result->params != NULL) {
            if (value.as.training_result->params->shape != NULL)
                free((void *)value.as.training_result->params->shape);
            if (value.as.training_result->params->strides != NULL)
                free((void *)value.as.training_result->params->strides);
            if (value.as.training_result->params->base == NULL &&
                value.as.training_result->params->data != NULL)
                free((void *)value.as.training_result->params->data);
            free(value.as.training_result->params);
        }
        if (value.as.training_result->steps != NULL) {
            for (index = 0u; index < value.as.training_result->steps->count; ++index)
                lana_value_free(value.as.training_result->steps->items[index]);
            free(value.as.training_result->steps->items);
            free(value.as.training_result->steps);
        }
        /* LIP-014: `data` is a shallow copy of the resolved dataset (the array
         * or lazy payload is shared with the original data value), so free only
         * the Value wrapper, not its contents. */
        if (value.as.training_result->data != NULL)
            free(value.as.training_result->data);
        free(value.as.training_result);
    } else if (value.type == VAL_INFERENCE_ALGORITHM && value.as.inference_algorithm != NULL) {
        free((void *)value.as.inference_algorithm->name);
        free((void *)value.as.inference_algorithm->family);
        free(value.as.inference_algorithm);
    } else if (value.type == VAL_POSTERIOR && value.as.posterior != NULL) {
        LanaPosterior *posterior = value.as.posterior;
        if (posterior->mean != NULL) {
            if (posterior->mean->shape != NULL) free((void *)posterior->mean->shape);
            if (posterior->mean->strides != NULL) free((void *)posterior->mean->strides);
            if (posterior->mean->base == NULL && posterior->mean->data != NULL)
                free((void *)posterior->mean->data);
            free(posterior->mean);
        }
        if (posterior->variance != NULL) {
            if (posterior->variance->shape != NULL) free((void *)posterior->variance->shape);
            if (posterior->variance->strides != NULL) free((void *)posterior->variance->strides);
            if (posterior->variance->base == NULL && posterior->variance->data != NULL)
                free((void *)posterior->variance->data);
            free(posterior->variance);
        }
        if (posterior->samples != NULL) {
            if (posterior->samples->shape != NULL) free((void *)posterior->samples->shape);
            if (posterior->samples->strides != NULL) free((void *)posterior->samples->strides);
            if (posterior->samples->base == NULL && posterior->samples->data != NULL)
                free((void *)posterior->samples->data);
            free(posterior->samples);
        }
        if (posterior->steps != NULL) {
            for (index = 0u; index < posterior->steps->count; ++index)
                lana_value_free(posterior->steps->items[index]);
            free(posterior->steps->items);
            free(posterior->steps);
        }
        free(posterior);
    }
}
