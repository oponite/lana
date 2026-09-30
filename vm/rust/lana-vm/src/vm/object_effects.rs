//! Effect masks shared by object preflight and pure object execution.
use super::*;

pub(super) fn instruction_effect(ins: &Instruction) -> Result<u32, LanaError> {
    use OpCode::*;
    Ok(match ins.opcode {
        HostCall => host_effect(ins.b),
        Observe | ObserveMap => 1,
        SampleStateDist | JointSample | InfoSample | EstimateMeasureProbability | EstimateMeasureDistribution => 2,
        Print => 4,
        SetIndex | ArraySet | HistoryConfig | ObjectNew | OoSet => 8,
        Fork | Join | JoinTimeout | JoinAll | Cancel | TaskgroupEnter | TaskgroupExit
            | Async | Await | RunAsync => 16,
        Halt | Count => return Err(LanaError::UnsupportedOperation),
        _ => 0,
    })
}

fn host_effect(id: u32) -> u32 {
    match lana_bytecode::assembler::HOST_CALL_NAMES.get(id as usize).copied() {
        Some("map_new") | Some("map_has") | Some("map_get") | Some("map_keys")
        | Some("index_get") | Some("json_parse") | Some("json_stringify") | Some("string_length")
        | Some("string_byte_at") | Some("string_slice") | Some("string_concat") | Some("number_to_string")
        | Some("array_new") | Some("string_hex") | Some("string_join") | Some("array_length")
        | Some("string_unescape") | Some("hash_update") | Some("lazy_bound") | Some("surprisal")
        | Some("tensor_alloc") | Some("tensor_zeros") | Some("tensor_ones") | Some("tensor_eye")
        | Some("tensor_dtype") | Some("tensor_shape") | Some("tensor_ndim") | Some("tensor_add")
        | Some("tensor_sub") | Some("tensor_mul") | Some("tensor_div") | Some("tensor_matmul")
        | Some("tensor_sum") | Some("tensor_mean") | Some("tensor_max") | Some("tensor_min")
        | Some("tensor") | Some("tensor_complex") | Some("set_new") | Some("set_add")
        | Some("set_contains") | Some("set_union") | Some("set_intersect") | Some("set_difference")
        | Some("floor") | Some("string_to_number") | Some("type_of") | Some("format")
        | Some("format_number") | Some("char_length") | Some("string_codepoint_slice") | Some("to_upper")
        | Some("to_lower") | Some("regex_compile") | Some("regex_match") | Some("regex_search")
        | Some("regex_replace") | Some("gpu_matmul") | Some("density_operator") | Some("povm")
        | Some("channel") | Some("observable") | Some("tensor_product") | Some("partial_trace")
        | Some("measure_with") | Some("apply_to") | Some("expect") | Some("mix")
        | Some("trace_distance") | Some("is_separable") | Some("to_state") | Some("grad")
        | Some("vjp") | Some("sgd") | Some("adam") | Some("train")
        | Some("mcmc") | Some("vi") | Some("smc") | Some("update")
        | Some("resume") | Some("state_tensor") | Some("append") | Some("measure")
        | Some("transform") | Some("policy_evaluate") | Some("tensor_cast") | Some("tensor_reshape")
        | Some("tensor_transpose") | Some("tensor_exp") | Some("tensor_log") | Some("tensor_sqrt")
        | Some("tensor_relu") | Some("tensor_softmax") | Some("tensor_logsumexp") | Some("tensor_argmax")
        | Some("tensor_compare") | Some("tensor_select") | Some("tensor_gather") | Some("cholesky_solve")
        | Some("random_uniform") | Some("random_normal") | Some("tensor_device") | Some("tensor_to_device")
        | Some("tensor_to_cpu") | Some("execution_authorize") | Some("core_entropy") | Some("core_conditional_entropy")
        | Some("core_mutual_information") | Some("core_broja") | Some("core_kernel") | Some("core_identity_kernel")
        | Some("core_compose_kernels") | Some("core_network") | Some("core_infer") | Some("core_forget_weights")
        | Some("core_assign_weights") | Some("dataset_evidence") | Some("dataset_exclusions") | Some("rules_learn")
        | Some("rules_predict") | Some("trees_fit") | Some("trees_predict") | Some("trees_explain")
        | Some("snapshot") => 0,
        Some("random") | Some("random_seed") | Some("infer") => 2,
        Some("read_text") | Some("write_text") | Some("csv_read") | Some("csv_write")
        | Some("directory_list") | Some("directory_create") | Some("path_exists") | Some("write_text_atomic")
        | Some("run_async") | Some("future_all") | Some("future_race") | Some("sleep")
        | Some("store_open") | Some("store_get") | Some("store_scan") | Some("store_current_revision")
        | Some("ledger_query") | Some("http_get") | Some("http_post") | Some("socket_connect")
        | Some("socket_send") | Some("socket_recv") | Some("socket_close") | Some("execution_execute")
        | Some("dataset_snapshot") | Some("rules_inspect") | Some("trees_load") | Some("dataset_sqlite")
        | Some("document_extract") => 4,
        Some("map_set") | Some("index_set") | Some("array_push") | Some("store_put")
        | Some("store_delete") | Some("store_commit") | Some("policy_store_decision") | Some("ledger_append")
        | Some("future_message") | Some("dataset_source") | Some("dataset_query") | Some("dataset_apply")
        | Some("rules_save") | Some("rules_add_counterexample") | Some("rules_rollback") | Some("trees_save") => 8,
        _ => 32, // Unknown and extension hosts require explicit external-call permission.
    }
}
