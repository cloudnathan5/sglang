"""The supplied-instance surface is measured on two axes, and may only shrink.

A callee that takes ``server_args`` keeps the supplied-instance contract: the
caller chose the object, so no global-read ratchet counts it. Step 12 changes
what that object *carries* — the instance stays at the user's raw input — so a
callee reading a field **resolution fills in** would start seeing the CLI default
instead of the effective value.

This pins that intersection. Each entry is one (file, field) pair where a
parameter named ``server_args`` is read for a field resolution writes; the plan
doc carries the proposed disposition per field
(``global_context/12-raw-input-config.md``, "the supplied-instance conversion
list"). New pairs fail: a new one is new step-12 work, and the moment to decide
where the value should come from is when the read is written, not during the
flip. Pairs that disappear also fail, with the entry to delete — the list is the
measurement, not a memory of one.

The written-field set is derived here rather than hardcoded: eight
representative configs are resolved and compared against the dataclass defaults,
the same matrix the context repo's audit tool uses. The read scan mirrors that
tool's three shapes — a parameter attribute, ``getattr(server_args, "literal")``,
and the parameter parked on ``self`` — because two implementations of one census
that disagree are worse than either alone.

The second axis is **already wrong today**, not after a flip. Some config is
decided *after* publish and recorded with ``get_context().override(...)`` —
elastic-EP resizing `ep_size`, a weight update rewriting `model_path` /
`load_format`, HiCache attach naming a storage backend, adaptive speculative
decoding moving `speculative_num_steps`. That write reaches the bags and never
the record, so a supplied-instance read of one of those fields answers with the
startup value from the moment the override lands. Whether that is a defect
depends on ordering — a value copied at construction, before any override, is
fine — so this axis is pinned as a measurement with the same growth guard rather
than as a list of bugs. One of them *was* a defect and is fixed at the base of
this stack: the linear-attn dispatch table rebuilt itself from the record after
the SM100 GDN prefill decision had been recorded in the bag, so a second runner's
rebuild dropped it. That choice is a per-runner stamp now and is not recorded
process-wide at all, so neither the read nor the field is on this axis.
"""

import ast
import dataclasses
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import sglang
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_PACKAGE_ROOT = Path(next(iter(sglang.__path__))) / "srt"

# The config the resolution pipeline owns; reading the in-flight record is their
# job, not a supplied-instance read.
_OWNERS = ("server_args.py", "runtime_context.py", "arg_groups/")

_MINI_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "model_type": "llama",
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 128,
    "max_position_embeddings": 2048,
}

# One config resolves only its own decisions, so the written set is a union.
_MATRIX = (
    {},
    {
        "speculative_algorithm": "EAGLE",
        "speculative_num_steps": 3,
        "speculative_eagle_topk": 1,
        "speculative_num_draft_tokens": 4,
    },
    {"dp_size": 2, "tp_size": 2, "enable_dp_attention": True},
    # DWDP resolves dp_size and enable_dp_attention *itself* -- the plain DP
    # entry above passes them in, and passed-in fields are excluded from the
    # written set, so without this entry the dp_size readers would never pin.
    {"tp_size": 2, "dwdp_size": 2},
    {"enable_hierarchical_cache": True, "hicache_ratio": 2.0},
    {"disaggregation_mode": "prefill"},
    {"tp_size": 2, "attn_cp_size": 2},
    {"enable_lora": True, "max_lora_rank": 16},
    {"kv_cache_dtype": "fp8_e4m3", "page_size": 64},
    # MIS resolves disable_radix_cache (and friends) itself; the backend is
    # passed in because the handler asserts flashinfer rather than switching.
    {"enable_mis": True, "attention_backend": "flashinfer"},
)

_PASSED = frozenset(
    {"model_path", "device", "random_seed", "tokenizer_path", "served_model_name"}
)

_EXPOSED = {
    ("configs/embedding_model_spec.py", "chunked_prefill_size"),
    ("configs/embedding_model_spec.py", "cuda_graph_config"),
    ("configs/embedding_model_spec.py", "disable_radix_cache"),
    ("configs/model_config.py", "_speculative_draft_quantization_explicitly_set"),
    ("constrained/base_grammar_backend.py", "grammar_backend"),
    ("disaggregation/decode_kvcache_offload_manager.py", "page_size"),
    ("disaggregation/encode_receiver.py", "encoder_transfer_backend"),
    ("disaggregation/encode_server.py", "dp_size"),
    ("disaggregation/encode_server.py", "encoder_transfer_backend"),
    ("disaggregation/encode_server.py", "mm_process_config"),
    (
        "distributed/device_communicators/mooncake_transfer_engine.py",
        "encoder_transfer_backend",
    ),
    ("dllm/config.py", "max_running_requests"),
    ("entrypoints/engine.py", "dp_size"),
    ("entrypoints/engine.py", "enable_dp_attention"),
    ("entrypoints/engine.py", "ep_size"),
    ("entrypoints/http_server.py", "dp_size"),
    ("eplb/eplb_manager.py", "expert_distribution_recorder_buffer_size"),
    ("eplb/expert_distribution.py", "chunked_prefill_size"),
    ("kv_canary/capacities.py", "chunked_prefill_size"),
    ("kv_canary/capacities.py", "cuda_graph_config"),
    ("kv_canary/token_oracle/install.py", "sampling_backend"),
    ("layers/attention/flashattention_backend.py", "enable_dp_attention"),
    ("layers/deep_gemm_wrapper/compile_utils.py", "chunked_prefill_size"),
    ("layers/dp_attention.py", "dp_size"),
    ("layers/dp_attention.py", "enable_dp_attention"),
    ("layers/moe/kt_ep_wrapper.py", "chunked_prefill_size"),
    ("layers/moe/utils.py", "speculative_moe_runner_backend"),
    ("lora/lora_manager.py", "cuda_graph_config"),
    ("lora/lora_manager.py", "enable_dp_attention"),
    ("managers/data_parallel_controller.py", "dp_size"),
    ("managers/data_parallel_controller.py", "enable_dp_attention"),
    (
        "managers/data_parallel_controller.py",
        "enable_dp_attention_local_control_broadcast",
    ),
    ("managers/data_parallel_controller.py", "ep_size"),
    ("managers/data_parallel_controller.py", "load_balance_method"),
    ("managers/load_snapshot.py", "dp_size"),
    ("managers/load_snapshot.py", "enable_dp_attention"),
    ("managers/load_snapshot.py", "load_balance_method"),
    ("managers/prefill_delayer.py", "enable_dp_attention"),
    ("managers/rust_server.py", "mm_process_config"),
    ("managers/scheduler.py", "dp_size"),
    ("managers/scheduler.py", "enable_dp_attention"),
    ("managers/scheduler.py", "ep_size"),
    ("managers/scheduler.py", "page_size"),
    (
        "managers/scheduler_components/new_token_ratio_tracker.py",
        "schedule_conservativeness",
    ),
    ("managers/scheduler_components/recv_skipper.py", "enable_dp_attention"),
    ("managers/tokenizer_control_mixin.py", "dp_size"),
    ("managers/tokenizer_manager.py", "disable_radix_cache"),
    ("managers/tokenizer_manager.py", "dp_size"),
    ("managers/tokenizer_manager.py", "enable_dp_attention"),
    ("managers/tokenizer_manager.py", "encoder_transfer_backend"),
    ("managers/tokenizer_manager.py", "mm_feature_transport"),
    ("mem_cache/allocation_sizing.py", "page_size"),
    ("mem_cache/kv_cache_builder.py", "chunked_prefill_size"),
    ("mem_cache/kv_cache_builder.py", "disable_radix_cache"),
    ("mem_cache/kv_cache_builder.py", "enable_dp_attention"),
    ("model_executor/model_runner.py", "page_size"),
    (
        "model_executor/model_runner_components/ngram_embedding_manager.py",
        "chunked_prefill_size",
    ),
    (
        "model_executor/runner_backend/tc_piecewise_cuda_graph_backend.py",
        "cuda_graph_config",
    ),
    ("models/sarvam_moe.py", "attention_backend"),
    ("multimodal/processors/base_processor.py", "mm_feature_transport"),
    ("multimodal/processors/base_processor.py", "mm_process_config"),
    ("ray/data_parallel_controller.py", "dp_size"),
    ("ray/data_parallel_controller.py", "enable_dp_attention"),
    ("ray/engine.py", "dp_size"),
    ("ray/engine.py", "enable_dp_attention"),
    ("speculative/dspark_components/dspark_planner.py", "max_running_requests"),
    ("speculative/dspark_components/dspark_worker_v2.py", "cuda_graph_config"),
    ("speculative/dspark_components/dspark_worker_v2.py", "disable_cuda_graph"),
    ("speculative/dspark_components/dspark_worker_v2.py", "enable_dp_attention"),
    ("speculative/eagle_worker_v2.py", "enable_dp_attention"),
    ("speculative/frozen_kv_mtp_worker_v2.py", "enable_dp_attention"),
    ("speculative/multi_layer_eagle_worker_v2.py", "enable_dp_attention"),
    ("speculative/standalone_worker_v2.py", "enable_dp_attention"),
    ("utils/common.py", "page_size"),
    ("utils/cuda_vmm_transport_utils.py", "dp_size"),
    ("utils/cuda_vmm_transport_utils.py", "enable_dp_attention"),
    ("utils/cuda_vmm_transport_utils.py", "mm_feature_transport"),
    ("utils/offloader.py", "dp_size"),
}


# Axis two: (file, field) pairs where a supplied-instance read names a field that
# some code overrides post-publish. Each needs an ordering judgment, not a blanket
# conversion; the list exists so a new one is a decision made when it is written.
_OVERRIDDEN_AND_READ = {
    ("configs/model_config.py", "dtype"),
    ("configs/model_config.py", "model_path"),
    ("constrained/base_grammar_backend.py", "grammar_backend"),
    ("disaggregation/decode_kvcache_offload_manager.py", "hicache_storage_backend"),
    (
        "disaggregation/decode_kvcache_offload_manager.py",
        "hicache_storage_backend_extra_config",
    ),
    ("disaggregation/encode_server.py", "dp_size"),
    ("disaggregation/encode_server.py", "load_format"),
    ("disaggregation/encode_server.py", "model_path"),
    (
        "distributed/device_communicators/mooncake_transfer_engine.py",
        "hicache_storage_backend",
    ),
    ("dllm/config.py", "model_path"),
    ("elastic_ep/expert_backup_manager.py", "load_format"),
    ("entrypoints/engine.py", "dp_size"),
    ("entrypoints/engine.py", "dtype"),
    ("entrypoints/engine.py", "ep_size"),
    ("entrypoints/engine.py", "load_format"),
    ("entrypoints/engine.py", "model_path"),
    ("entrypoints/http_server.py", "dp_size"),
    ("entrypoints/http_server.py", "model_path"),
    ("kv_canary/api.py", "speculative_num_steps"),
    ("kv_canary/capacities.py", "speculative_num_draft_tokens"),
    ("layers/dp_attention.py", "dp_size"),
    ("managers/data_parallel_controller.py", "dp_size"),
    ("managers/data_parallel_controller.py", "ep_size"),
    ("managers/load_snapshot.py", "dp_size"),
    ("managers/scheduler.py", "dp_size"),
    ("managers/scheduler.py", "ep_size"),
    ("managers/scheduler.py", "hicache_storage_backend"),
    ("managers/tokenizer_control_mixin.py", "dp_size"),
    ("managers/tokenizer_manager.py", "dp_size"),
    ("managers/tokenizer_manager.py", "model_path"),
    ("managers/tokenizer_manager.py", "speculative_num_draft_tokens"),
    ("managers/tp_worker.py", "model_path"),
    ("managers/utils.py", "speculative_num_steps"),
    ("mem_cache/allocation_sizing.py", "speculative_num_steps"),
    ("mem_cache/hiradix_cache.py", "hicache_storage_backend"),
    ("mem_cache/hiradix_cache.py", "hicache_storage_backend_extra_config"),
    ("mem_cache/hiradix_cache.py", "hicache_storage_prefetch_policy"),
    ("mem_cache/hiradix_cache.py", "hicache_write_policy"),
    ("mem_cache/hybrid_cache/hybrid_pool_assembler.py", "hicache_storage_backend"),
    ("mem_cache/hybrid_cache/hybrid_pool_assembler.py", "hicache_write_policy"),
    ("mem_cache/kv_cache_builder.py", "hicache_storage_backend"),
    ("mem_cache/pool_host/common.py", "hicache_storage_backend"),
    ("mem_cache/pool_host/common.py", "hicache_storage_backend_extra_config"),
    ("mem_cache/radix_cache_cpp.py", "hicache_write_policy"),
    ("mem_cache/unified_radix_cache.py", "hicache_storage_backend"),
    ("mem_cache/unified_radix_cache.py", "hicache_storage_backend_extra_config"),
    ("mem_cache/unified_radix_cache.py", "hicache_storage_prefetch_policy"),
    ("mem_cache/unified_radix_cache.py", "hicache_write_policy"),
    ("model_executor/model_runner_components/load_model_utils.py", "load_format"),
    ("parser/template_detection.py", "model_path"),
    ("ray/data_parallel_controller.py", "dp_size"),
    ("ray/engine.py", "dp_size"),
    ("speculative/dflash_worker_v2.py", "speculative_num_draft_tokens"),
    ("speculative/dspark_components/dspark_worker_v2.py", "disable_cuda_graph"),
    (
        "speculative/dspark_components/dspark_worker_v2.py",
        "speculative_num_draft_tokens",
    ),
    ("speculative/eagle_disaggregation.py", "speculative_num_steps"),
    ("speculative/eagle_worker_v2.py", "speculative_num_draft_tokens"),
    ("speculative/eagle_worker_v2.py", "speculative_num_steps"),
    ("speculative/frozen_kv_mtp_worker_v2.py", "speculative_num_draft_tokens"),
    ("speculative/frozen_kv_mtp_worker_v2.py", "speculative_num_steps"),
    ("speculative/multi_layer_eagle_worker_v2.py", "speculative_num_draft_tokens"),
    ("speculative/multi_layer_eagle_worker_v2.py", "speculative_num_steps"),
    ("speculative/ngram_worker.py", "speculative_num_draft_tokens"),
    ("speculative/ngram_worker.py", "speculative_num_steps"),
    ("speculative/spec_info.py", "speculative_num_draft_tokens"),
    ("speculative/spec_info.py", "speculative_num_steps"),
    ("speculative/spec_utils.py", "speculative_num_draft_tokens"),
    ("speculative/standalone_worker_v2.py", "speculative_num_draft_tokens"),
    ("speculative/standalone_worker_v2.py", "speculative_num_steps"),
    ("utils/common.py", "speculative_num_draft_tokens"),
    ("utils/common.py", "speculative_num_steps"),
    ("utils/cuda_vmm_transport_utils.py", "dp_size"),
    ("utils/offloader.py", "dp_size"),
}


def _expanded_override_keys(rel, tree, call, kw) -> set:
    """The statically visible keys behind an ``override(..., **expr)``.

    Handles a dict literal, a conditional between dict literals, and a name
    bound to a dict literal in the enclosing function (plus constant-subscript
    stores onto it -- the HiCache attach shape). One expansion is unresolvable
    by design and exempted by name: ``update_server_args`` forwards
    operator-chosen fields, so its key set is the API's, not this file's.
    Anything else unresolvable fails -- a silently skipped expansion would
    shrink the written set.
    """
    for a in call.args:
        if isinstance(a, ast.Constant) and a.value == "update_server_args":
            return set()
    for k in call.keywords:
        if (
            k.arg == "source"
            and isinstance(k.value, ast.Constant)
            and k.value.value == "update_server_args"
        ):
            return set()

    def dict_keys(node) -> set:
        assert isinstance(node, ast.Dict) and all(
            isinstance(key, ast.Constant) for key in node.keys
        ), f"non-literal dict in override expansion at {rel}:{call.lineno}"
        return {key.value for key in node.keys}

    if isinstance(kw.value, ast.Dict):
        return dict_keys(kw.value)
    if isinstance(kw.value, ast.IfExp):
        keys = set()
        for branch in (kw.value.body, kw.value.orelse):
            if isinstance(branch, ast.Dict) and branch.keys:
                keys |= dict_keys(branch)
            elif isinstance(branch, ast.Dict):
                pass
            else:
                raise AssertionError(
                    f"unresolvable override expansion at {rel}:{call.lineno}"
                )
        return keys
    assert isinstance(
        kw.value, ast.Name
    ), f"unresolvable override expansion at {rel}:{call.lineno}"
    name = kw.value.id
    enclosing = None
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if (
                fn.lineno
                <= call.lineno
                <= max(getattr(fn, "end_lineno", fn.lineno), fn.lineno)
            ):
                if enclosing is None or fn.lineno > enclosing.lineno:
                    enclosing = fn
    assert (
        enclosing is not None
    ), f"override expansion outside any function at {rel}:{call.lineno}"
    keys = set()
    found = False
    for node in ast.walk(enclosing):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == name
            and isinstance(node.value, ast.Dict)
        ):
            found = True
            keys |= dict_keys(node.value)
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Subscript)
            and isinstance(node.targets[0].value, ast.Name)
            and node.targets[0].value.id == name
            and isinstance(node.targets[0].slice, ast.Constant)
        ):
            keys.add(node.targets[0].slice.value)
    assert found, (
        f"override expansion '{name}' at {rel}:{call.lineno} has no "
        "dict-literal assignment in its function; extend the resolver"
    )
    return keys


class TestSuppliedInstanceExposure(CustomTestCase):
    def setUp(self):
        # Resolving the matrix writes process state on the way through (the
        # multimodal transport handler sets SGLANG_USE_CUDA_IPC_TRANSPORT, and
        # `EnvField.set()` flips a descriptor flag `os.environ` does not carry).
        # Leaking it makes *later* files in the same worker fail, which is how
        # this was found -- so the case restores what it touched.
        super().setUp()
        state = (dict(os.environ), self._env_field_flags())
        self.addCleanup(self._restore_process_state, state)

    @staticmethod
    def _env_field_flags() -> dict:
        from sglang.srt.environ import EnvField, envs

        flags = {}
        for klass in reversed(type(envs).__mro__):
            for name, field in vars(klass).items():
                if isinstance(field, EnvField):
                    flags[name] = field._set_to_none
        return flags

    @staticmethod
    def _restore_process_state(state) -> None:
        from sglang.srt.environ import envs

        saved_environ, saved_flags = state
        os.environ.clear()
        os.environ.update(saved_environ)
        for name, was_none in saved_flags.items():
            getattr(type(envs), name)._set_to_none = was_none

    def _config_dir(self) -> str:
        config_dir = tempfile.mkdtemp(prefix="supplied_instance_")
        self.addCleanup(shutil.rmtree, config_dir, ignore_errors=True)
        with open(os.path.join(config_dir, "config.json"), "w") as handle:
            json.dump(_MINI_CONFIG, handle)
        return config_dir

    def _resolution_written_fields(self) -> set:
        """The union of what resolution fills in across the matrix.

        Every entry must resolve. A silently skipped one would shrink this set,
        which makes pinned pairs look like they disappeared -- the list would
        then drift by environment rather than by code, and the failure would
        point at the wrong thing.
        """
        model_path = self._config_dir()
        written = set()
        for extra in _MATRIX:
            try:
                resolved = ServerArgs(
                    model_path=model_path, device="cuda", random_seed=42, **extra
                )
            except Exception as exc:
                self.fail(
                    f"the matrix entry {extra} did not resolve in this "
                    f"environment ({type(exc).__name__}: {exc}); the written-field "
                    "union would be short and the pinned list would drift"
                )
            for field in dataclasses.fields(resolved):
                if field.name in _PASSED or field.name in extra:
                    continue
                if field.default is dataclasses.MISSING:
                    continue
                if getattr(resolved, field.name) != field.default:
                    written.add(field.name)
        return written

    def _supplied_instance_reads(self) -> set:
        """Three spellings of the same read: ``server_args.field`` off the
        parameter, ``getattr(server_args, "field", default)`` with a literal
        name, and the *parked* form -- ``self.x = server_args`` in a method
        that takes the parameter, read as ``self.x.field`` anywhere in the
        class. Parking under a different object, a container, or a computed
        name stays invisible, like in every census of this family."""
        pairs = set()
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            rel = path.relative_to(_PACKAGE_ROOT).as_posix()
            if rel.startswith(_OWNERS):
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for fn in ast.walk(tree):
                if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                params = {a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)}
                if "server_args" not in params:
                    continue
                for node in ast.walk(fn):
                    if (
                        isinstance(node, ast.Attribute)
                        and isinstance(node.value, ast.Name)
                        and node.value.id == "server_args"
                        and isinstance(node.ctx, ast.Load)
                    ):
                        pairs.add((rel, node.attr))
                    elif (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "getattr"
                        and len(node.args) >= 2
                        and isinstance(node.args[0], ast.Name)
                        and node.args[0].id == "server_args"
                        and isinstance(node.args[1], ast.Constant)
                        and isinstance(node.args[1].value, str)
                    ):
                        # The same read in optional clothing. Only a literal
                        # name is censusable; a computed one is not.
                        pairs.add((rel, node.args[1].value))
            for cls in ast.walk(tree):
                if not isinstance(cls, ast.ClassDef):
                    continue
                # Parked: `self.x = server_args` in a method that takes the
                # parameter, read as `self.x.field` anywhere in the class.
                parked = set()
                for fn in cls.body:
                    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    if "server_args" not in {
                        a.arg for a in list(fn.args.args) + list(fn.args.kwonlyargs)
                    }:
                        continue
                    for node in ast.walk(fn):
                        if (
                            isinstance(node, ast.Assign)
                            and len(node.targets) == 1
                            and isinstance(node.targets[0], ast.Attribute)
                            and isinstance(node.targets[0].value, ast.Name)
                            and node.targets[0].value.id == "self"
                            and isinstance(node.value, ast.Name)
                            and node.value.id == "server_args"
                        ):
                            parked.add(node.targets[0].attr)
                for node in ast.walk(cls):
                    if (
                        isinstance(node, ast.Attribute)
                        and isinstance(node.ctx, ast.Load)
                        and isinstance(node.value, ast.Attribute)
                        and node.value.attr in parked
                        and isinstance(node.value.value, ast.Name)
                        and node.value.value.id == "self"
                    ):
                        pairs.add((rel, node.attr))
        return pairs

    @staticmethod
    def _override_written_fields() -> set:
        """Fields written post-publish through ``get_context().override(...)``."""
        written = set()
        for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
            rel = path.relative_to(_PACKAGE_ROOT).as_posix()
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "override"
                ):
                    continue
                base = node.func.value
                if (
                    isinstance(base, ast.Call)
                    and isinstance(base.func, ast.Name)
                    and base.func.id == "get_context"
                ):
                    for kw in node.keywords:
                        if kw.arg:
                            written.add(kw.arg)
                        else:
                            written |= _expanded_override_keys(rel, tree, node, kw)
        return written

    def test_the_post_publish_override_surface_matches_the_pinned_list(self):
        written = self._override_written_fields()
        self.assertGreater(
            len(written), 5, "found almost no override targets; the scan broke"
        )
        found = {pair for pair in self._supplied_instance_reads() if pair[1] in written}
        new = sorted(found - _OVERRIDDEN_AND_READ)
        gone = sorted(_OVERRIDDEN_AND_READ - found)
        self.assertEqual(
            ([], []),
            (new, gone),
            "the post-publish override surface drifted. A read here answers with "
            "the startup value once the override lands, so a new pair needs an "
            "ordering judgment: copied before any override (fine), or read after "
            "one (then it must come from the bags).\n"
            f"  new: {new}\n"
            f"  gone (delete from _OVERRIDDEN_AND_READ): {gone}",
        )

    def test_the_exposed_set_matches_the_pinned_list(self):
        written = self._resolution_written_fields()
        found = {pair for pair in self._supplied_instance_reads() if pair[1] in written}
        new = sorted(found - _EXPOSED)
        gone = sorted(_EXPOSED - found)
        self.assertEqual(
            ([], []),
            (new, gone),
            "the supplied-instance step-12 surface drifted.\n"
            f"  new (decide where the resolved value comes from): {new}\n"
            f"  gone (delete from _EXPOSED): {gone}",
        )


if __name__ == "__main__":
    unittest.main()
