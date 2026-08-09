"""Decisions keyed on the attention backend must read the configured pair.

`--attention-backend` is one of three fields: the base one and the two split
ones (`--prefill-attention-backend` / `--decode-attention-backend`). A launch
that sets only a split field leaves the base at `None`, so a decision that reads
`attention_backend` alone answers from a field the operator never set. What that
cost, before the sweep these cases guard:

  - a weight sized for the wrong dtype (gpt-oss `sinks` under trtllm_mha; that
    decision is now gone -- the weight stays bfloat16 and the trtllm backend
    upcasts at its call site, since at model-build time no config read can say
    which backend serves this runner's forwards),
  - a prefill feature switched off (chunked prefix cache),
  - a triton kernel chosen for a backend that cannot host it (`support_triton(None)`
    answers True), in mrope and in the req-to-token writer,
  - a version guard that never fires (flashinfer),
  - a deterministic-inference knob left unset (prefill truncation align).

`attention_backends()` is the shared answer: the pair with the base-field
fallback applied. The callable decisions are checked by calling them; the rest
are pinned statically, since reproducing them means building a model or a
scheduler.
"""

import ast
import unittest
from pathlib import Path

from sglang.srt.runtime_context import attention_backends, get_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import sglang

_PACKAGE_ROOT = Path(next(iter(sglang.__path__))) / "srt"

# The decisions this file is about, and which half of the pair each one needs.
# A base-only read here is the regression; the resolution pipeline and the two
# modules that own the config are exempt because "did the operator pin the base
# field?" is a real question *there*.
_PAIR_READERS = {
    "models/gpt_oss.py": "no read: sinks stay bfloat16, the trtllm backend upcasts",
    "models/inkling_common/attn.py": "the half serving the forward (mirrors hybrid dispatch)",
    "model_executor/model_runner_components/misc_utils.py": "prefill (chunked prefix cache)",
    "layers/rotary_embedding/mrope.py": "both (triton availability)",
    "mem_cache/allocation.py": "prefill (req-to-token writer); both (get_last_loc)",
    "batch_overlap/two_batch_overlap.py": "prefill (extend positions)",
    "managers/scheduler.py": "prefill (truncation align knobs)",
}


class TestSplitBackendsReachTheDecisions(CustomTestCase):
    def setUp(self):
        super().setUp()
        self._saved = get_context()._server_args

    def tearDown(self):
        if self._saved is not None:
            get_context().set_server_args(self._saved)
        super().tearDown()

    def _publish(self, **fields):
        override = get_context().override_server_args(**fields)
        override.install()
        self.addCleanup(override.restore)

    def test_the_pair_is_what_a_split_only_launch_configures(self):
        self._publish(
            attention_backend=None,
            prefill_attention_backend="triton",
            decode_attention_backend="trtllm_mha",
        )
        self.assertEqual(attention_backends(), ("triton", "trtllm_mha"))

    def test_chunked_prefix_cache_follows_the_prefill_backend(self):
        from sglang.srt.model_executor.model_runner_components.misc_utils import (
            maybe_disable_chunked_prefix_cache,
        )
        from sglang.srt.runtime_context import get_schedule

        # A prefill backend that supports the feature, configured *only* through
        # the split field: the gate must leave it on.
        self._publish(
            attention_backend=None,
            prefill_attention_backend="fa3",
            decode_attention_backend="triton",
            disable_chunked_prefix_cache=False,
        )
        maybe_disable_chunked_prefix_cache(use_mla_backend=True, is_draft_worker=False)
        self.assertFalse(get_schedule().disable_chunked_prefix_cache)

        # And an unsupported one still switches it off.
        self._publish(
            attention_backend=None,
            prefill_attention_backend="torch_native",
            decode_attention_backend="fa3",
            disable_chunked_prefix_cache=False,
        )
        maybe_disable_chunked_prefix_cache(use_mla_backend=True, is_draft_worker=False)
        self.assertTrue(get_schedule().disable_chunked_prefix_cache)

    def test_inkling_selects_the_half_serving_the_forward(self):
        from types import SimpleNamespace

        from sglang.srt.model_executor.forward_batch_info import ForwardMode
        from sglang.srt.model_executor.forward_context import (
            ForwardContext,
            forward_context,
        )
        from sglang.srt.models.inkling_common.attn import serving_attention_backend

        self._publish(
            attention_backend=None,
            prefill_attention_backend="triton",
            decode_attention_backend="fa4",
            speculative_attention_mode="prefill",
        )

        def batch(mode):
            return SimpleNamespace(forward_mode=mode)

        unstamped = SimpleNamespace(
            prefill_attention_backend_str=None, decode_attention_backend_str=None
        )
        with forward_context(ForwardContext(attn_backend=unstamped)):
            self.assertEqual(
                serving_attention_backend(batch(ForwardMode.EXTEND)), "triton"
            )
            self.assertEqual(
                serving_attention_backend(batch(ForwardMode.DECODE)), "fa4"
            )
            self.assertEqual(serving_attention_backend(batch(ForwardMode.IDLE)), "fa4")
            self.assertEqual(
                serving_attention_backend(batch(ForwardMode.TARGET_VERIFY)), "triton"
            )

        # Draft-extend routes through the hybrid dispatcher's prefill branch
        # regardless of the spec mode.
        with forward_context(ForwardContext(attn_backend=unstamped)):
            self.assertEqual(
                serving_attention_backend(batch(ForwardMode.DRAFT_EXTEND_V2)),
                "triton",
            )

        self._publish(
            attention_backend=None,
            prefill_attention_backend="triton",
            decode_attention_backend="fa4",
            speculative_attention_mode="decode",
        )
        with forward_context(ForwardContext(attn_backend=unstamped)):
            self.assertEqual(
                serving_attention_backend(batch(ForwardMode.TARGET_VERIFY)), "fa4"
            )
            self.assertEqual(
                serving_attention_backend(batch(ForwardMode.DRAFT_EXTEND_V2)),
                "triton",
            )

        # The pair the runner stamped on its backend wins over the bags: a
        # draft runner serves every phase with its own backend.
        stamped = SimpleNamespace(
            prefill_attention_backend_str="fa4", decode_attention_backend_str="fa4"
        )
        with forward_context(ForwardContext(attn_backend=stamped)):
            self.assertEqual(
                serving_attention_backend(batch(ForwardMode.EXTEND)), "fa4"
            )

    def test_the_flashinfer_version_guard_sees_a_split_launch(self):
        # The launcher runs before any publish, so it asks the record; the
        # member and the accessor answer the same pair.
        args = ServerArgs.__new__(ServerArgs)
        for name, value in (
            ("attention_backend", None),
            ("prefill_attention_backend", None),
            ("decode_attention_backend", "flashinfer"),
        ):
            object.__setattr__(args, name, value)
        self.assertIn("flashinfer", args.get_attention_backends())

    def test_support_triton_is_the_regression_being_guarded(self):
        from sglang.srt.utils.common import support_triton

        # This is why a base-only read is not merely imprecise: the unset field
        # reads as "supported".
        self.assertTrue(support_triton(None))

    def test_no_listed_decision_reads_the_base_field_alone(self):
        offenders = []
        for rel, why in _PAIR_READERS.items():
            tree = ast.parse((_PACKAGE_ROOT / rel).read_text())
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Attribute) and node.attr == "attention_backend"
                ):
                    continue
                base = node.value
                # `get_exec().kernel.attention_backend` / `x.server_args.attention_backend`
                if isinstance(base, ast.Attribute) and base.attr in (
                    "kernel",
                    "server_args",
                ):
                    offenders.append(f"{rel}:{node.lineno}: base-only read ({why})")
                elif isinstance(base, ast.Name) and base.id in ("server_args", "cfg"):
                    offenders.append(f"{rel}:{node.lineno}: base-only read ({why})")
        self.assertEqual(
            [],
            offenders,
            "these decisions must read attention_backends() (the pair with the "
            "base-field fallback), not the base field:\n" + "\n".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
