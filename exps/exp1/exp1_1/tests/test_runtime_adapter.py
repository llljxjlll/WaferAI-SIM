from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import runtime_adapter as adapter


@dataclass(frozen=True)
class _Policy:
    inter_die: object = None
    intra_die: object = None


@dataclass(frozen=True)
class _Spec:
    policy: _Policy = _Policy()


class RuntimeAdapterTest(unittest.TestCase):
    def test_point_pads_before_tp_partition(self) -> None:
        case = SimpleNamespace(
            case_id="pad",
            operator="AG_GEMM",
            logical_mnk=(2048, 11008, 4096),
            mesh=SimpleNamespace(dies=6, rows=2, columns=3),
        )
        point = adapter._point(case)
        self.assertEqual((point.tokens, point.intermediate_size, point.hidden_size),
                         (2052, 5508, 4128))
        self.assertEqual(point.tokens % 6, 0)
        self.assertEqual(point.intermediate_size % 6, 0)
        self.assertEqual(point.hidden_size % 6, 0)

    def test_swizzle_uses_common_ir2_16_core_compile(self) -> None:
        linked_profile = SimpleNamespace(manifest=SimpleNamespace())
        compilation = SimpleNamespace(
            linked=SimpleNamespace(entries=(linked_profile,))
        )
        context = SimpleNamespace(fabric="fabric", hbm_address_spaces=("hbm",))
        with patch.object(adapter, "compile_naive", return_value=compilation) as compile_mock:
            source, algorithm = adapter._linked_source(
                object(), object(), "swizzle", spec=_Spec(), context=context
            )
        self.assertIs(source, linked_profile)
        self.assertEqual(algorithm, "wang_1d_bidirectional")
        kwargs = compile_mock.call_args.kwargs
        self.assertEqual(kwargs["hbm_address_spaces"], ("hbm",))
        options = kwargs["intra_die_refine_options"]
        self.assertEqual(options.split_k_parts, 16)
        self.assertEqual(options.compute_groups_per_die, 16)
        self.assertTrue(options.enable_reduce)
        self.assertFalse(options.enable_direct_dma)


if __name__ == "__main__":
    unittest.main()
