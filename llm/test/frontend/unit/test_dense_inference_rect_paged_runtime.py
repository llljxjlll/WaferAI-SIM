from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.errors import SchemaError
from llm.frontend.wafer_frontend.passes.dense_inference_rect_paged_runtime import (
    _insert_record_identity,
)


class DenseInferenceRectPagedRuntimeTest(unittest.TestCase):
    def test_outer_inner_record_identity_conflict_is_rejected(self) -> None:
        records: dict[tuple[int, int, str, int], object] = {}
        streams: dict[tuple[int, int, str, int], object] = {}
        key = (0, 0, "shared_fragment_identity", 0)
        first_record, second_record = object(), object()
        first_stream, second_stream = object(), object()
        _insert_record_identity(
            records, streams, key, first_record, first_stream,
        )
        with self.assertRaisesRegex(
            SchemaError, "outer/inner fragment record identity conflicts",
        ):
            _insert_record_identity(
                records, streams, key, second_record, second_stream,
            )
        self.assertIs(records[key], first_record)
        self.assertIs(streams[key], first_stream)


if __name__ == "__main__":
    unittest.main()
