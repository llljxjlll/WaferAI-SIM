from __future__ import annotations

import unittest

from llm.frontend.wafer_frontend.policies import swizzle
from llm.frontend.wafer_frontend.policies.swizzle import chunking


class SwizzleChunkingExportsTest(unittest.TestCase):
    def test_package_exports_are_unique_and_identity_preserving(self) -> None:
        self.assertEqual(len(swizzle.__all__), len(set(swizzle.__all__)))
        for name in chunking.__all__:
            with self.subTest(name=name):
                self.assertIn(name, swizzle.__all__)
                self.assertIs(getattr(swizzle, name), getattr(chunking, name))


if __name__ == "__main__":
    unittest.main()
