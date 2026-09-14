"""Standard-library entry points for function-style exp4 tests."""

from pathlib import Path
import tempfile
import unittest

import test_exp2_mapping as mapping
import test_pareto_plot as pareto


class Exp4FunctionTests(unittest.TestCase):
    def test_exp2_mapping_contracts(self):
        mapping.test_exact_36_primitive_inventory_and_field_mapping()
        mapping.test_stable_main_pipeline_api()
        mapping.test_sw_opt_only_is_exact_exp2_row_join()
        mapping.test_source_inventory_is_repeatable_and_inherits_publish_status()
        mapping.test_mapping_is_deterministic_unique_and_groups_are_connected()
        mapping.test_6x6_mapping_preserves_exp2_row_major_modules()
        mapping.test_small_topology_has_explicit_infeasible_status()

    def test_pareto_contracts(self):
        pareto.test_aggregation_excludes_infeasible_and_computes_pareto_metrics()
        pareto.test_average_is_after_per_model_normalization_and_argmax_switches()
        pareto.test_speedup_common_denominator_and_projection_marker()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            pareto.test_svg_renderers_emit_both_figures(path)
            pareto.test_load_rows_accepts_wrapped_json(path)


if __name__ == "__main__":
    unittest.main()
