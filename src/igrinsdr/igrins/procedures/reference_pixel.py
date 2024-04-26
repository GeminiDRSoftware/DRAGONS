from ..igrins_pipeline.procedures.readout_pattern_guard import (
    remove_pattern_from_guard)


def fix_pattern_using_reference_pixel(d):
    return remove_pattern_from_guard(d)
