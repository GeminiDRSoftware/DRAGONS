import sys

from recipe_system.utils.reduce_utils import buildParser, normalize_args
from recipe_system import __version__ as rs_version


def test_single_arg_params_dont_overeat():
    """
    I found that single-arg options to reduce were eating any non-fits arguments
    that weren't otherwise claimed.  These are then silently dropped.

    The test here is to add unclaimed extra arguments and see that the parser
    does not pull them into these single-arg options
    """
    parser = buildParser(rs_version)
    sys.argv = ["reduce.py", "-r", "display", "foo.fits", "extname=DQ"]
    args = parser.parse_args()
    args = normalize_args(args)
    assert(len(args.files) == 2)  # extname=DQ should have been left in the file list


def test_single_arg_params_dont_overeat_upload():
    """
    I found that single-arg options to reduce were eating any non-fits arguments
    that weren't otherwise claimed.  These are then silently dropped.

    The test here is to add unclaimed extra arguments and see that the parser
    does not pull them into these single-arg options
    """
    parser = buildParser(rs_version)
    sys.argv = ["reduce.py", "--upload", "calibs", "metrics", "science", "foo.fits", "extname=DQ"]
    args = parser.parse_args()
    args = normalize_args(args)
    assert(len(args.files) == 2)  # extname=DQ should have been left in the file list
