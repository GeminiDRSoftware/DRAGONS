import subprocess
import sys
import textwrap

import pytest


# The configuration is a singleton that reads the user's config file once.
# Importing the interactive modules must therefore not instantiate it before
# the config file has been loaded.  This needs a fresh interpreter since the
# modules and the singleton are cached for the lifetime of a process.
SCRIPT = textwrap.dedent("""
    import sys

    # Import the modules *before* the configuration is loaded, as happens
    # when running "reduce".
    from geminidr.interactive.fit import fit1d, telluric
    from recipe_system.config import load_config
    load_config([sys.argv[1]])

    from geminidr.interactive.interactive_config import interactive_conf

    ic = interactive_conf()
    assert ic.browser == "chrome", ic.browser
    assert ic.port_number == 5123, ic.port_number
    assert ic.bokeh_data_color == "green", ic.bokeh_data_color

    # The glyph colors must follow the configuration, too.  Glyphs do not
    # depend on the instance state so we can call the properties directly.
    assert fit1d.InteractiveModel.Good.fget(None).palette == "green"
    assert telluric.InteractiveTelluricCorrection.Good.fget(None).palette == "green"
""")


def test_interactive_config_loaded_after_import(tmp_path):
    rcfile = tmp_path / "dragonsrc"
    rcfile.write_text("[interactive]\n"
                      "browser = chrome\n"
                      "port_number = 5123\n"
                      "line_color = green\n")

    result = subprocess.run([sys.executable, "-c", SCRIPT, str(rcfile)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# Same as above, but with something asking for the configuration before the
# config file is loaded.  The user's settings must still be picked up.
SCRIPT_EARLY_REQUEST = textwrap.dedent("""
    import sys

    from geminidr.interactive.interactive_config import interactive_conf
    assert interactive_conf().browser is None  # not loaded yet

    from recipe_system.config import load_config
    load_config([sys.argv[1]])

    assert interactive_conf().browser == "chrome", interactive_conf().browser
""")


def test_interactive_config_requested_before_load(tmp_path):
    rcfile = tmp_path / "dragonsrc"
    rcfile.write_text("[interactive]\nbrowser = chrome\n")

    result = subprocess.run(
        [sys.executable, "-c", SCRIPT_EARLY_REQUEST, str(rcfile)],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
