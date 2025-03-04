from _pytest.logging import LogCaptureHandler

import logging
from gempy.utils import logutils

log = logutils.get_logger()


def test_sanity(caplog):
    caplog.set_level(logging.NOTSET)
    log.info("Hello, world!")
    assert 'Hello, world!' in caplog.text

def test_fullinfo(caplog):
    caplog.set_level(logging.NOTSET)
    log.fullinfo("Fullinfo is an extra log level")
    assert "Fullinfo is an extra log level" in caplog.text

def test_nofullinfo(caplog):
    # At INFO level (20), FULLINFO (15) should not show up
    caplog.set_level(logging.INFO)
    log.fullinfo("Fullinfo is an extra log level")
    assert "Fullinfo is an extra log level" not in caplog.text

def test_conditional_levelname(caplog):
    caplog.set_level(logging.INFO)
    logutils.change_level(logging.INFO)
    for handler in log.handlers:
        if isinstance(handler, LogCaptureHandler):
            handler.formatter = logutils.DragonsConsoleFormatter()
    log.info("Info message")
    assert "Info message" in caplog.text
    assert "INFO" not in caplog.text

def test_conditional_levelname2(caplog):
    caplog.set_level(logging.NOTSET)
    for handler in log.handlers:
        if isinstance(handler, LogCaptureHandler):
            handler.formatter = logutils.DragonsConsoleFormatter()
    log.warning("Warn message")
    assert "WARNING - Warn message" in caplog.text

def test_indent(caplog):
    caplog.set_level(logging.NOTSET)
    for handler in log.handlers:
        if isinstance(handler, LogCaptureHandler):
            handler.formatter = logutils.DragonsConsoleFormatter()
    log.info("zero indent")
    assert caplog.text == "zero indent\n"

    # indents are three spaces
    caplog.clear()
    logutils.update_indent(1)
    log.info("one indent")
    assert caplog.text == "   one indent\n"

    caplog.clear()
    logutils.update_indent(2)
    log.info("two indent")
    assert caplog.text == "      two indent\n"
