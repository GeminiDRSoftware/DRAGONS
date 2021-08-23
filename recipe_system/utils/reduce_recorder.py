import json
import sys

from gempy.utils import logutils
from recipe_system import __version__


__all__ = ["init_reduce_recorder", "record_interactive", "record_reduction", "load_reduce_record",
           "load_replay_interactive_settings"]

reduce_recorder = None
replay_record = None
replay_step = 0
reduce_filename = None


log = logutils.get_logger(__name__)


def init_reduce_recorder(filename):
    global reduce_recorder
    global reduce_filename
    reduce_recorder = {
        "version": __version__,
        "args": sys.argv[1:],
        "interactive": [],
    }
    reduce_filename = filename


def record_interactive(record):
    global reduce_recorder
    if reduce_recorder is not None:
        reduce_recorder["interactive"].append(record)


def record_reduction():
    if reduce_recorder is not None:
        with open(reduce_filename, 'w') as reduce_file:
            output = json.dumps(reduce_recorder, indent=4)
            reduce_file.write(f"{output}")


def load_reduce_record(filename):
    with open(filename, 'r') as record_file:
        global replay_record
        replay_record = json.loads(record_file.read())
        if replay_record["version"] != __version__:
            log.warning("This version of DRAGONS ({}) does not match the version for this replay record: {}"
                        .format(__version__, replay_record["version"]))
    return replay_record["args"] if replay_record else []


def load_replay_interactive_settings(visualizer):
    global replay_step
    if replay_record and replay_step < len(replay_record["interactive"]):
        retval = replay_record["interactive"][replay_step]
        replay_step += 1
        visualizer.load(retval)
    if replay_record and replay_step >= len(replay_record["interactive"]):
        log.warning("Request for interactive settings beyond that recorded in the replay file.  This replay "
                    "is probably not compatible with your current DRAGONS install.")

