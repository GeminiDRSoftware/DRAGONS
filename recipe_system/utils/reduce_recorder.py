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
    """
    Setup the reduce job to record interactive parameters to the named file.

    This call sets up the reduce job to save each interactive session to a
    json file for future reuse.

    Parameters
    ----------
    filename : str
        Name of the file to save the state in
    """
    global reduce_recorder
    global reduce_filename
    reduce_recorder = {
        "version": __version__,
        "args": sys.argv[1:],
        "interactive": [],
    }
    reduce_filename = filename


def record_interactive(record):
    """
    Add a json record to the record to be saved for this reduce job.

    This call takes a single dictionary of state representing the current interactive
    tool's state and adds it to the set to be saved for this session overall.

    Parameters
    ----------
    record : dict
        Dictionary describing the state of the current interactive tool
    """
    global reduce_recorder
    if reduce_recorder is not None:
        reduce_recorder["interactive"].append(record)


def record_reduction():
    """
    Save a record of this reduction session to the json file.

    This call writes all of the information needed for this reduce session,
    including the interactive tools, to a json file.
    """
    if reduce_recorder is not None:
        with open(reduce_filename, 'w') as reduce_file:
            output = json.dumps(reduce_recorder, indent=4)
            reduce_file.write(f"{output}")


def load_reduce_record(filename):
    """
    Load the reduce session from the given save file.

    This call opens a previously saved reduce session from a json file
    and prepares it for use by the current reduce.

    Parameters
    ----------
    filename : str
        Name of the json file to read
    """
    with open(filename, 'r') as record_file:
        global replay_record
        replay_record = json.loads(record_file.read())
        if replay_record["version"] != __version__:
            log.warning("This version of DRAGONS ({}) does not match the version for this replay record: {}"
                        .format(__version__, replay_record["version"]))
    return replay_record["args"] if replay_record else []


def load_replay_interactive_settings(visualizer):
    """
    Load the current interactive tool state from the record.

    This call initializes an interactive tool based on the current
    step in the loaded json file.  Each time an interface is loaded,
    the system advances to the next saved interactive state to use for
    the next tool.

    Parameters
    ----------
    visualizer : :class:`~geminidr.interactive.PrimitiveVisualizer`
        visualizer to be initialized
    """
    global replay_step
    if replay_record and replay_step < len(replay_record["interactive"]):
        retval = replay_record["interactive"][replay_step]
        replay_step += 1
        visualizer.load(retval)
    elif replay_record and replay_step >= len(replay_record["interactive"]):
        log.warning("Request for interactive settings beyond that recorded in the replay file.  This replay "
                    "is probably not compatible with your current DRAGONS install.")

