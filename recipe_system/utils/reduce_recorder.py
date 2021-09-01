import json
import sys

import astrodata

from gempy.utils import logutils
from recipe_system import __version__


__all__ = ["in_replay", "init_reduce_recorder", "record_interactive", "record_reduction", "load_reduce_record",
           "load_replay_interactive_settings", "record_reduction_in_ad", "load_reduce_record_from_ad"]

reduce_recorder = None
replay_record = None
replay_step = 0
reduce_filename = None
disable_replay = False
warned_user = False

log = logutils.get_logger(__name__)


def in_replay():
    """
    Check if we are in an active replay.

    This is a utlity call to check if we are currently doing a
    replay of a recorded session.

    Returns
    -------
    bool : True if we are in a replay, False if not (including if the user aborted the replay by modifying some inputs
        earlier)
    """
    if replay_record and not disable_replay:
        return True
    return False


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
    if replay_record is not None and not disable_replay:
        # We're doing a replay, check if this interactive step was modified
        validate_replay_step(record)


def record_reduction():
    """
    Save a record of this reduction session to the json file.

    This call writes all of the information needed for this reduce session,
    including the interactive tools, to a json file.
    """
    if reduce_recorder is not None and reduce_filename is not None:
        with open(reduce_filename, 'w') as reduce_file:
            output = json.dumps(reduce_recorder, indent=4)
            reduce_file.write(f"{output}")


def record_reduction_in_ad(ad):
    if reduce_recorder is not None:
        record = json.dumps(reduce_recorder, indent=4)
        for ext in ad:
            ext.record = record


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
    if filename.endswith('.json'):
        with open(filename, 'r') as record_file:
            global replay_record
            replay_record = json.loads(record_file.read())
            if replay_record["version"] != __version__:
                log.warning("This version of DRAGONS ({}) does not match the version for this replay record: {}"
                            .format(__version__, replay_record["version"]))
        return replay_record["args"] if replay_record else []
    else:
        ad = astrodata.open(f"{filename}.fits")
        return load_reduce_record_from_ad(ad)


def load_reduce_record_from_ad(ad):
    global replay_record
    record = ad[0].record
    if record:
        replay_record = json.loads(record)
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
    if disable_replay:
        return
    global replay_step
    if replay_record and replay_step < len(replay_record["interactive"]):
        retval = replay_record["interactive"][replay_step]
        replay_step += 1
        visualizer.load(retval)
    elif replay_record and replay_step >= len(replay_record["interactive"]):
        log.warning("Request for interactive settings beyond that recorded in the replay file.  This replay "
                    "is probably not compatible with your current DRAGONS install.")


def validate_replay_step(record):
    """
    This call validates the exit state of an interactive step vs what
    was recorded.

    Check the output state of an interactive step vs what was recorded.
    If the interactive step has been modified, disable the replay functionality
    for all remaining steps.

    Parameters
    ----------
    record : dict
        Dictionary describing state of the interactive step, as would be saved when recording the session
    """
    global replay_step
    global disable_replay
    global warned_user
    if replay_record and replay_step-1 < len(replay_record["interactive"]):
        retval = replay_record["interactive"][replay_step-1]
        if retval != record:
            if not warned_user:
                log.warning("Interactive settings differ from recorded values, "
                            "replay turned off for remainder of reduction")
                warned_user = True
            disable_replay = True
    elif replay_record and replay_step >= len(replay_record["interactive"]):
        log.warning("Request to validate interactive settings beyond that recorded in the replay file.  This replay "
                    "is probably not compatible with your current DRAGONS install.")
