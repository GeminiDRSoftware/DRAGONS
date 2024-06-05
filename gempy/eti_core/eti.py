from multiprocessing import Process, Queue
from subprocess import STDOUT, CalledProcessError, check_output

from ..utils import logutils

log = logutils.get_logger(__name__)


def loop_process(in_queue, out_queue):
    """
    Code to spawn a subprocess for running external tasks

    Parameters
    ----------
    in_queue : multiprocessing.Queue
        (add docstring)
    out_queue : multiprocessing.Queue
        (add docstring)
    """
    while True:
        try:
            cmd = in_queue.get()
            try:
                result = check_output(cmd, stderr=STDOUT)
            except CalledProcessError as e:
                result = e
            except Exception as ex:
                result = ex
            out_queue.put(result)
        except KeyboardInterrupt:
            break


class ETISubprocess:
    """
    A singleton class that creates an instance of __ETISubprocess, which
    any future instances of ETISubprocess will point to.
    """
    class __ETISubprocess:
        def __init__(self):
            self.inQueue = Queue()
            self.outQueue = Queue()
            self.process = Process(target=loop_process,
                                   args=(self.inQueue, self.outQueue))
            self.process.start()

        def terminate(self):
            self.process.terminate()

    instance = None

    def __new__(cls):
        if not ETISubprocess.instance:
            ETISubprocess.instance = ETISubprocess.__ETISubprocess()
        return ETISubprocess.instance

    def __getattr__(self, name):
        return getattr(self.instance, name)

    def __setattr__(self, name, value):
        return setattr(self.instance, name, value)


class ExternalTaskInterface:
    """
    The External Task Interface base class. This is a way for the Recipe
    System to interact with ouside software. It prepares, executes, recovers,
    and cleans all files and parameters pertaining to any external task
    that interfaces with the recipe system.

    Parameters
    ----------
    primitives_class : (add input type)
        (add docstring)
    inputs : (add input type)
        (add docstring)
    params : (add input type)
        (add docstring)
    """
    param_objs = None
    file_objs = None
    inputs = None
    params = None

    def __init__(self, primitives_class=None, inputs=None, params=None):
        """
        :param rc: Used to store reduction information
        :type rc: ReductionContext
        """
        log.debug("ExternalTaskInterface __init__")
        self.inputs = inputs
        self.params = params
        self.param_objs = []
        self.file_objs = []
        self.inQueue = None
        self.outQueue = None
        try:
            self.inQueue = primitives_class.eti_subprocess.inQueue
            self.outQueue = primitives_class.eti_subprocess.outQueue
        except AttributeError:
            log.debug("ETI: Cannot access Queues")
            return
        if self.inQueue._closed or self.outQueue._closed:
            log.warning("ETI: One or both Queues is closed")
            self.inQueue = None
            self.outQueue = None

    def run(self):
        log.debug("ExternalTaskInterface.run()")
        self.prepare()
        self.execute()
        self.recover()
        self.clean()

    def add_param(self, param):
        log.debug("ExternalTaskInterface.add_param()")
        self.param_objs.append(param)

    def add_file(self, a_file):
        log.debug("ExternalTaskInterface.add_file()")
        self.file_objs.append(a_file)

    def prepare(self):
        log.debug("ExternalTaskInterface.prepare()")
        for par in self.param_objs:
            par.prepare()
        for fil in self.file_objs:
            fil.prepare()

    def execute(self):
        log.debug("ExternalTaskInterface.execute()")

    def recover(self):
        log.debug("ExternalTaskInterface.recover()")
        for par in self.param_objs:
            par.recover()
        for fil in self.file_objs:
            fil.recover()

    def clean(self):
        log.debug("ExternalTaskInterface.clean()")
        for par in self.param_objs:
            par.clean()
        for fil in self.file_objs:
            fil.clean()
