import os


class PidFileError(Exception):
    pass

# Stole this utility code from the FITS Servers.  For now, keeping it with interactive as that's the only place
# it is being used.  However, if we find a use for it elsewhere we can move it to somewhere more generally
# appropriate.
#
# It could also be moved up to somewhere like FitsStorageDB and shared with the FITS Server that way


class PidFile(object):
    def __init__(self, logger, name, lockfile_dir='/tmp', dummy=False):
        try:
            self.path = os.path.join(lockfile_dir, name + '.lock')
        except TypeError:
            # We got None as 'name'
            self.path = os.path.join(lockfile_dir, 'unknown_name.lock')
        self.dummy = dummy
        self.l = logger

    def __enter__(self):
        """If there's no PID file, create a new one and return the context manager.

           If there's a PID file, remove and create a new one ONLY if the we can read
           the PID and it corresponds to a non-existent process, or to one that belongs
           to other user. Otherwise, bail out"""

        if self.dummy:
            return self

        if os.path.exists(self.path):
            self.l.info("Lockfile {} exists; testing for viability".format(self.path))

            try:
                try:
                    oldpid = int(open(self.path, 'r').read())
                except OSError:
                    raise PidFileError("Could not read PID from lockfile {}".format(self.path))
                # Test if we have any control over the process that created the PID
                os.kill(oldpid, 0)
                raise PidFileError("Lockfile {} refers to PID {} which appears to be valid".format(self.path, oldpid))
            except (TypeError, ValueError):
                raise PidFileError("Cannot recognize the PID in lockfile {}".format(self.path))
            except OSError:
                # This is ok - go on
                self.l.error("PID in lockfile refers to a process which either doesn't exist, or is not ours - {}".format(oldpid))

            self.l.error("Removing the current PID file")
            os.unlink(self.path)
        else:
            self.l.info("Lockfile {} does not exist".format(self.path))

        self.l.info("Creating new lockfile {}".format(self.path))
        with open(self.path, 'w') as lfd:
            lfd.write('{}\n'.format(os.getpid()))

        return self

    def __exit__(self, exc_type, exc_value, tb):
        if not self.dummy:
            self.l.info("Removing the PID file")
            os.unlink(self.path)

        if exc_type is not None:
            raise
