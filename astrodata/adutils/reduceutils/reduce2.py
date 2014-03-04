#!/usr/bin/env python
#
#                                                                  gemini_python
#
#                                                              astrodata/scripts
#                                                                     reduce2.py
# ------------------------------------------------------------------------------
# $Id$
# ------------------------------------------------------------------------------
__version__      = '$Rev$'[11:-3]
__version_date__ = '$Date$'[7:-3]
# ------------------------------------------------------------------------------
# reduce2.py -- refactored reduce, cl parsing exported, functionalized.
#               see parseUtils.py
# ------------------------------------------------------------------------------
__version__ = '2.0'
# ------------------------------------------------------------------------------
"""
This module (reduce3) provides a functionalized interface to running the core 
processing of the reduce pipeline. This core functionality is provided by the 
imported coreReduce module. 

It provides both a nominal 'reduce' command line interface and a 'main'
function that can be called with an 'args' parameter.

Eg.,

>>> args = [defined by ArgumentParser or ProxyInterface]
>>> import reduce3
>>> reduce3.main(args)

In the above example, 'args' is one of 

-- ArgumentParser Namespace instance, delivered from the command line.
-- ProxyInterface instance, instantiaed by a caller.

Thoough it is possible for a caller to instantiate a defined ArgumentParser 
object as delivered by parseUtils.buildParser(), and create an "args"
instance with a call on parse_args(), as is done for the cli, functionally, a
caller should use the ProxyInterface class and create an "args" instance from
that class's constructor.

ProxyInterface is provided for progammatic convenience, i.e. one does not
need to import argparse and use argparse just to get an 'args' object.
ProxyInterface provides an equivalent API to that of the ArgumentParser 
Namespace instance, through standard gettr and settr behaviour on attributes
of the instance. The ProxyInterface constructor provides an equivalent 
set of options and defaults to those of the command line interface. Callers 
can then provide argument values to the 'args' instance as would have passed 
from a command line to an ArgumentParser Namespace instance.

Eg.,

ArgumentParser:
--------------
>>> parser = argparse.ArgumentParser()
>>> args = parser.parse_args()
>>> args.logfile
'reduce.log'
>>> args.logfile = "another_reduce.log"

ProxyInterface:
--------------
>>> args = ProxyInterface()
>>> args.logfile
'reduce.log'
>>> args.logfile = "another_reduce.log"
>>> args.files
[]
>>> args.files = ['S20130616S0019.fits']
>>> args.recipename = 'recipe.ArgTests'
>>> args.recipename
'recipe.ArgTests'
>>>

At which point, the caller would then call main()

>>> reduce3.main(args)

When run from the command line, the program exits with an appropriate
exit status.
"""

import os
import sys

from signal import SIGINT
# ---------------------------- Package Import ----------------------------------
from astrodata import Proxies
from astrodata import Errors

from astrodata.AstroData import AstroData
from astrodata.RecipeManager import RecipeLibrary

from astrodata.adutils import terminal
from astrodata.adutils import logutils

from astrodata.gdpgutil   import cluster_by_groupid
from astrodata.debugmodes import set_descriptor_throw

import parseUtils
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
def convert_inputs(allinputs, intel):
    if intel:
        typeIndex = cluster_by_groupid(allinputs)
        # If super intelligence, it would determine ordering. Now, recipes in
        # simple order, (i.e. the order of values()).
        allinputs = typeIndex.values()
    else:
        nl = []
        for inp in allinputs:
            try:
                ad = AstroData(inp)
                ad.filename = os.path.basename(ad.filename)
                ad.mode = "readonly"
            except Errors.AstroDataError, err:
                log.warning(err)
                log.warning("Can't Load Dataset: %s" % inp)
                continue
            nl.append(ad)
            ad = None                 # danger of accidentally using this!
    try:
        assert(nl)
        allinputs = [nl]          # Why is allinputs blown away here?
    except AssertionError:
        msg = "No AstroData objects were created."
        log.warning(msg)
        raise IOError, msg
    return allinputs

def signal_invoked(invoked):
    if invoked:
        opener = "reduce started in adcc mode (--invoked)"
        log.fullinfo("."*len(opener))
        log.fullinfo(opener)
        log.fullinfo("."*len(opener))
        sys.stdout.flush()
    return

def start_proxy_servers():
    adcc_proc = None
    pprox     = Proxies.PRSProxy.get_adcc(check_once=True)
    if not pprox:
        adcc_proc = Proxies.start_adcc()

    # launch xmlrpc interface for control and communication
    reduceServer = Proxies.ReduceServer()
    prs = Proxies.PRSProxy.get_adcc(reduce_server=reduceServer)

    return (adcc_proc, reduceServer, prs)

# ------------------------------------------------------------------------------
def main(args):
    """
    See the module docstring on how to call main.

    parameters: <inst>, 'args' object
    return:     <void>
    """
    global log

    from coreReduce import CoreReduce
    from astrodata.adutils import logutils
    # --------------------------------------------------------------------------
    terminal.forceWidth  = args.forceWidth
    terminal.forceHeight = args.forceHeight

    logutils.config(mode=args.logmode,
                    console_lvl=args.loglevel,
                    file_name=args.logfile)

    log = logutils.get_logger(__name__)

    # set user parameters 
    user_params, global_params = parseUtils.set_user_params(args.userparam)
    args.user_params  = user_params
    args.globalParams = global_params
    args.logindent    = logutils.SW

    #global configs
    set_descriptor_throw(args.throwDescriptorExceptions)
    signal_invoked(args.invoked)

    # first check on os accessibilty, the convert to ad instances.
    # allinputs is returned as a list of lists, [[ad1, ad2, ...], []]
    valid_inputs = parseUtils.check_files(args)
    allinputs    = convert_inputs(valid_inputs, args.intelligence)

    # defined error messages, status
    red_msg  = ("Unable to get ReductionObject for type %s" % args.astrotype)
    rec_msg  = ("No recipes found for types: %s")
    adcc_msg = ("Trouble unregistering from adcc shared services.")
    estat    = 0

    # Get a recipe library
    rl = RecipeLibrary()

    # --------------------------------------------------------------------------
    #                                START ADCC
    # Start adcc always, since it wants the reduceServer. Prefer not to provide
    # to every component that wants to use the adcc as an active-library
    adcc_proc, reduceServer, prs = start_proxy_servers()

    # --------------------------------------------------------------------------
    # called once per substep (every yeild in any primitive when struck)
    # registered with the reduction object.
    from astrodata.ReductionObjects import command_clause

    # --------------------------------------------------------------------------
    i = 0
    nof_ad_sets = len(allinputs)
    for infiles in allinputs:
        i += 1
        log.stdinfo("Starting Reduction on set #%d of %d" % (i, nof_ad_sets))
        title = "  Processing dataset(s):\n"
        title += "\n".join("\t" + ad.filename for ad in infiles)
        log.stdinfo("\n" + title + "\n")
        print "Instantiating CoreReduce ..."

        try:
            core_reduce = CoreReduce(infiles, args, rl)
            core_reduce.runr(command_clause)
        except KeyboardInterrupt:
            log.error("User Interrupt -- main() recieved SIGINT")
            estat = SIGINT
            reduceServer.finished = True
            prs.registered = False
            log.stdinfo("\n\n\tSUMMARY reduce exit state")
            log.stdinfo("\t-----------------------------------")
            log.stdinfo("\texit code:\t\t\t%d" % estat)
            log.stdinfo("\treduceServer finish  state:  %s" % str(reduceServer.finished))
            log.stdinfo("\tprsproxy registered  state: %s" % str(prs.registered))
            log.stdinfo("\t-----------------------------------")
            break

    if reduceServer.finished:
        pass
    else:
        reduceServer.finished = True
    if prs.registered:
        prs.unregister()
    print ("reduce finished on exit status %d" % estat)
    return (estat)

# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = parseUtils.buildParser(__version__)
    args   = parser.parse_args()

    if args.displayflags:
        parseUtils.show_parser_options(parser, args)
        for item in ["Input fits file(s):\t%s" % inf for inf in args.files]:
            print item
        #sys.exit()

    # Deal with argparse structures, which are different than optparse 
    # implementation. astrotype, recipename, etc. are now lists.
    args   = parseUtils.normalize_args(args)

    sys.exit(main(args))
