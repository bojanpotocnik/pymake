# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
Makefile execution.

Multiple `makes` can be run within the same process. Each one has an entirely data.Makefile and .Target
structure, environment, and working directory. Typically they will all share a parallel execution context,
except when a submake specifies -j1 when the parent make is building in parallel.
"""

import logging
import os
import re
import sys
from optparse import OptionParser

from . import data, parserdata, process, util
from . import errors

# TODO: If this ever goes from relocatable package to system-installed, this may need to be
# a configured-in path.

make_py_path = util.normaljoin(os.path.dirname(__file__), '../make.py')

_simple_options = re.compile(r'^[a-zA-Z]+(\s|$)')


def parse_make_flags(env):
    """
    Parse MAKEFLAGS from the environment into a sequence of command-line arguments.
    """

    makeflags = env.get('MAKEFLAGS', '')
    makeflags = makeflags.strip()

    if makeflags == '':
        return []

    if _simple_options.match(makeflags):
        makeflags = '-' + makeflags

    options = []
    current_options = ''

    i = 0
    while i < len(makeflags):
        c = makeflags[i]
        if c.isspace():
            options.append(current_options)
            current_options = ''
            i += 1
            while i < len(makeflags) and makeflags[i].isspace():
                i += 1
            continue

        if c == '\\':
            i += 1
            if i == len(makeflags):
                raise errors.DataError("MAKEFLAGS has trailing backslash")
            c = makeflags[i]

        current_options += c
        i += 1

    if current_options != '':
        options.append(current_options)

    return options


# noinspection PyUnusedLocal
def _version(*args):
    print("""pymake: GNU-compatible make program
Copyright (C) 2009 The Mozilla Foundation <http://www.mozilla.org/>
This is free software; see the source for copying conditions.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.""")


_log = logging.getLogger('pymake.execution')


class _MakeContext(object):
    def __init__(self, make_flags, make_level, working_dir, context, env, targets, options, ostmts, overrides, cb):
        self.make_flags = make_flags
        self.make_level = make_level

        self.working_dir = working_dir
        self.context = context
        self.env = env
        self.targets = targets
        self.options = options
        self.ostmts = ostmts
        self.overrides = overrides
        self.cb = cb

        self.restarts = 0

        self.makefile = None
        self.real_targets = None
        self.t_stack = None

        self.remake_cb(True)

    def remake_cb(self, remade, error=None):
        if error is not None:
            print(error)
            self.context.defer(self.cb, 2)
            return

        if remade:
            if self.restarts > 0:
                _log.info("make.py[%i]: Restarting makefile parsing", self.make_level)

            self.makefile = data.Makefile(restarts=self.restarts,
                                          make='%s %s' % (
                                              sys.executable.replace('\\', '/'), make_py_path.replace('\\', '/')),
                                          makeflags=self.make_flags,
                                          make_overrides=self.overrides,
                                          work_dir=self.working_dir,
                                          context=self.context,
                                          env=self.env,
                                          make_level=self.make_level,
                                          targets=self.targets,
                                          keep_going=self.options.keepgoing,
                                          silent=self.options.silent,
                                          just_print=self.options.justprint)

            self.restarts += 1

            try:
                self.ostmts.execute(self.makefile)
                for f in self.options.makefiles:
                    self.makefile.include(f)
                self.makefile.finishparsing()
                self.makefile.remakemakefiles(self.remake_cb)
            except errors.MakeError as e:
                print(e)
                self.context.defer(self.cb, 2)

            return

        if len(self.targets) == 0:
            if self.makefile.defaulttarget is None:
                print("No target specified and no default target found.")
                self.context.defer(self.cb, 2)
                return

            _log.info("Making default target %s", self.makefile.defaulttarget)
            self.real_targets = [self.makefile.defaulttarget]
            self.t_stack = ['<default-target>']
        else:
            self.real_targets = self.targets
            self.t_stack = ['<command-line>']

        self.makefile.gettarget(self.real_targets.pop(0)).make(self.makefile, self.t_stack, cb=self.make_cb)

    # noinspection PyUnusedLocal
    def make_cb(self, error, did_anything):
        assert error in (True, False)

        if error:
            self.context.defer(self.cb, 2)
            return

        if not len(self.real_targets):
            if self.options.printdir:
                print("make.py[%i]: Leaving directory '%s'" % (self.make_level, self.working_dir))
            sys.stdout.flush()

            self.context.defer(self.cb, 0)
        else:
            self.makefile.gettarget(self.real_targets.pop(0)).make(self.makefile, self.t_stack, self.make_cb)


def main(args, env, cwd, cb):
    """
    Start a single makefile execution, given a command line, working directory, and environment.

    :param args:
    :param env:
    :param cwd:
    :param cb: a callback to notify with an exit code when make execution is finished.
    """

    # noinspection SpellCheckingInspection
    make_level = int(env.get('MAKELEVEL', '0'))
    options = None

    try:
        op = OptionParser()
        op.add_option('-f', '--file', '--makefile',
                      action='append',
                      dest='makefiles',
                      default=[])
        op.add_option('-d',
                      action="store_true",
                      dest="verbose", default=False)
        # noinspection SpellCheckingInspection
        op.add_option('-k', '--keep-going',
                      action="store_true",
                      dest="keepgoing", default=False)
        # noinspection SpellCheckingInspection
        op.add_option('--debug-log',
                      dest="debuglog", default=None)
        op.add_option('-C', '--directory',
                      dest="directory", default=None)
        # noinspection SpellCheckingInspection
        op.add_option('-v', '--version', action="store_true",
                      dest="printversion", default=False)
        # noinspection SpellCheckingInspection
        op.add_option('-j', '--jobs', type="int",
                      dest="jobcount", default=1)
        op.add_option('-w', '--print-directory', action="store_true",
                      dest="printdir")
        op.add_option('--no-print-directory', action="store_false",
                      dest="printdir", default=True)
        op.add_option('-s', '--silent', action="store_true",
                      dest="silent", default=False)
        # noinspection SpellCheckingInspection
        op.add_option('-n', '--just-print', '--dry-run', '--recon',
                      action="store_true",
                      dest="justprint", default=False)

        options, arguments1 = op.parse_args(parse_make_flags(env))
        options, arguments2 = op.parse_args(args, values=options)

        op.destroy()

        arguments = arguments1 + arguments2

        if options.printversion:
            _version()
            cb(0)
            return

        short_flags = []
        long_flags = []

        if options.keepgoing:
            short_flags.append('k')

        if options.printdir:
            short_flags.append('w')

        if options.silent:
            short_flags.append('s')
            options.printdir = False

        if options.justprint:
            short_flags.append('n')

        log_level = logging.WARNING
        if options.verbose:
            log_level = logging.DEBUG
            short_flags.append('d')

        log_kwargs = {}
        if options.debuglog:
            log_kwargs['filename'] = options.debuglog
            long_flags.append('--debug-log=%s' % options.debuglog)

        if options.directory is None:
            work_dir = cwd
        else:
            work_dir = util.normaljoin(cwd, options.directory)

        if options.jobcount != 1:
            long_flags.append('-j%i' % (options.jobcount,))

        makeflags = ''.join(short_flags)
        if len(long_flags):
            makeflags += ' ' + ' '.join(long_flags)

        logging.basicConfig(level=log_level, **log_kwargs)

        context = process.getcontext(options.jobcount)

        if options.printdir:
            print("make.py[%i]: Entering directory '%s'" % (make_level, work_dir))
            sys.stdout.flush()

        if len(options.makefiles) == 0:
            if os.path.exists(util.normaljoin(work_dir, 'Makefile')):
                options.makefiles.append('Makefile')
            else:
                print("No makefile found")
                cb(2)
                return

        statements, targets, overrides = parserdata.parsecommandlineargs(arguments)

        _MakeContext(makeflags, make_level, work_dir, context, env, targets, options, statements, overrides, cb)
    except errors.MakeError as e:
        print(e)
        if options is not None:
            if options.printdir:
                # noinspection PyUnboundLocalVariable
                print("make.py[%i]: Leaving directory '%s'" % (make_level, work_dir))
        sys.stdout.flush()
        cb(2)
        return
