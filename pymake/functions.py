"""
Makefile functions.
"""
from __future__ import print_function

import abc
import logging
import os
import subprocess
import sys
from typing import TYPE_CHECKING, List

from . import data
from . import errors
from . import parser, util
from .globrelative import glob

if TYPE_CHECKING:  # To prevent cyclic and unused imports in runtime.
    import io
    from . import parserdata

log = logging.getLogger('pymake.data')


def emit_expansions(descend, *expansions):
    """Helper function to emit all expansions within an input set."""
    for expansion in expansions:
        yield expansion

        if not descend or not isinstance(expansion, list):
            continue

        for e, is_func in expansion:
            if is_func:
                for exp in e.expansions(True):
                    yield exp
            else:
                yield e


class Function:
    """
    An object that represents a function call. This class is always subclassed
    with the following methods and attributes:

    name = function name
    min_args = minimum # of arguments
    max_args = maximum # of arguments (0 means unlimited)

    def resolve(self, makefile, variables, fd, setting)
        Calls the function
        calls fd.write() with strings
    """
    name: str = None
    """Function name"""
    min_args: int = None
    """Minimum number of arguments"""
    max_args: int = None
    """Maximum number of arguments (0 means unlimited)"""

    __slots__ = ('_arguments', 'loc')

    def __init__(self, loc: 'parserdata.Location'):
        self._arguments = []
        self.loc: 'parserdata.Location' = loc
        assert self.min_args > 0

    def __getitem__(self, key):
        return self._arguments[key]

    def setup(self):
        argc = len(self._arguments)

        if argc < self.min_args:
            raise errors.DataError("Not enough arguments to function %s, requires %s"
                                   % (self.name, self.min_args), self.loc)

        assert self.max_args == 0 or argc <= self.max_args, "Parser screwed up, gave us too many args"

    def append(self, arg):
        assert isinstance(arg, (data.Expansion, data.StringExpansion))
        self._arguments.append(arg)

    def to_source(self):
        """Convert the function back to make file "source" code."""
        if not hasattr(self, 'name'):
            raise Exception("%s must implement to_source()." % self.__class__)

        # The default implementation simply prints the function name and all
        # the arguments joined by a comma.
        # According to the GNU make manual Section 8.1, whitespace around
        # arguments is *not* part of the argument's value. So, we trim excess
        # white space so we have consistent behavior.
        args = []
        curly = False
        for i, arg in enumerate(self._arguments):
            arg = arg.to_source()

            if i == 0:
                arg = arg.lstrip()

            # Are balanced parents even OK?
            if arg.count('(') != arg.count(')'):
                curly = True

            args.append(arg)

        if curly:
            return '${%s %s}' % (self.name, ','.join(args))

        return '$(%s %s)' % (self.name, ','.join(args))

    def expansions(self, descend=False):
        """Obtain all expansions contained within this function.

        By default, only expansions directly part of this function are
        returned. If descend is True, we will descend into child expansions and
        return all of the composite parts.

        This is a generator for pymake.data.BaseExpansion instances.
        """
        # Our default implementation simply returns arguments. More advanced
        # functions like variable references may need their own implementation.
        return emit_expansions(descend, *self._arguments)

    @property
    def is_filesystem_dependent(self):
        """Exposes whether this function depends on the filesystem for results.

        If True, the function touches the filesystem as part of evaluation.

        This only tests whether the function itself uses the filesystem. If
        this function has arguments that are functions that touch the
        filesystem, this will return False.
        """
        return False

    @abc.abstractmethod
    def resolve(self, makefile: 'data.Makefile', variables: 'data.Variables',
                fd: 'io.StringIO', setting: List[str]) -> None:
        raise NotImplementedError()

    def __len__(self):
        return len(self._arguments)

    def __repr__(self):
        return "%s<%s>(%r)" % (
            self.__class__.__name__, self.loc,
            ','.join([repr(a) for a in self._arguments]),
        )

    def __eq__(self, other):
        if not hasattr(self, 'name'):
            raise Exception("%s must implement __eq__." % self.__class__)

        if type(self) != type(other):
            return False

        if self.name != other.name:
            return False

        # noinspection PyProtectedMember
        if len(self._arguments) != len(other._arguments):
            return False

        for i in range(len(self._arguments)):
            # According to the GNU make manual Section 8.1, whitespace around
            # arguments is *not* part of the argument's value. So, we do a
            # whitespace-agnostic comparison.
            if i == 0:
                a = self._arguments[i]
                a.lstrip()

                # noinspection PyProtectedMember
                b = other._arguments[i]
                b.lstrip()

                if a != b:
                    return False

                continue

            # noinspection PyProtectedMember
            if self._arguments[i] != other._arguments[i]:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class VariableRef(Function):
    AUTOMATIC_VARIABLES = {'@', '%', '<', '?', '^', '+', '|', '*'}

    __slots__ = ('vname', 'loc')

    # noinspection PyMissingConstructor
    def __init__(self, loc, vname):
        self.loc = loc
        assert isinstance(vname, (data.Expansion, data.StringExpansion))
        self.vname = vname

    def setup(self):
        assert False, "Shouldn't get here"

    def resolve(self, makefile, variables, fd, setting):
        vname = self.vname.resolvestr(makefile, variables, setting)
        if vname in setting:
            raise errors.DataError("Setting variable '%s' recursively references itself." % (vname,), self.loc)

        flavor, source, value = variables.get(vname)
        if value is None:
            log.debug("%s: variable '%s' was not set" % (self.loc, vname))
            return

        value.resolve(makefile, variables, fd, setting + [vname])

    def to_source(self):
        if isinstance(self.vname, data.StringExpansion):
            if self.vname.s in self.AUTOMATIC_VARIABLES:
                return '$%s' % self.vname.s

            return '$(%s)' % self.vname.s

        return '$(%s)' % self.vname.to_source()

    def expansions(self, descend=False):
        return emit_expansions(descend, self.vname)

    def __repr__(self):
        return "VariableRef<%s>(%r)" % (self.loc, self.vname)

    def __eq__(self, other):
        if not isinstance(other, VariableRef):
            return False

        return self.vname == other.vname


# noinspection SpellCheckingInspection
class SubstitutionRef(Function):
    """$(VARNAME:.c=.o) and $(VARNAME:%.c=%.o)"""

    __slots__ = ('loc', 'vname', 'substfrom', 'substto')

    # noinspection PyMissingConstructor
    def __init__(self, loc, var_name, subst_from, subst_to):
        self.loc = loc
        self.vname = var_name
        self.substfrom = subst_from
        self.substto = subst_to

    def setup(self):
        assert False, "Shouldn't get here"

    def resolve(self, makefile, variables, fd, setting):
        vname = self.vname.resolvestr(makefile, variables, setting)
        if vname in setting:
            raise errors.DataError("Setting variable '%s' recursively references itself." % (vname,), self.loc)

        substfrom = self.substfrom.resolvestr(makefile, variables, setting)
        substto = self.substto.resolvestr(makefile, variables, setting)

        flavor, source, value = variables.get(vname)
        if value is None:
            log.debug("%s: variable '%s' was not set" % (self.loc, vname))
            return

        f = data.Pattern(substfrom)
        if not f.ispattern():
            f = data.Pattern('%' + substfrom)
            substto = '%' + substto

        fd.write(' '.join([f.subst(substto, word, False)
                           for word in value.resolvesplit(makefile, variables, setting + [vname])]))

    def to_source(self):
        return '$(%s:%s=%s)' % (
            self.vname.to_source(),
            self.substfrom.to_source(),
            self.substto.to_source())

    def expansions(self, descend=False):
        return emit_expansions(descend, self.vname, self.substfrom,
                               self.substto)

    def __repr__(self):
        return "SubstitutionRef<%s>(%r:%r=%r)" % (
            self.loc, self.vname, self.substfrom, self.substto,)

    def __eq__(self, other):
        if not isinstance(other, SubstitutionRef):
            return False

        return (self.vname == other.vname and self.substfrom == other.substfrom
                and self.substto == other.substto)


class SubstFunction(Function):
    name = 'subst'
    min_args = 3
    max_args = 3

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        s = self._arguments[0].resolvestr(makefile, variables, setting)
        r = self._arguments[1].resolvestr(makefile, variables, setting)
        d = self._arguments[2].resolvestr(makefile, variables, setting)
        fd.write(d.replace(s, r))


# noinspection SpellCheckingInspection
class PatSubstFunction(Function):
    name = 'patsubst'
    min_args = 3
    max_args = 3

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        s = self._arguments[0].resolvestr(makefile, variables, setting)
        r = self._arguments[1].resolvestr(makefile, variables, setting)

        p = data.Pattern(s)
        fd.write(' '.join([p.subst(r, word, False)
                           for word in self._arguments[2].resolvesplit(makefile, variables, setting)]))


class StripFunction(Function):
    name = 'strip'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        util.joiniter(fd, self._arguments[0].resolvesplit(makefile, variables, setting))


# noinspection SpellCheckingInspection
class FindstringFunction(Function):
    name = 'findstring'
    min_args = 2
    max_args = 2

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        s = self._arguments[0].resolvestr(makefile, variables, setting)
        r = self._arguments[1].resolvestr(makefile, variables, setting)
        if r.find(s) == -1:
            return
        fd.write(s)


class FilterFunction(Function):
    name = 'filter'
    min_args = 2
    max_args = 2

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        plist = [data.Pattern(p)
                 for p in self._arguments[0].resolvesplit(makefile, variables, setting)]

        fd.write(' '.join([w for w in self._arguments[1].resolvesplit(makefile, variables, setting)
                           if any((p.match(w) for p in plist))]))


# noinspection SpellCheckingInspection
class FilteroutFunction(Function):
    name = 'filter-out'
    min_args = 2
    max_args = 2

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        plist = [data.Pattern(p)
                 for p in self._arguments[0].resolvesplit(makefile, variables, setting)]

        fd.write(' '.join([w for w in self._arguments[1].resolvesplit(makefile, variables, setting)
                           if not any((p.match(w) for p in plist))]))


class SortFunction(Function):
    name = 'sort'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        d = set(self._arguments[0].resolvesplit(makefile, variables, setting))
        util.joiniter(fd, sorted(d))


class WordFunction(Function):
    name = 'word'
    min_args = 2
    max_args = 2

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        n = self._arguments[0].resolvestr(makefile, variables, setting)
        # TODO: provide better error if this doesn't convert
        n = int(n)
        words = list(self._arguments[1].resolvesplit(makefile, variables, setting))
        if n < 1 or n > len(words):
            return
        fd.write(words[n - 1])


# noinspection SpellCheckingInspection
class WordlistFunction(Function):
    name = 'wordlist'
    min_args = 3
    max_args = 3

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        nfrom = self._arguments[0].resolvestr(makefile, variables, setting)
        nto = self._arguments[1].resolvestr(makefile, variables, setting)
        # TODO: provide better errors if this doesn't convert
        nfrom = int(nfrom)
        nto = int(nto)

        words = list(self._arguments[2].resolvesplit(makefile, variables, setting))

        if nfrom < 1:
            nfrom = 1
        if nto < 1:
            nto = 1

        util.joiniter(fd, words[nfrom - 1:nto])


class WordsFunction(Function):
    name = 'words'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        fd.write(str(len(self._arguments[0].resolvesplit(makefile, variables, setting))))


# noinspection SpellCheckingInspection
class FirstWordFunction(Function):
    name = 'firstword'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        lst = self._arguments[0].resolvesplit(makefile, variables, setting)
        if len(lst):
            fd.write(lst[0])


# noinspection SpellCheckingInspection
class LastWordFunction(Function):
    name = 'lastword'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        lst = self._arguments[0].resolvesplit(makefile, variables, setting)
        if len(lst):
            fd.write(lst[-1])


# noinspection SpellCheckingInspection
def pathsplit(path, default='./'):
    """
    Splits a path into dirpart, filepart on the last slash. If there is no slash, dirpart
    is ./
    """
    dir_, slash, file = util.strrpartition(path, '/')
    if dir_ == '':
        return default, file

    return dir_ + slash, file


class DirFunction(Function):
    name = 'dir'
    min_args = 1
    max_args = 1

    def resolve(self, makefile, variables, fd, setting):
        fd.write(' '.join([pathsplit(path)[0]
                           for path in self._arguments[0].resolvesplit(makefile, variables, setting)]))


# noinspection SpellCheckingInspection
class NotDirFunction(Function):
    name = 'notdir'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        fd.write(' '.join([pathsplit(path)[1]
                           for path in self._arguments[0].resolvesplit(makefile, variables, setting)]))


class SuffixFunction(Function):
    name = 'suffix'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    @staticmethod
    def suffixes(words):
        for w in words:
            dir_, file = pathsplit(w)
            base, dot, suffix = util.strrpartition(file, '.')
            if base != '':
                yield dot + suffix

    def resolve(self, makefile, variables, fd, setting):
        util.joiniter(fd, self.suffixes(self._arguments[0].resolvesplit(makefile, variables, setting)))


# noinspection SpellCheckingInspection
class BasenameFunction(Function):
    name = 'basename'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    @staticmethod
    def basenames(words):
        for w in words:
            dir_, file = pathsplit(w, '')
            base, dot, suffix = util.strrpartition(file, '.')
            if dot == '':
                base = suffix

            yield dir_ + base

    def resolve(self, makefile, variables, fd, setting):
        util.joiniter(fd, self.basenames(self._arguments[0].resolvesplit(makefile, variables, setting)))


# noinspection SpellCheckingInspection
class AddSuffixFunction(Function):
    name = 'addsuffix'
    min_args = 2
    max_args = 2

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        suffix = self._arguments[0].resolvestr(makefile, variables, setting)

        fd.write(' '.join([w + suffix for w in self._arguments[1].resolvesplit(makefile, variables, setting)]))


# noinspection SpellCheckingInspection
class AddPrefixFunction(Function):
    name = 'addprefix'
    min_args = 2
    max_args = 2

    def resolve(self, makefile, variables, fd, setting):
        prefix = self._arguments[0].resolvestr(makefile, variables, setting)

        fd.write(' '.join([prefix + w for w in self._arguments[1].resolvesplit(makefile, variables, setting)]))


# noinspection SpellCheckingInspection
class JoinFunction(Function):
    name = 'join'
    min_args = 2
    max_args = 2

    __slots__ = Function.__slots__

    @staticmethod
    def iterjoin(l1, l2):
        for i in range(0, max(len(l1), len(l2))):
            i1 = i < len(l1) and l1[i] or ''
            i2 = i < len(l2) and l2[i] or ''
            yield i1 + i2

    def resolve(self, makefile, variables, fd, setting):
        list1 = list(self._arguments[0].resolvesplit(makefile, variables, setting))
        list2 = list(self._arguments[1].resolvesplit(makefile, variables, setting))

        util.joiniter(fd, self.iterjoin(list1, list2))


class WildcardFunction(Function):
    name = 'wildcard'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        patterns = self._arguments[0].resolvesplit(makefile, variables, setting)

        fd.write(' '.join([x.replace('\\', '/')
                           for p in patterns
                           for x in glob(makefile.workdir, p)]))

    @property
    def is_filesystem_dependent(self):
        return True


# noinspection SpellCheckingInspection
class RealpathFunction(Function):
    name = 'realpath'
    min_args = 1
    max_args = 1

    def resolve(self, makefile, variables, fd, setting):
        fd.write(' '.join([os.path.realpath(os.path.join(makefile.workdir, path)).replace('\\', '/')
                           for path in self._arguments[0].resolvesplit(makefile, variables, setting)]))

    def is_filesystem_dependent(self):
        return True


class AbspathFunction(Function):
    name = 'abspath'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        assert os.path.isabs(makefile.workdir)
        fd.write(' '.join([util.normaljoin(makefile.workdir, path).replace('\\', '/')
                           for path in self._arguments[0].resolvesplit(makefile, variables, setting)]))


class IfFunction(Function):
    name = 'if'
    min_args = 1
    max_args = 3

    __slots__ = Function.__slots__

    def setup(self):
        Function.setup(self)
        self._arguments[0].lstrip()
        self._arguments[0].rstrip()

    def resolve(self, makefile, variables, fd, setting):
        condition = self._arguments[0].resolvestr(makefile, variables, setting)

        if len(condition):
            self._arguments[1].resolve(makefile, variables, fd, setting)
        elif len(self._arguments) > 2:
            return self._arguments[2].resolve(makefile, variables, fd, setting)


class OrFunction(Function):
    name = 'or'
    min_args = 1
    max_args = 0

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        for arg in self._arguments:
            r = arg.resolvestr(makefile, variables, setting)
            if r != '':
                fd.write(r)
                return


class AndFunction(Function):
    name = 'and'
    min_args = 1
    max_args = 0

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        r = ''

        for arg in self._arguments:
            r = arg.resolvestr(makefile, variables, setting)
            if r == '':
                return

        fd.write(r)


class ForEachFunction(Function):
    name = 'foreach'
    min_args = 3
    max_args = 3

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        vname = self._arguments[0].resolvestr(makefile, variables, setting)
        e = self._arguments[2]

        v = data.Variables(parent=variables)
        first_word = True

        for w in self._arguments[1].resolvesplit(makefile, variables, setting):
            if first_word:
                first_word = False
            else:
                fd.write(' ')

            # The $(origin) of the local variable must be "automatic" to
            # conform with GNU make. However, automatic variables have low
            # priority. So, we must force its assignment to occur.
            v.set(vname, data.Variables.Flavor.SIMPLE, data.Variables.Source.AUTOMATIC, w, force=True)
            e.resolve(makefile, v, fd, setting)


class CallFunction(Function):
    name = 'call'
    min_args = 1
    max_args = 0

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        vname = self._arguments[0].resolvestr(makefile, variables, setting)
        if vname in setting:
            raise errors.DataError("Recursively setting variable '%s'" % (vname,))

        v = data.Variables(parent=variables)
        v.set('0', data.Variables.Flavor.SIMPLE, data.Variables.Source.AUTOMATIC, vname)
        for i in range(1, len(self._arguments)):
            param = self._arguments[i].resolvestr(makefile, variables, setting)
            v.set(str(i), data.Variables.Flavor.SIMPLE, data.Variables.Source.AUTOMATIC, param)

        flavor, source, e = variables.get(vname)

        if e is None:
            return

        if flavor == data.Variables.Flavor.SIMPLE:
            log.warning("%s: calling variable '%s' which is simply-expanded" % (self.loc, vname))

        # but we'll do it anyway
        e.resolve(makefile, v, fd, setting + [vname])


class ValueFunction(Function):
    name = 'value'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        var_name = self._arguments[0].resolvestr(makefile, variables, setting)

        flavor, source, value = variables.get(var_name, expand=False)
        if value is not None:
            fd.write(value)


class EvalFunction(Function):
    name = 'eval'
    min_args = 1
    max_args = 1

    def resolve(self, makefile, variables, _, setting):
        if makefile.parsingfinished:
            # GNU make allows variables to be set by recursive expansion during
            # command execution. This seems really dumb to me, so I don't!
            raise errors.DataError("$(eval) not allowed via recursive expansion after parsing is finished", self.loc)

        statements = parser.parsestring(self._arguments[0].resolvestr(makefile, variables, setting),
                                        'evaluation from %s' % self.loc)
        statements.execute(makefile)


class OriginFunction(Function):
    name = 'origin'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        vname = self._arguments[0].resolvestr(makefile, variables, setting)

        flavor, source, value = variables.get(vname)
        if source is None:
            r = 'undefined'
        elif source == data.Variables.Source.OVERRIDE:
            r = 'override'

        elif source == data.Variables.Source.MAKEFILE:
            r = 'file'
        elif source == data.Variables.Source.ENVIRONMENT:
            r = 'environment'
        elif source == data.Variables.Source.COMMANDLINE:
            r = 'command line'
        elif source == data.Variables.Source.AUTOMATIC:
            r = 'automatic'
        elif source == data.Variables.Source.IMPLICIT:
            r = 'default'
        else:
            raise ValueError("Unknown source: {}".format(source))

        fd.write(r)


class FlavorFunction(Function):
    name = 'flavor'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        var_name = self._arguments[0].resolvestr(makefile, variables, setting)

        flavor, source, value = variables.get(var_name)
        if flavor is None:
            r = 'undefined'
        elif flavor == data.Variables.Flavor.RECURSIVE:
            r = 'recursive'
        elif flavor == data.Variables.Flavor.SIMPLE:
            r = 'simple'
        else:
            raise ValueError("Unknown flavor {}".format(flavor))

        fd.write(r)


# noinspection SpellCheckingInspection
class ShellFunction(Function):
    name = 'shell'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, fd, setting):
        from .process import prepare_command

        cline = self._arguments[0].resolvestr(makefile, variables, setting)
        executable, cline = prepare_command(cline, makefile.workdir, self.loc)

        # subprocess.Popen doesn't use the PATH set in the env argument for
        # finding the executable on some platforms (but strangely it does on
        # others!), so set os.environ['PATH'] explicitly.
        old_path = os.environ['PATH']

        log.debug("%s: running command '%s'" % (self.loc, ' '.join(cline)))
        while True:
            if makefile.env is not None and 'PATH' in makefile.env:
                os.environ['PATH'] = makefile.env['PATH']

            try:
                p = subprocess.Popen(cline, executable=executable, env=makefile.env, shell=False,
                                     stdout=subprocess.PIPE, cwd=makefile.workdir)
                break
            except OSError as e:
                if isinstance(e, FileNotFoundError) and executable:
                    # FileNotFoundError: [WinError 2] The system cannot find the file specified
                    # Windows have trouble with the executable parameter, it shall be passed via arguments.
                    executable = None
                else:
                    print("Error executing command %s" % cline[0], e, file=sys.stderr)
                    return
            finally:
                os.environ['PATH'] = old_path

        # noinspection PyUnboundLocalVariable
        stdout, stderr = p.communicate()
        if isinstance(stdout, bytes):
            stdout = stdout.decode()
        if isinstance(stderr, bytes):
            _ = stderr.decode()
        stdout = stdout.replace('\r\n', '\n')
        if stdout.endswith('\n'):
            stdout = stdout[:-1]
        stdout = stdout.replace('\n', ' ')

        fd.write(stdout)


class ErrorFunction(Function):
    name = 'error'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, _, setting):
        v = self._arguments[0].resolvestr(makefile, variables, setting)
        raise errors.DataError(v, self.loc)


class WarningFunction(Function):
    name = 'warning'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, _, setting):
        v = self._arguments[0].resolvestr(makefile, variables, setting)
        log.warning(v)


class InfoFunction(Function):
    name = 'info'
    min_args = 1
    max_args = 1

    __slots__ = Function.__slots__

    def resolve(self, makefile, variables, _, setting):
        v = self._arguments[0].resolvestr(makefile, variables, setting)
        print(v)


# noinspection SpellCheckingInspection
functionmap = {
    'subst': SubstFunction,
    'patsubst': PatSubstFunction,
    'strip': StripFunction,
    'findstring': FindstringFunction,
    'filter': FilterFunction,
    'filter-out': FilteroutFunction,
    'sort': SortFunction,
    'word': WordFunction,
    'wordlist': WordlistFunction,
    'words': WordsFunction,
    'firstword': FirstWordFunction,
    'lastword': LastWordFunction,
    'dir': DirFunction,
    'notdir': NotDirFunction,
    'suffix': SuffixFunction,
    'basename': BasenameFunction,
    'addsuffix': AddSuffixFunction,
    'addprefix': AddPrefixFunction,
    'join': JoinFunction,
    'wildcard': WildcardFunction,
    'realpath': RealpathFunction,
    'abspath': AbspathFunction,
    'if': IfFunction,
    'or': OrFunction,
    'and': AndFunction,
    'foreach': ForEachFunction,
    'call': CallFunction,
    'value': ValueFunction,
    'eval': EvalFunction,
    'origin': OriginFunction,
    'flavor': FlavorFunction,
    'shell': ShellFunction,
    'error': ErrorFunction,
    'warning': WarningFunction,
    'info': InfoFunction,
}

# noinspection PyPep8
# from . import data
