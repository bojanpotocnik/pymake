"""
A representation of makefile data structures.
"""
import enum
import logging
import os
import re
import sys
from functools import reduce
from io import StringIO
from typing import Tuple, Dict, Optional, Mapping, Union, Iterator, List

from . import errors
from . import globrelative
from . import parserdata, parser, functions, process, util, implicit

_log = logging.getLogger('pymake.data')


def without_duplicates(it):
    r = set()
    for i in it:
        if i not in r:
            r.add(i)
            yield i


def modification_time_is_later(dependency_time, target_time):
    """
    Is the modification time of the dependency later than the target?
    """

    if dependency_time is None:
        return True
    if target_time is None:
        return False
    # int(1000*x) because of http://bugs.python.org/issue10148
    return int(1000 * dependency_time) > int(1000 * target_time)


def getmtime(path):
    try:
        s = os.stat(path)
        return s.st_mtime
    except OSError:
        return None


def strip_dot_slash(s):
    if s.startswith('./'):
        st = s[2:]
        return st if st != '' else '.'
    return s


def strip_dot_slashes(sl):
    for s in sl:
        yield strip_dot_slash(s)


def get_indent(stack):
    return ''.ljust(len(stack) - 1)


def _if_else(c, t, f):
    if c:
        return t()
    return f()


class BaseExpansion(object):
    """Base class for expansions.

    A make expansion is the parsed representation of a string, which may
    contain references to other elements.
    """

    @property
    def is_static_string(self):
        """Returns whether the expansion is composed of static string content.

        This is always True for StringExpansion. It will be True for Expansion
        only if all elements of that Expansion are static strings.
        """
        raise Exception('Must be implemented in child class.')

    def functions(self, descend=False):
        """Obtain all functions inside this expansion.

        This is a generator for pymake.functions.Function instances.

        By default, this only returns functions existing as the primary
        elements of this expansion. If `descend` is True, it will descend into
        child expansions and extract all functions in the tree.
        """
        # An empty generator. Yeah, it's weird.
        for x in []:
            yield x

    def variable_references(self, descend=False):
        """Obtain all variable references in this expansion.

        This is a generator for pymake.functionsVariableRef instances.

        To retrieve the names of variables, simply query the `vname` field on
        the returned instances. Most of the time these will be StringExpansion
        instances.
        """
        for f in self.functions(descend=descend):
            if not isinstance(f, functions.VariableRef):
                continue

            yield f

    @property
    def is_filesystem_dependent(self):
        """Whether this expansion may query the filesystem for evaluation.

        This effectively asks "is any function in this expansion dependent on
        the filesystem.
        """
        for f in self.functions(descend=True):
            if f.is_filesystem_dependent:
                return True

        return False

    @property
    def is_shell_dependent(self):
        """Whether this expansion may invoke a shell for evaluation."""

        for f in self.functions(descend=True):
            if isinstance(f, functions.ShellFunction):
                return True

        return False


class StringExpansion(BaseExpansion):
    """An Expansion representing a static string.

    This essentially wraps a single str instance.
    """

    __slots__ = ('loc', 's',)
    simple = True

    def __init__(self, s, loc):
        assert isinstance(s, str)
        self.s = s
        self.loc = loc

    # noinspection SpellCheckingInspection
    def lstrip(self):
        self.s = self.s.lstrip()

    # noinspection SpellCheckingInspection
    def rstrip(self):
        self.s = self.s.rstrip()

    def is_empty(self):
        return self.s == ''

    def resolve(self, i, j, fd, k=None):
        fd.write(self.s)

    def resolvestr(self, i, j, k=None):
        return self.s

    def resolvesplit(self, i, j, k=None):
        return self.s.split()

    def clone(self):
        e = Expansion(self.loc)
        e.appendstr(self.s)
        return e

    @property
    def is_static_string(self):
        return True

    def __len__(self):
        return 1

    def __getitem__(self, i):
        assert i == 0
        return self.s, False

    def __repr__(self):
        return "Exp<%s>(%r)" % (self.loc, self.s)

    def __eq__(self, other):
        """We only compare the string contents."""
        return self.s == other

    def __ne__(self, other):
        return not self.__eq__(other)

    def to_source(self, escape_variables=False, escape_comments=False):
        s = self.s

        if escape_comments:
            s = s.replace('#', '\\#')

        if escape_variables:
            return s.replace('$', '$$')

        return s


class Expansion(BaseExpansion, list):
    """
    A representation of expanded data.

    This is effectively an ordered list of StringExpansion and pymake.function.Function instances.
    Every item in the collection appears in the same context in a make file.
    """

    __slots__ = ('loc',)
    simple = False

    # noinspection PyMissingConstructor
    def __init__(self, loc=None):
        # A list of (element, isfunc) tuples
        # element is either a string or a function
        self.loc = loc

    def __iter__(self) -> Iterator[Tuple[Union[str, functions.Function], bool]]:
        return super().__iter__()

    @staticmethod
    def fromstring(s, path):
        return StringExpansion(s, parserdata.Location(path, 1, 0))

    def clone(self):
        e = Expansion()
        e.extend(self)
        return e

    def appendstr(self, s: str):
        assert isinstance(s, str)

        if s:
            self.append((s, False))

    def appendfunc(self, func: functions.Function):
        assert isinstance(func, functions.Function)

        self.append((func, True))

    def concat(self, o):
        """Concatenate the other expansion on to this one."""
        if o.simple:
            self.appendstr(o.s)
        else:
            self.extend(o)

    def is_empty(self):
        return (not len(self)) or self[0] == ('', False)

    def lstrip(self):
        """Strip leading literal whitespace from this expansion."""
        while True:
            i, isfunc = self[0]
            if isfunc:
                return

            i = i.lstrip()
            if i != '':
                self[0] = i, False
                return

            del self[0]

    def rstrip(self):
        """Strip trailing literal whitespace from this expansion."""
        while True:
            i, isfunc = self[-1]
            if isfunc:
                return

            i = i.rstrip()
            if i != '':
                self[-1] = i, False
                return

            del self[-1]

    def finish(self):
        # Merge any adjacent literal strings:
        strings = []
        elements = []
        for (e, isfunc) in self:
            if isfunc:
                if strings:
                    s = ''.join(strings)
                    if s:
                        elements.append((s, False))
                    strings = []
                elements.append((e, True))
            else:
                strings.append(e)

        if not elements:
            # This can only happen if there were no function elements.
            return StringExpansion(''.join(strings), self.loc)

        if strings:
            s = ''.join(strings)
            if s:
                elements.append((s, False))

        if len(elements) < len(self):
            self[:] = elements

        return self

    def resolve(self, makefile: 'Makefile', variables: 'Variables', fd: StringIO, setting: List[str] = None) -> None:
        """
        Resolve this variable into a value, by interpolating the value of other variables.

        :param makefile:  Makefile.
        :param variables: Existing variables.
        :param fd:        String file object to which string expansion can be written to.
        :param setting:   The variable currently being set, if any. Setting variables must avoid self-referential loops.
        """
        assert isinstance(makefile, Makefile)
        assert isinstance(variables, Variables)
        assert isinstance(setting, list)

        if setting is None:
            setting = []

        for e, is_func in self:
            if is_func:
                e.resolve(makefile, variables, fd, setting)
            else:
                assert isinstance(e, str)
                fd.write(e)

    def resolvestr(self, makefile: 'Makefile', variables: 'Variables', setting: List[str] = None) -> str:
        if setting is None:
            setting = []
        fd = StringIO()
        self.resolve(makefile, variables, fd, setting)
        return fd.getvalue()

    def resolvesplit(self, makefile, variables, setting: List[str] = None):
        if setting is None:
            setting = []
        return self.resolvestr(makefile, variables, setting).split()

    @property
    def is_static_string(self):
        """An Expansion is static if all its components are strings, not
        functions."""
        for e, is_func in self:
            if is_func:
                return False

        return True

    def functions(self, descend=False):
        for e, is_func in self:
            if is_func:
                yield e

            if descend:
                for exp in e.expansions(descend=True):
                    for f in exp.functions(descend=True):
                        yield f

    def __repr__(self):
        return "<Expansion with elements: %r>" % ([e for e, isfunc in self],)

    def to_source(self, escape_variables=False, escape_comments=False):
        parts = []
        for e, is_func in self:
            if is_func:
                parts.append(e.to_source())
                continue

            if escape_variables:
                parts.append(e.replace('$', '$$'))
                continue

            parts.append(e)

        return ''.join(parts)

    def __eq__(self, other):
        if not isinstance(other, (Expansion, StringExpansion)):
            return False

        # Expansions are equivalent if adjacent string literals normalize to
        # the same value. So, we must normalize before any comparisons are
        # made.
        a = self.clone().finish()

        if isinstance(other, StringExpansion):
            if isinstance(a, StringExpansion):
                return a == other

            # A normalized Expansion != StringExpansion.
            return False

        b = other.clone().finish()

        # b could be a StringExpansion now.
        if isinstance(b, StringExpansion):
            if isinstance(a, StringExpansion):
                return a == b

            # Our normalized Expansion != normalized StringExpansion.
            return False

        if len(a) != len(b):
            return False

        for i in range(len(self)):
            e1, is_func1 = a[i]
            e2, is_func2 = b[i]

            if is_func1 != is_func2:
                return False

            if type(e1) != type(e2):
                return False

            if e1 != e2:
                return False

        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class Variables:
    """
    A mapping from variable names to variables. Variables have flavor, source, and value. The value is an
    expansion object.
    """

    __slots__ = ('parent', '_map')

    @enum.unique
    class Flavor(enum.IntEnum):
        RECURSIVE = 0
        SIMPLE = 1
        APPEND = 2

        def __bool__(self) -> bool:
            return True

    @enum.unique
    class Source(enum.IntEnum):
        """Sources of the variables. Values represent priorities (lower value means higher priority)."""
        OVERRIDE = 0
        COMMANDLINE = 1
        MAKEFILE = 2
        ENVIRONMENT = 3
        AUTOMATIC = 4
        IMPLICIT = 5

        def __bool__(self) -> bool:
            return True

    def __init__(self, parent: 'Variables' = None):
        self._map: Dict[str, Tuple[self.Flavor, self.Source, str, Optional[Union[Expansion, StringExpansion]]]] = {}
        """vname -> flavor, source, valuestr, valueexp"""
        self.parent: 'Variables' = parent

    def read_from_environment(self, env: Mapping[str, str]) -> 'Variables':
        """
        Copy environment variables to the Makefile variables.

        :param env: Mapping of the environment variables.
        """
        for k, v in env.items():
            self.set(k, self.Flavor.RECURSIVE, self.Source.ENVIRONMENT, v)
        return self

    def from_statements(self, statements: parserdata.StatementList) -> 'Variables':
        for stmt in statements:
            if not isinstance(stmt, parserdata.SetVariable):
                continue
            # https://www.gnu.org/software/make/manual/html_node/Setting.html
            if stmt.token == "+=":
                self.append(stmt.name, stmt.source, stmt.value, self, None)
            elif stmt.token in ("?=", ":=", "::=", "="):
                flavor = self.Flavor.SIMPLE if (stmt.token in (":=", "::=")) else self.Flavor.RECURSIVE

                if (stmt.token != "?=") or (stmt.name not in self):
                    self.set(stmt.name, flavor, stmt.source, stmt.value)
            else:
                raise ValueError(f"Unknown token '{stmt.token}'")
        return self

    def get(self, name: str, expand: bool = True) \
            -> Tuple[Optional[Flavor], Optional[Source], Optional[Union[str, Expansion, StringExpansion]]]:
        """
        Get the value of a named variable. Returns a tuple (flavor, source, value)

        If the variable is not present, returns (None, None, None)

        :param name:   Variable name.
        :param expand: If true, the value will be returned as an expansion.
                       If false, it will be returned as an unexpanded string.
        """
        flavor, source, value_str, value_exp = self._map.get(name, (None, None, None, None))
        if flavor is not None:
            if expand and (flavor != self.Flavor.SIMPLE) and (value_exp is None):
                d = parser.Data.fromstring(value_str, parserdata.Location(f"Expansion of variable '{name}'", 1, 0))
                value_exp, t, o = parser.parse_make_syntax(d, 0, (), parser.iterdata)
                self._map[name] = flavor, source, value_str, value_exp

            if flavor == self.Flavor.APPEND:
                if self.parent:
                    parent_flavor, parent_source, parent_value = self.parent.get(name, expand)
                else:
                    parent_flavor, parent_source, parent_value = None, None, None

                if parent_value is None:
                    flavor = self.Flavor.RECURSIVE
                    # Fall through
                else:
                    if source > parent_source:
                        # TODO: log a warning?
                        return parent_flavor, parent_source, parent_value

                    if not expand:
                        return parent_flavor, parent_source, parent_value + ' ' + value_str

                    parent_value = parent_value.clone()
                    parent_value.appendstr(' ')
                    parent_value.concat(value_exp)

                    return parent_flavor, parent_source, parent_value

            if not expand:
                return flavor, source, value_str

            if flavor == self.Flavor.RECURSIVE:
                val = value_exp
            else:
                val = Expansion.fromstring(value_str, f"Expansion of variable '{name}'")

            return flavor, source, val

        if self.parent is not None:
            return self.parent.get(name, expand)

        return None, None, None

    def set(self, name: str, flavor: Flavor, source: Source, value: str, *, force: bool = False) -> None:
        """
        Set variable data in the map of variables.
        If the new variable source has higher priority (lower value) than the already existing variable (if any)
        then the existing variable is automatically overwritten, otherwise `force` flag decides.

        :param name:   Name of the variable.
        :param flavor: Variable flavor.
        :param source: Variable source.
        :param value:  Value of the variable.
        :param force:  Whether to forcibly overwrite existing variable (if any) even if the source priority
                       of the new variable is lower than the one of the existing variable.
        """
        assert flavor in (self.Flavor.RECURSIVE, self.Flavor.SIMPLE)  # Flavor.APPEND is not allowed here.
        # All sources are allowed.
        assert isinstance(value, str), "expected str, got %s" % type(value)

        prev_flavor, prev_source, prev_value = self.get(name)
        if (prev_source is not None) and (source > prev_source) and (not force):
            # TODO: give a location for this warning
            _log.info(f"Not setting variable '{name}' (source {source}),"
                      f" set by higher-priority source {prev_source} to value '{prev_value}'")
            return

        self._map[name] = flavor, source, value, None

    def append(self, name: str, source: Source, value: str,
               variables: Optional['Variables'], makefile: Optional['Makefile']) -> None:
        """
        Append variable value to already existing variable or create a new one if it doesnt exist.
        If the appending variable source has lower priority than the already existing variable (if any) then
        this action is ignored.

        :param name:      Name of the variable.
        :param source:    Variable source.
        :param value:     Value of the variable.
        :param variables: Existing variables.
        :param makefile:  Makefile context.
        """
        assert source in (self.Source.OVERRIDE, self.Source.MAKEFILE, self.Source.AUTOMATIC)
        assert isinstance(value, str)

        if name not in self._map:
            self._map[name] = self.Flavor.APPEND, source, value, None
            return

        prev_flavor, prev_source, prev_value, prev_value_exp = self._map[name]
        if source > prev_source:
            _log.info(f"Not appending variable '{name}' (source {source}, value '{value}'),"
                      f" set by higher-priority source {prev_source} to value '{prev_value}'")
            return

        if prev_flavor == self.Flavor.SIMPLE:
            d = parser.Data.fromstring(value, parserdata.Location(f"Expansion of variable '{name}'", 1, 0))
            prev_value_exp, t, o = parser.parse_make_syntax(d, 0, (), parser.iterdata)

            val = prev_value_exp.resolvestr(makefile, variables, [name])
            self._map[name] = prev_flavor, prev_source, prev_value + ' ' + val, None
            return

        new_value = prev_value + ' ' + value
        self._map[name] = prev_flavor, prev_source, new_value, None

    def merge(self, other: 'Variables'):
        assert isinstance(other, Variables)

        for k, flavor, source, value in other:
            self.set(k, flavor, source, value)

    def __iter__(self) -> Iterator[Tuple[str, Flavor, Source, str]]:
        for k, (flavor, source, value, value_exp) in self._map.items():
            yield k, flavor, source, value

    def __contains__(self, name: str) -> bool:
        return name in self._map

    def __str__(self) -> str:
        vars_ = []
        for name, (flavor, source, value, value_exp) in self._map.items():
            vars_.append(f"{name}<{flavor.name, source.name}>={value} ({value_exp})")
        return f"{type(self).__name__}({', '.join(vars_)})"


class Pattern(object):
    """
    A pattern is a string, possibly with a % substitution character. From the GNU make manual:

    '%' characters in pattern rules can be quoted with precending backslashes ('\'). Backslashes that
    would otherwise quote '%' charcters can be quoted with more backslashes. Backslashes that
    quote '%' characters or other backslashes are removed from the pattern before it is compared t
    file names or has a stem substituted into it. Backslashes that are not in danger of quoting '%'
    characters go unmolested. For example, the pattern the\%weird\\%pattern\\ has `the%weird\' preceding
    the operative '%' character, and 'pattern\\' following it. The final two backslashes are left alone
    because they cannot affect any '%' character.

    This insane behavior probably doesn't matter, but we're compatible just for shits and giggles.
    """

    __slots__ = 'data'

    def __init__(self, s):
        r = []
        i = 0
        slen = len(s)
        while i < slen:
            c = s[i]
            if c == '\\':
                nc = s[i + 1]
                if nc == '%':
                    r.append('%')
                    i += 1
                elif nc == '\\':
                    r.append('\\')
                    i += 1
                else:
                    r.append(c)
            elif c == '%':
                self.data = (''.join(r), s[i + 1:])
                return
            else:
                r.append(c)
            i += 1

        # This is different than (s,) because \% and \\ have been unescaped. Parsing patterns is
        # context-sensitive!
        self.data = (''.join(r),)

    def ismatchany(self):
        return self.data == ('', '')

    def ispattern(self):
        return len(self.data) == 2

    def __hash__(self):
        return self.data.__hash__()

    def __eq__(self, o):
        assert isinstance(o, Pattern)
        return self.data == o.data

    def gettarget(self):
        assert not self.ispattern()
        return self.data[0]

    def hasslash(self):
        return self.data[0].find('/') != -1 or self.data[1].find('/') != -1

    def match(self, word):
        """
        Match this search pattern against a word (string).

        @returns None if the word doesn't match, or the matching stem.
                      If this is a %-less pattern, the stem will always be ''
        """
        d = self.data
        if len(d) == 1:
            if word == d[0]:
                return word
            return None
        elif len(d) != 2:
            raise ValueError(d)

        # noinspection PyTupleAssignmentBalance
        d0, d1 = d
        l1 = len(d0)
        l2 = len(d1)
        if len(word) >= l1 + l2 and word.startswith(d0) and word.endswith(d1):
            if l2 == 0:
                return word[l1:]
            return word[l1:-l2]

        return None

    def resolve(self, dir_, stem):
        if self.ispattern():
            return dir_ + self.data[0] + stem + self.data[1]

        return self.data[0]

    def subst(self, replacement, word, mustmatch):
        """
        Given a word, replace the current pattern with the replacement pattern, a la 'patsubst'

        @param mustmatch If true and this pattern doesn't match the word, throw a DataError. Otherwise
                         return word unchanged.
        """
        assert isinstance(replacement, str)

        stem = self.match(word)
        if stem is None:
            if mustmatch:
                raise errors.DataError("target '%s' doesn't match pattern" % (word,))
            return word

        if not self.ispattern():
            # if we're not a pattern, the replacement is not parsed as a pattern either
            return replacement

        return Pattern(replacement).resolve('', stem)

    def __repr__(self):
        return "<Pattern with data %r>" % (self.data,)

    _backre = re.compile(r'[%\\]')

    def __str__(self):
        if not self.ispattern():
            return self._backre.sub(r'\\\1', self.data[0])

        return self._backre.sub(r'\\\1', self.data[0]) + '%' + self.data[1]


class RemakeTargetSerially(object):
    __slots__ = ('target', 'makefile', 'indent', 'rlist')

    def __init__(self, target, makefile, indent, rlist):
        self.target = target
        self.makefile = makefile
        self.indent = indent
        self.rlist = rlist
        self.commandscb(False)

    def resolvecb(self, error, didanything):
        assert error in (True, False)

        if didanything:
            self.target.didanything = True

        if error:
            self.target.error = True
            self.makefile.error = True
            if not self.makefile.keepgoing:
                self.target.notifydone(self.makefile)
                return
            else:
                # don't run the commands!
                del self.rlist[0]
                self.commandscb(error=False)
        else:
            self.rlist.pop(0).runcommands(self.indent, self.commandscb)

    def commandscb(self, error):
        assert error in (True, False)

        if error:
            self.target.error = True
            self.makefile.error = True

        if self.target.error and not self.makefile.keepgoing:
            self.target.notifydone(self.makefile)
            return

        if not len(self.rlist):
            self.target.notifydone(self.makefile)
        else:
            self.rlist[0].resolvedeps(True, self.resolvecb)


class RemakeTargetParallel(object):
    __slots__ = ('target', 'makefile', 'indent', 'rlist', 'rulesremaining', 'currunning')

    def __init__(self, target, makefile, indent, rlist):
        self.target = target
        self.makefile = makefile
        self.indent = indent
        self.rlist = rlist

        self.rulesremaining = len(rlist)
        self.currunning = False

        for r in rlist:
            makefile.context.defer(self.doresolve, r)

    def doresolve(self, r):
        if self.makefile.error and not self.makefile.keepgoing:
            r.error = True
            self.resolvecb(True, False)
        else:
            r.resolvedeps(False, self.resolvecb)

    def resolvecb(self, error, didanything):
        assert error in (True, False)

        if error:
            self.target.error = True

        if didanything:
            self.target.didanything = True

        self.rulesremaining -= 1

        # commandscb takes care of the details if we're currently building
        # something
        if self.currunning:
            return

        self.runnext()

    def runnext(self):
        assert not self.currunning

        if self.makefile.error and not self.makefile.keepgoing:
            self.rlist = []
        else:
            while len(self.rlist) and self.rlist[0].error:
                del self.rlist[0]

        if not len(self.rlist):
            if not self.rulesremaining:
                self.target.notifydone(self.makefile)
            return

        if self.rlist[0].depsremaining != 0:
            return

        self.currunning = True
        rule = self.rlist.pop(0)
        self.makefile.context.defer(rule.runcommands, self.indent, self.commandscb)

    def commandscb(self, error):
        assert error in (True, False)
        if error:
            self.target.error = True
            self.makefile.error = True

        assert self.currunning
        self.currunning = False
        self.runnext()


class RemakeRuleContext(object):
    def __init__(self, target, makefile, rule, deps,
                 targetstack, avoidremakeloop):
        self.target = target
        self.makefile = makefile
        self.rule = rule
        self.deps = deps
        self.targetstack = targetstack
        self.avoidremakeloop = avoidremakeloop

        self.running = False
        self.error = False
        self.depsremaining = len(deps) + 1
        self.remake = False

    def resolvedeps(self, serial, cb):
        self.resolvecb = cb
        self.didanything = False
        if serial:
            self._resolvedepsserial()
        else:
            self._resolvedepsparallel()

    def _weakdepfinishedserial(self, error, didanything):
        if error:
            self.remake = True
        self._depfinishedserial(False, didanything)

    def _depfinishedserial(self, error, didanything):
        assert error in (True, False)

        if didanything:
            self.didanything = True

        if error:
            self.error = True
            if not self.makefile.keepgoing:
                self.resolvecb(error=True, didanything=self.didanything)
                return

        if len(self.resolvelist):
            dep, weak = self.resolvelist.pop(0)
            self.makefile.context.defer(dep.make,
                                        self.makefile, self.targetstack,
                                        weak and self._weakdepfinishedserial or self._depfinishedserial)
        else:
            self.resolvecb(error=self.error, didanything=self.didanything)

    def _resolvedepsserial(self):
        self.resolvelist = list(self.deps)
        self._depfinishedserial(False, False)

    def _startdepparallel(self, d):
        dep, weak = d
        if weak:
            depfinished = self._weakdepfinishedparallel
        else:
            depfinished = self._depfinishedparallel
        if self.makefile.error:
            depfinished(True, False)
        else:
            dep.make(self.makefile, self.targetstack, depfinished)

    def _weakdepfinishedparallel(self, error, didanything):
        if error:
            self.remake = True
        self._depfinishedparallel(False, didanything)

    def _depfinishedparallel(self, error, didanything):
        assert error in (True, False)

        if error:
            print("<%s>: Found error" % self.target.target)
            self.error = True
        if didanything:
            self.didanything = True

        self.depsremaining -= 1
        if self.depsremaining == 0:
            self.resolvecb(error=self.error, didanything=self.didanything)

    def _resolvedepsparallel(self):
        self.depsremaining -= 1
        if self.depsremaining == 0:
            self.resolvecb(error=self.error, didanything=self.didanything)
            return

        self.didanything = False

        for d in self.deps:
            self.makefile.context.defer(self._startdepparallel, d)

    def _commandcb(self, error):
        assert error in (True, False)

        if error:
            self.runcb(error=True)
            return

        if len(self.commands):
            self.commands.pop(0)(self._commandcb)
        else:
            self.runcb(error=False)

    def runcommands(self, indent, cb):
        assert not self.running
        self.running = True

        self.runcb = cb

        if self.rule is None or not len(self.rule.commands):
            if self.target.mtime is None:
                self.target.beingremade()
            else:
                for d, weak in self.deps:
                    if modification_time_is_later(d.mtime, self.target.mtime):
                        if d.mtime is None:
                            self.target.beingremade()
                        else:
                            _log.info("%sNot remaking %s ubecause it would have no effect, even though %s is newer.",
                                      indent, self.target.target, d.target)
                        break
            cb(error=False)
            return

        if self.rule.doublecolon:
            if len(self.deps) == 0:
                if self.avoidremakeloop:
                    _log.info("%sNot remaking %s using rule at %s because it would introduce an infinite loop.", indent,
                              self.target.target, self.rule.loc)
                    cb(error=False)
                    return

        remake = self.remake
        if remake:
            _log.info("%sRemaking %s using rule at %s: weak dependency was not found.", indent, self.target.target,
                      self.rule.loc)
        else:
            if self.target.mtime is None:
                remake = True
                _log.info("%sRemaking %s using rule at %s: target doesn't exist or is a forced target", indent,
                          self.target.target, self.rule.loc)

        if not remake:
            if self.rule.doublecolon:
                if len(self.deps) == 0:
                    _log.info(
                        ("%sRemaking %s using rule at %s because there are"
                         " no prerequisites listed for a double-colon rule."),
                        indent, self.target.target, self.rule.loc)
                    remake = True

        if not remake:
            for d, weak in self.deps:
                if modification_time_is_later(d.mtime, self.target.mtime):
                    _log.info("%sRemaking %s using rule at %s because %s is newer.", indent, self.target.target,
                              self.rule.loc, d.target)
                    remake = True
                    break

        if remake:
            self.target.beingremade()
            self.target.didanything = True
            try:
                self.commands = [c for c in self.rule.getcommands(self.target, self.makefile)]
            except errors.MakeError as e:
                print(e)
                sys.stdout.flush()
                cb(error=True)
                return

            self._commandcb(False)
        else:
            cb(error=False)


MAKESTATE_NONE = 0
MAKESTATE_FINISHED = 1
MAKESTATE_WORKING = 2


class Target(object):
    """
    An actual (non-pattern) target.

    It holds target-specific variables and a list of rules. It may also point to a parent
    PatternTarget, if this target is being created by an implicit rule.

    The rules associated with this target may be Rule instances or, in the case of static pattern
    rules, PatternRule instances.
    """

    wasremade = False

    def __init__(self, target, makefile):
        assert isinstance(target, str)
        self.target = target
        self.vpathtarget = None
        self.rules = []
        self.variables = Variables(makefile.variables)
        self.explicit = False
        self._state = MAKESTATE_NONE

    def addrule(self, rule):
        assert isinstance(rule, (Rule, PatternRuleInstance))
        if len(self.rules) and rule.doublecolon != self.rules[0].doublecolon:
            raise errors.DataError(
                "Cannot have single- and double-colon rules for the same target. Prior rule location: %s" % self.rules[
                    0].loc, rule.loc)

        if isinstance(rule, PatternRuleInstance):
            if len(rule.prule.targetpatterns) != 1:
                raise errors.DataError("Static pattern rules must only have one target pattern", rule.prule.loc)
            if rule.prule.targetpatterns[0].match(self.target) is None:
                raise errors.DataError("Static pattern rule doesn't match target '%s'" % self.target, rule.loc)

        self.rules.append(rule)

    def isdoublecolon(self):
        return self.rules[0].doublecolon

    def isphony(self, makefile):
        """Is this a phony target? We don't check for existence of phony targets."""
        return makefile.gettarget('.PHONY').hasdependency(self.target)

    def hasdependency(self, t):
        for rule in self.rules:
            if t in rule.prerequisites:
                return True

        return False

    def resolveimplicitrule(self, makefile, targetstack, rulestack):
        """
        Try to resolve an implicit rule to build this target.
        """
        # The steps in the GNU make manual Implicit-Rule-Search.html are very detailed. I hope they can be trusted.

        indent = get_indent(targetstack)

        _log.info("%sSearching for implicit rule to make '%s'", indent, self.target)

        dir_, s, file = util.strrpartition(self.target, '/')
        dir_ = dir_ + s

        candidates = []  # list of PatternRuleInstance

        hasmatch = any((r.hasspecificmatch(file) for r in makefile.implicitrules))

        for r in makefile.implicitrules:
            if r in rulestack:
                _log.info("%s %s: Avoiding implicit rule recursion", indent, r.loc)
                continue

            if not len(r.commands):
                continue

            for ri in r.matchesfor(dir_, file, hasmatch):
                candidates.append(ri)

        newcandidates = []

        for r in candidates:
            depfailed = None
            for p in r.prerequisites:
                t = makefile.gettarget(p)
                t.resolvevpath(makefile)
                if not t.explicit and t.mtime is None:
                    depfailed = p
                    break

            if depfailed is not None:
                if r.doublecolon:
                    _log.info(
                        "%s Terminal rule at %s doesn't match: prerequisite '%s' not mentioned and doesn't exist.",
                        indent, r.loc, depfailed)
                else:
                    newcandidates.append(r)
                continue

            _log.info("%sFound implicit rule at %s for target '%s'", indent, r.loc, self.target)
            self.rules.append(r)
            return

        # Try again, but this time with chaining and without terminal (double-colon) rules

        for r in newcandidates:
            newrulestack = rulestack + [r.prule]

            depfailed = None
            for p in r.prerequisites:
                t = makefile.gettarget(p)
                try:
                    t.resolvedeps(makefile, targetstack, newrulestack, True)
                except errors.ResolutionError:
                    depfailed = p
                    break

            if depfailed is not None:
                _log.info("%s Rule at %s doesn't match: prerequisite '%s' could not be made.", indent, r.loc, depfailed)
                continue

            _log.info("%sFound implicit rule at %s for target '%s'", indent, r.loc, self.target)
            self.rules.append(r)
            return

        _log.info("%sCouldn't find implicit rule to remake '%s'", indent, self.target)

    def ruleswithcommands(self):
        """The number of rules with commands"""
        return reduce(lambda i, rule: i + (len(rule.commands) > 0), self.rules, 0)

    def resolvedeps(self, makefile, targetstack, rulestack, recursive):
        """
        Resolve the actual path of this target, using vpath if necessary.

        Recursively resolve dependencies of this target. This means finding implicit
        rules which match the target, if appropriate.

        Figure out whether this target needs to be rebuild, and set self.outofdate
        appropriately.

        @param targetstack is the current stack of dependencies being resolved. If
               this target is already in targetstack, bail to prevent infinite
               recursion.
        @param rulestack is the current stack of implicit rules being used to resolve
               dependencies. A rule chain cannot use the same implicit rule twice.
        """
        assert makefile.parsingfinished

        if self.target in targetstack:
            raise errors.ResolutionError("Recursive dependency: %s -> %s" % (
                " -> ".join(targetstack), self.target))

        targetstack += [self.target]

        indent = get_indent(targetstack)

        _log.info("%sConsidering target '%s'", indent, self.target)

        self.resolvevpath(makefile)

        # Sanity-check our rules. If we're single-colon, only one rule should have commands
        ruleswithcommands = self.ruleswithcommands()
        if len(self.rules) and not self.isdoublecolon():
            if ruleswithcommands > 1:
                # In GNU make this is a warning, not an error. I'm going to be stricter.
                # TODO: provide locations
                raise errors.DataError("Target '%s' has multiple rules with commands." % self.target)

        if ruleswithcommands == 0:
            self.resolveimplicitrule(makefile, targetstack, rulestack)

        # If a target is mentioned, but doesn't exist, has no commands and no
        # prerequisites, it is special and exists just to say that targets which
        # depend on it are always out of date. This is like .FORCE but more
        # compatible with other makes.
        # Otherwise, we don't know how to make it.
        if not len(self.rules) and self.mtime is None and not any((len(rule.prerequisites) > 0
                                                                   for rule in self.rules)):
            raise errors.ResolutionError("No rule to make target '%s' needed by %r" % (self.target,
                                                                                       targetstack))

        if recursive:
            for r in self.rules:
                newrulestack = rulestack + [r]
                for d in r.prerequisites:
                    dt = makefile.gettarget(d)
                    if dt.explicit:
                        continue

                    dt.resolvedeps(makefile, targetstack, newrulestack, True)

        for v in makefile.getpatternvariablesfor(self.target):
            self.variables.merge(v)

    def resolvevpath(self, makefile):
        if self.vpathtarget is not None:
            return

        if self.isphony(makefile):
            self.vpathtarget = self.target
            self.mtime = None
            return

        if self.target.startswith('-l'):
            stem = self.target[2:]
            f, s, e = makefile.variables.get('.LIBPATTERNS')
            if e is not None:
                libpatterns = [Pattern(strip_dot_slash(s)) for s in e.resolvesplit(makefile, makefile.variables)]
                if len(libpatterns):
                    searchdirs = ['']
                    searchdirs.extend(makefile.getvpath(self.target))

                    for lp in libpatterns:
                        if not lp.ispattern():
                            raise errors.DataError('.LIBPATTERNS contains a non-pattern')

                        libname = lp.resolve('', stem)

                        for dir_ in searchdirs:
                            libpath = util.normaljoin(dir_, libname).replace('\\', '/')
                            fspath = util.normaljoin(makefile.workdir, libpath)
                            mtime = getmtime(fspath)
                            if mtime is not None:
                                self.vpathtarget = libpath
                                self.mtime = mtime
                                return

                    self.vpathtarget = self.target
                    self.mtime = None
                    return

        search = [self.target]
        if not os.path.isabs(self.target):
            search += [util.normaljoin(dir_, self.target).replace('\\', '/')
                       for dir_ in makefile.getvpath(self.target)]

        targetandtime = self.searchinlocs(makefile, search)
        if targetandtime is not None:
            (self.vpathtarget, self.mtime) = targetandtime
            return

        self.vpathtarget = self.target
        self.mtime = None

    def searchinlocs(self, makefile, locs):
        """
        Look in the given locations relative to the makefile working directory
        for a file. Return a pair of the target and the mtime if found, None
        if not.
        """
        for t in locs:
            fspath = util.normaljoin(makefile.workdir, t).replace('\\', '/')
            mtime = getmtime(fspath)
            #            _log.info("Searching %s ... checking %s ... mtime %r" % (t, fspath, mtime))
            if mtime is not None:
                return t, mtime

        return None

    def beingremade(self):
        """
        When we remake ourself, we have to drop any vpath prefixes.
        """
        self.vpathtarget = self.target
        self.wasremade = True

    def notifydone(self, makefile):
        assert self._state == MAKESTATE_WORKING, "State was %s" % self._state
        # If we were remade then resolve mtime again
        if self.wasremade:
            targetandtime = self.searchinlocs(makefile, [self.target])
            if targetandtime is not None:
                (_, self.mtime) = targetandtime
            else:
                self.mtime = None

        self._state = MAKESTATE_FINISHED
        for cb in self._callbacks:
            makefile.context.defer(cb, error=self.error, didanything=self.didanything)
        del self._callbacks

    def make(self, makefile, targetstack, cb, avoidremakeloop=False, printerror=True):
        """
        If we are out of date, asynchronously make ourself. This is a multi-stage process, mostly handled
        by the helper objects RemakeTargetSerially, RemakeTargetParallel,
        RemakeRuleContext. These helper objects should keep us from developing
        any cyclical dependencies.

        * resolve dependencies (synchronous)
        * gather a list of rules to execute and related dependencies (synchronous)
        * for each rule (in parallel)
        ** remake dependencies (asynchronous)
        ** build list of commands to execute (synchronous)
        ** execute each command (asynchronous)
        * asynchronously notify when all rules are complete

        @param cb A callback function to notify when remaking is finished. It is called
               thusly: callback(error=True/False, didanything=True/False)
               If there is no asynchronous activity to perform, the callback may be called directly.
        """

        serial = makefile.context.job_count == 1

        if self._state == MAKESTATE_FINISHED:
            cb(error=self.error, didanything=self.didanything)
            return

        if self._state == MAKESTATE_WORKING:
            assert not serial
            self._callbacks.append(cb)
            return

        assert self._state == MAKESTATE_NONE

        self._state = MAKESTATE_WORKING
        self._callbacks = [cb]
        self.error = False
        self.didanything = False

        indent = get_indent(targetstack)

        try:
            self.resolvedeps(makefile, targetstack, [], False)
        except errors.MakeError as e:
            if printerror:
                print(e)
            self.error = True
            self.notifydone(makefile)
            return

        assert self.vpathtarget is not None, "Target was never resolved!"
        if not len(self.rules):
            self.notifydone(makefile)
            return

        if self.isdoublecolon():
            rulelist = [RemakeRuleContext(self, makefile, r, [(makefile.gettarget(p), False) for p in r.prerequisites],
                                          targetstack, avoidremakeloop) for r in self.rules]
        else:
            alldeps = []

            commandrule = None
            for r in self.rules:
                rdeps = [(makefile.gettarget(p), r.weakdeps) for p in r.prerequisites]
                if len(r.commands):
                    assert commandrule is None
                    commandrule = r
                    # The dependencies of the command rule are resolved before other dependencies,
                    # no matter the ordering of the other no-command rules
                    alldeps[0:0] = rdeps
                else:
                    alldeps.extend(rdeps)

            rulelist = [RemakeRuleContext(self, makefile, commandrule, alldeps, targetstack, avoidremakeloop)]

        targetstack += [self.target]

        if serial:
            RemakeTargetSerially(self, makefile, indent, rulelist)
        else:
            RemakeTargetParallel(self, makefile, indent, rulelist)


def dirpart(p):
    d, s, f = util.strrpartition(p, '/')
    if d == '':
        return '.'

    return d


def filepart(p):
    d, s, f = util.strrpartition(p, '/')
    return f


def setautomatic(v, name, plist):
    v.set(name, Variables.Flavor.SIMPLE, Variables.Source.AUTOMATIC, ' '.join(plist))
    v.set(name + 'D', Variables.Flavor.SIMPLE, Variables.Source.AUTOMATIC, ' '.join((dirpart(p) for p in plist)))
    v.set(name + 'F', Variables.Flavor.SIMPLE, Variables.Source.AUTOMATIC, ' '.join((filepart(p) for p in plist)))


def setautomaticvariables(v, makefile, target, prerequisites):
    prtargets = [makefile.gettarget(p) for p in prerequisites]
    prall = [pt.vpathtarget for pt in prtargets]
    proutofdate = [pt.vpathtarget for pt in without_duplicates(prtargets)
                   if target.mtime is None or modification_time_is_later(pt.mtime, target.mtime)]

    setautomatic(v, '@', [target.vpathtarget])
    if len(prall):
        setautomatic(v, '<', [prall[0]])

    setautomatic(v, '?', proutofdate)
    setautomatic(v, '^', list(without_duplicates(prall)))
    setautomatic(v, '+', prall)


def splitcommand(command):
    """
    Using the esoteric rules, split command lines by unescaped newlines.
    """
    start = 0
    i = 0
    while i < len(command):
        c = command[i]
        if c == '\\':
            i += 1
        elif c == '\n':
            yield command[start:i]
            i += 1
            start = i
            continue

        i += 1

    if i > start:
        yield command[start:i]


def findmodifiers(command):
    """
    Find any of +-@% prefixed on the command.
    @returns (command, is_hidden, is_recursive, ignore_errors, is_native)
    """

    is_hidden = False
    is_recursive = False
    ignore_errors = False
    is_native = False

    real_command = command.lstrip(' \t\n@+-%')
    mod_set = set(command[:-len(real_command)])
    return real_command, '@' in mod_set, '+' in mod_set, '-' in mod_set, '%' in mod_set


class _CommandWrapper(object):
    def __init__(self, cline, ignore_errors, loc, context, **kwargs):
        self.ignoreErrors = ignore_errors
        self.loc = loc
        self.cline = cline
        self.kwargs = kwargs
        self.context = context

    def _cb(self, res):
        if res != 0 and not self.ignoreErrors:
            print("%s: command '%s' failed, return code %i" % (self.loc, self.cline, res))
            self.usercb(error=True)
        else:
            self.usercb(error=False)

    def __call__(self, cb):
        self.usercb = cb
        process.call(self.cline, loc=self.loc, cb=self._cb, context=self.context, **self.kwargs)


class _NativeWrapper(_CommandWrapper):
    def __init__(self, cline, ignore_errors, loc, context, py_command_path, **kwargs):
        _CommandWrapper.__init__(self, cline, ignore_errors, loc, context,
                                 **kwargs)
        if py_command_path:
            self.pycommandpath = re.split('[%s\s]+' % os.pathsep,
                                          py_command_path)
        else:
            self.pycommandpath = None

    def __call__(self, cb):
        # get the module and method to call
        parts, badchar = process.clinetoargv(self.cline, self.kwargs['cwd'])
        if parts is None:
            raise errors.DataError(
                "native command '%s': shell metacharacter '%s' in command line" % (self.cline, badchar), self.loc)
        if len(parts) < 2:
            raise errors.DataError("native command '%s': no method name specified" % self.cline, self.loc)
        module = parts[0]
        method = parts[1]
        cline_list = parts[2:]
        self.usercb = cb
        process.call_native(module, method, cline_list,
                            loc=self.loc, cb=self._cb, context=self.context,
                            pycommandpath=self.pycommandpath, **self.kwargs)


def getcommandsforrule(rule, target, makefile, prerequisites, stem):
    v = Variables(parent=target.variables)
    setautomaticvariables(v, makefile, target, prerequisites)
    if stem is not None:
        setautomatic(v, '*', [stem])

    env = makefile.getsubenvironment(v)

    for c in rule.commands:
        cstring = c.resolvestr(makefile, v)
        for cline in splitcommand(cstring):
            cline, is_hidden, is_recursive, ignore_errors, is_native = findmodifiers(cline)
            if (is_hidden or makefile.silent) and not makefile.justprint:
                echo = None
            else:
                echo = "%s$ %s" % (c.loc, cline)
            if not is_native:
                yield _CommandWrapper(cline, ignore_errors=ignore_errors, env=env, cwd=makefile.workdir, loc=c.loc,
                                      context=makefile.context,
                                      echo=echo, justprint=makefile.justprint)
            else:
                f, s, e = v.get("PYCOMMANDPATH", True)
                if e:
                    e = e.resolvestr(makefile, v, ["PYCOMMANDPATH"])
                yield _NativeWrapper(cline, ignore_errors=ignore_errors,
                                     env=env, cwd=makefile.workdir,
                                     loc=c.loc, context=makefile.context,
                                     echo=echo, justprint=makefile.justprint,
                                     py_command_path=e)


class Rule(object):
    """
    A rule contains a list of prerequisites and a list of commands. It may also
    contain rule-specific variables. This rule may be associated with multiple targets.
    """

    def __init__(self, prereqs, doublecolon, loc, weakdeps):
        self.prerequisites = prereqs
        self.doublecolon = doublecolon
        self.commands = []
        self.loc = loc
        self.weakdeps = weakdeps

    def addcommand(self, c):
        assert isinstance(c, (Expansion, StringExpansion))
        self.commands.append(c)

    def getcommands(self, target, makefile):
        assert isinstance(target, Target)
        # Prerequisites are merged if the target contains multiple rules and is
        # not a terminal (double colon) rule. See
        # https://www.gnu.org/software/make/manual/make.html#Multiple-Targets.
        prereqs = []
        prereqs.extend(self.prerequisites)

        if not self.doublecolon:
            for rule in target.rules:
                # The current rule comes first, which is already in prereqs so
                # we don't need to add it again.
                if rule != self:
                    prereqs.extend(rule.prerequisites)

        return getcommandsforrule(self, target, makefile, prereqs, stem=None)
        # TODO: $* in non-pattern rules?


class PatternRuleInstance(object):
    weakdeps = False

    """
    A pattern rule instantiated for a particular target. It has the same API as Rule, but
    different internals, forwarding most information on to the PatternRule.
    """

    def __init__(self, prule, dir_, stem, ismatchany):
        assert isinstance(prule, PatternRule)

        self.dir = dir_
        self.stem = stem
        self.prule = prule
        self.prerequisites = prule.prerequisitesforstem(dir_, stem)
        self.doublecolon = prule.doublecolon
        self.loc = prule.loc
        self.ismatchany = ismatchany
        self.commands = prule.commands

    def getcommands(self, target, makefile):
        assert isinstance(target, Target)
        return getcommandsforrule(self, target, makefile, self.prerequisites, stem=self.dir + self.stem)

    def __str__(self):
        return "Pattern rule at %s with stem '%s', matchany: %s doublecolon: %s" % (self.loc,
                                                                                    self.dir + self.stem,
                                                                                    self.ismatchany,
                                                                                    self.doublecolon)


class PatternRule(object):
    """
    An implicit rule or static pattern rule containing target patterns, prerequisite patterns,
    and a list of commands.
    """

    def __init__(self, targetpatterns, prerequisites, doublecolon, loc):
        self.targetpatterns = targetpatterns
        self.prerequisites = prerequisites
        self.doublecolon = doublecolon
        self.loc = loc
        self.commands = []

    def addcommand(self, c):
        assert isinstance(c, (Expansion, StringExpansion))
        self.commands.append(c)

    def ismatchany(self):
        return any((t.ismatchany() for t in self.targetpatterns))

    def hasspecificmatch(self, file):
        for p in self.targetpatterns:
            if not p.ismatchany() and p.match(file) is not None:
                return True

        return False

    def matchesfor(self, dir_, file, skipsinglecolonmatchany):
        """
        Determine all the target patterns of this rule that might match target t.
        @yields a PatternRuleInstance for each.
        """

        for p in self.targetpatterns:
            matchany = p.ismatchany()
            if matchany:
                if skipsinglecolonmatchany and not self.doublecolon:
                    continue

                yield PatternRuleInstance(self, dir_, file, True)
            else:
                stem = p.match(dir_ + file)
                if stem is not None:
                    yield PatternRuleInstance(self, '', stem, False)
                else:
                    stem = p.match(file)
                    if stem is not None:
                        yield PatternRuleInstance(self, dir_, stem, False)

    def prerequisitesforstem(self, dir_, stem):
        return [p.resolve(dir_, stem) for p in self.prerequisites]


class _RemakeContext(object):
    def __init__(self, makefile, cb):
        self.makefile = makefile
        self.included = [(makefile.gettarget(f), required)
                         for f, required in makefile.included]
        self.toremake = list(self.included)
        self.cb = cb

        self.remakecb(error=False, didanything=False)

    def remakecb(self, error, didanything):
        assert error in (True, False)

        if error:
            if self.required:
                self.cb(remade=False, error=errors.MakeError(
                    'Error remaking required makefiles'))
                return
            else:
                print('Error remaking makefiles (ignored)')

        if len(self.toremake):
            target, self.required = self.toremake.pop(0)
            target.make(self.makefile, [], avoidremakeloop=True, cb=self.remakecb, printerror=False)
        else:
            for t, required in self.included:
                if t.wasremade:
                    _log.info("Included file %s was remade, restarting make", t.target)
                    self.cb(remade=True)
                    return
                elif required and t.mtime is None:
                    self.cb(remade=False,
                            error=errors.DataError("No rule to remake missing include file %s" % t.target))
                    return

            self.cb(remade=False)


class Makefile(object):
    """
    The top-level data structure for makefile execution. It holds Targets, implicit rules, and other
    state data.
    """

    def __init__(self, workdir=None, env=None, restarts=0, make=None,
                 makeflags='', makeoverrides='',
                 makelevel=0, context=None, targets=(), keepgoing=False,
                 silent=False, justprint=False):
        self.defaulttarget = None

        if env is None:
            env = os.environ
        self.env = env

        self.variables = Variables()
        self.variables.read_from_environment(env)

        self.context = context
        self.exportedvars = {}
        self._targets = {}
        self.keepgoing = keepgoing
        self.silent = silent
        self.justprint = justprint
        self._patternvariables = []  # of (pattern, variables)
        self.implicitrules = []
        self.parsingfinished = False

        self._patternvpaths = []  # of (pattern, [dir, ...])

        if workdir is None:
            workdir = os.getcwd()
        workdir = os.path.realpath(workdir)
        self.workdir = workdir
        self.variables.set('CURDIR', Variables.Flavor.SIMPLE, Variables.Source.AUTOMATIC, workdir.replace('\\', '/'))

        # the list of included makefiles, whether or not they existed
        self.included = []

        self.variables.set('MAKE_RESTARTS', Variables.Flavor.SIMPLE, Variables.Source.AUTOMATIC,
                           restarts > 0 and str(restarts) or '')

        self.variables.set('.PYMAKE', Variables.Flavor.SIMPLE, Variables.Source.MAKEFILE, "1")
        if make is not None:
            self.variables.set('MAKE', Variables.Flavor.SIMPLE, Variables.Source.MAKEFILE, make)

        if makeoverrides != '':
            self.variables.set('-*-command-variables-*-', Variables.Flavor.SIMPLE, Variables.Source.AUTOMATIC,
                               makeoverrides)
            makeflags += ' -- $(MAKEOVERRIDES)'

        self.variables.set('MAKEOVERRIDES', Variables.Flavor.RECURSIVE, Variables.Source.ENVIRONMENT,
                           '${-*-command-variables-*-}')

        self.variables.set('MAKEFLAGS', Variables.Flavor.RECURSIVE, Variables.Source.MAKEFILE, makeflags)
        self.exportedvars['MAKEFLAGS'] = True

        self.makelevel = makelevel
        self.variables.set('MAKELEVEL', Variables.Flavor.SIMPLE, Variables.Source.MAKEFILE, str(makelevel))

        self.variables.set('MAKECMDGOALS', Variables.Flavor.SIMPLE, Variables.Source.AUTOMATIC, ' '.join(targets))

        for vname, val in implicit.variables.items():
            self.variables.set(vname, Variables.Flavor.SIMPLE, Variables.Source.IMPLICIT, val)

    def foundtarget(self, t):
        """
        Inform the makefile of a target which is a candidate for being the default target,
        if there isn't already a default target.
        """
        flavor, source, value = self.variables.get('.DEFAULT_GOAL')
        if self.defaulttarget is None and t != '.PHONY' and value is None:
            self.defaulttarget = t
            self.variables.set('.DEFAULT_GOAL', Variables.Flavor.SIMPLE, Variables.Source.AUTOMATIC, t)

    def getpatternvariables(self, pattern):
        assert isinstance(pattern, Pattern)

        for p, v in self._patternvariables:
            if p == pattern:
                return v

        v = Variables()
        self._patternvariables.append((pattern, v))
        return v

    def getpatternvariablesfor(self, target):
        for p, v in self._patternvariables:
            if p.match(target):
                yield v

    def hastarget(self, target):
        return target in self._targets

    _globcheck = re.compile('[\[*?]')

    def gettarget(self, target):
        assert isinstance(target, str)

        target = target.rstrip('/')

        assert target != '', "empty target?"

        assert not self._globcheck.match(target)

        t = self._targets.get(target, None)
        if t is None:
            t = Target(target, self)
            self._targets[target] = t
        return t

    def appendimplicitrule(self, rule):
        assert isinstance(rule, PatternRule)
        self.implicitrules.append(rule)

    def finishparsing(self):
        """
        Various activities, such as "eval", are not allowed after parsing is
        finished. In addition, various warnings and errors can only be issued
        after the parsing data model is complete. All dependency resolution
        and rule execution requires that parsing be finished.
        """
        self.parsingfinished = True

        flavor, source, value = self.variables.get('GPATH')
        if value is not None and value.resolvestr(self, self.variables, ['GPATH']).strip() != '':
            raise errors.DataError('GPATH was set: pymake does not support GPATH semantics')

        flavor, source, value = self.variables.get('VPATH')
        if value is None:
            self._vpath = []
        else:
            self._vpath = [e for e in re.split('[%s\s]+' % os.pathsep,
                                               value.resolvestr(self, self.variables, ['VPATH'])) if e != '']

        # Must materialize target values because
        # gettarget() modifies self._targets.
        targets = list(self._targets.values())
        for t in targets:
            t.explicit = True
            for r in t.rules:
                for p in r.prerequisites:
                    self.gettarget(p).explicit = True

        np = self.gettarget('.NOTPARALLEL')
        if len(np.rules):
            self.context = process.getcontext(1)

        flavor, source, value = self.variables.get('.DEFAULT_GOAL')
        if value is not None:
            self.defaulttarget = value.resolvestr(self, self.variables, ['.DEFAULT_GOAL']).strip()

        self.error = False

    def include(self, path, required=True, weak=False, loc=None):
        """
        Include the makefile at `path`.
        """
        if self._globcheck.search(path):
            paths = globrelative.glob(self.workdir, path)
        else:
            paths = [path]
        for path in paths:
            self.included.append((path, required))
            fspath = util.normaljoin(self.workdir, path)
            if os.path.exists(fspath):
                if weak:
                    stmts = parser.parsedepfile(fspath)
                else:
                    stmts = parser.parsefile(fspath)
                self.variables.append('MAKEFILE_LIST', Variables.Source.AUTOMATIC, path, None, self)
                stmts.execute(self, weak=weak)
                self.gettarget(path).explicit = True

    def addvpath(self, pattern, dirs):
        """
        Add a directory to the vpath search for the given pattern.
        """
        self._patternvpaths.append((pattern, dirs))

    def clearvpath(self, pattern):
        """
        Clear vpaths for the given pattern.
        """
        self._patternvpaths = [(p, dirs)
                               for p, dirs in self._patternvpaths
                               if not p.match(pattern)]

    def clearallvpaths(self):
        self._patternvpaths = []

    def getvpath(self, target):
        vp = list(self._vpath)
        for p, dirs in self._patternvpaths:
            if p.match(target):
                vp.extend(dirs)

        return without_duplicates(vp)

    def remakemakefiles(self, cb):
        mlist = []
        for f, required in self.included:
            t = self.gettarget(f)
            t.explicit = True
            t.resolvevpath(self)
            oldmtime = t.mtime

            mlist.append((t, oldmtime))

        _RemakeContext(self, cb)

    def getsubenvironment(self, variables):
        env = dict(self.env)
        for vname, v in self.exportedvars.items():
            if v:
                flavor, source, val = variables.get(vname)
                if val is None:
                    strval = ''
                else:
                    strval = val.resolvestr(self, variables, [vname])
                env[vname] = strval
            else:
                env.pop(vname, None)

        makeflags = ''

        env['MAKELEVEL'] = str(self.makelevel + 1)
        return env
