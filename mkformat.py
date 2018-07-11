#!/usr/bin/env python

import sys

from . import pymake

filename = sys.argv[1]

with open(filename, 'rU') as fh:
    source = fh.read()

statements = pymake.parser.parsestring(source, filename)
print(statements.to_source())
