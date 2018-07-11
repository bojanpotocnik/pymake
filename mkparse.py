#!/usr/bin/env python

import sys

from . import pymake

for f in sys.argv[1:]:
    print("Parsing %s" % f)
    fd = open(f, 'rU')
    s = fd.read()
    fd.close()
    statements = pymake.parser.parsestring(s, f)
    print(statements)
