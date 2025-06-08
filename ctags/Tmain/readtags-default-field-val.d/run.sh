#!/bin/sh

# Copyright: 2024 Masatake YAMATO
# License: GPL-2

READTAGS=$3

. ../utils.sh

#V="valgrind --leak-check=full -v"
V=

skip_if_no_readtags "$READTAGS"

# ?a
# 97
# (format "%c" 96)
# => `
"${READTAGS}" -t output.tags -S '(<> ($ "properties" "`") (& "properties" "`"))' -F '(list $name " " ($ "properties" "noprop") #t)' -l
