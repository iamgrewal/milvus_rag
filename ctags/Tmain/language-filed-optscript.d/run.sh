# Copyright: 2025 Masatake YAMATO
# License: GPL-2
CTAGS=$1

${CTAGS} --quiet --options=NONE --options=foo.ctags -o - input.foo
