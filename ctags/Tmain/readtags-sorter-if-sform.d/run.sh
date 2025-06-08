#!/bin/sh

# Copyright: 2020 Masatake YAMATO
# License: GPL-2

READTAGS=$3

. ../utils.sh

#V="valgrind --leak-check=full -v"
V=

skip_if_no_readtags "$READTAGS"

${READTAGS} -t output.tags -ne -S '(if (eq? $kind &kind)
    0
  (+
   (if (or (eq? $kind "t"))
       -4
     (if (eq? $kind "v")
	 -3
       (if (eq? $kind "d")
	   -2
	 (if (eq? $kind "e")
	     -1
	   0))))
   (if (or (eq? &kind "t"))
       4
     (if (eq? &kind "v")
	 3
       (if (eq? &kind "d")
	   2
	 (if (eq? &kind "e")
	     1
	   0))))))' mutex
