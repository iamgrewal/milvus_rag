# -*- makefile -*-
.PHONY: check units fuzz noise tmain tinst tlib man-test clean-units clean-tlib clean-tmain clean-gcov clean-man-test run-gcov codecheck cppcheck dicts validate-input check-genfile tutil

EXTRA_DIST += misc/units misc/units.py misc/man-test.py
EXTRA_DIST += misc/tlib misc/mini-geany.expected
MAN_TEST_TMPDIR = ManTest

check: tmain units tlib man-test check-genfile tutil

# We may use CLEANFILES, DISTCLEANFILES, or etc.
clean-local: clean-units clean-tmain clean-man-test clean-tlib clean-gcov

CTAGS_TEST = ./ctags$(EXEEXT)
READTAGS_TEST = ./readtags$(EXEEXT)
MINI_GEANY_TEST = ./mini-geany$(EXEEXT)
OPTSCRIPT_TEST = ./optscript$(EXEEXT)
UTILTEST_TEST = ./utiltest$(EXEEXT)

# Make these macros empty from make's command line
# if you don't want to (re)build these executables
# before testing.
# e.g.
#
#    $ make units CTAGS_DEP=
#
CTAGS_DEP = $(CTAGS_TEST)
READTAGS_DEP = $(READTAGS_TEST)
MINI_GEANY_DEP = $(MINI_GEANY_TEST)
OPTSCRIPT_DEP = $(OPTSCRIPT_TEST)
UTILTEST_DEP = $(UTILTEST_TEST)

if HAVE_TIMEOUT
TIMEOUT = 1
else
TIMEOUT = 0
endif

LANGUAGES=
CATEGORIES=
UNITS=
PMAP=

SILENT = $(SILENT_@AM_V@)
SILENT_ = $(SILENT_@AM_DEFAULT_V@)
SILENT_0 = @

V_RUN = $(V_RUN_@AM_V@)
V_RUN_ = $(V_RUN_@AM_DEFAULT_V@)
V_RUN_0 = @echo "  RUN      $@";

#
# FUZZ Target
#
# SHELL must be dash or bash.
#
fuzz: $(CTAGS_DEP)
	$(V_RUN) \
	if test -n "$${ZSH_VERSION+set}"; then set -o SH_WORD_SPLIT; fi; \
	if test x$(VG) = x1; then		\
		VALGRIND=--with-valgrind;	\
	fi;					\
	c="$(srcdir)/misc/units fuzz \
		--ctags=$(CTAGS_TEST) \
		--languages=$(LANGUAGES) \
		$${VALGRIND} --run-shrink \
		--with-timeout=`expr $(TIMEOUT) '*' 10`"; \
	$(SHELL) $${c} $(srcdir)/Units

#
# NOISE Target
#
noise: $(CTAGS_DEP)
	$(V_RUN) \
	if test -n "$${ZSH_VERSION+set}"; then set -o SH_WORD_SPLIT; fi; \
	if test x$(VG) = x1; then		\
		VALGRIND=--with-valgrind;	\
	fi;					\
	c="$(srcdir)/misc/units noise \
		--ctags=$(CTAGS_TEST) \
		--languages=$(LANGUAGES) \
		$${VALGRIND} --run-shrink \
		--with-timeout=$(TIMEOUT)"; \
	$(SHELL) $${c} $(srcdir)/Units

#
# CHOP Target
#
chop: $(CTAGS_DEP)
	$(V_RUN) \
	if test -n "$${ZSH_VERSION+set}"; then set -o SH_WORD_SPLIT; fi; \
	if test x$(VG) = x1; then		\
		VALGRIND=--with-valgrind;	\
	fi;					\
	c="$(srcdir)/misc/units chop \
		--ctags=$(CTAGS_TEST) \
		--languages=$(LANGUAGES) \
		$${VALGRIND} --run-shrink \
		--with-timeout=$(TIMEOUT)"; \
	$(SHELL) $${c} $(srcdir)/Units
slap: $(CTAGS_DEP)
	$(V_RUN) \
	if test -n "$${ZSH_VERSION+set}"; then set -o SH_WORD_SPLIT; fi; \
	if test x$(VG) = x1; then		\
		VALGRIND=--with-valgrind;	\
	fi;					\
	c="$(srcdir)/misc/units slap \
		--ctags=$(CTAGS_TEST) \
		--languages=$(LANGUAGES) \
		$${VALGRIND} --run-shrink \
		--with-timeout=$(TIMEOUT)"; \
	$(SHELL) $${c} $(srcdir)/Units

#
# UNITS Target
#
units: $(CTAGS_DEP)
	$(V_RUN) \
	if test -n "$${ZSH_VERSION+set}"; then set -o SH_WORD_SPLIT; fi; \
	if test x$(VG) = x1; then		\
		VALGRIND=--with-valgrind;	\
	fi;					\
	if ! test x$(CI) = x; then	\
		SHOW_DIFF_OUTPUT=--show-diff-output;		\
	fi;							\
	builddir=$$(pwd); \
	if ! test x$(PYTHON) = x; then	\
		PROG=$(PYTHON);		\
		SCRIPT=$(srcdir)/misc/units.py;	\
		if type cygpath > /dev/null 2>&1; then	\
			builddir=$$(cygpath -m "$$(pwd)");	\
			if ! test x$(SHELL) = x; then	\
				SHELL_OPT=--shell=$$(cygpath -m $(SHELL));	\
			fi;	\
		else	\
			if ! test x$(SHELL) = x; then	\
				SHELL_OPT=--shell=$(SHELL);	\
			fi;	\
		fi;	\
	else	\
		PROG=$(SHELL);		\
		SCRIPT=$(srcdir)/misc/units;	\
	fi;	\
	if ! test x$(THREADS) = x; then \
		THREADS_OPT=--threads=$(THREADS); \
	fi; \
	mkdir -p $${builddir}/Units && \
	\
	c="$${SCRIPT} run \
		--ctags=$(CTAGS_TEST) \
		--languages=$(LANGUAGES) \
		--categories=$(CATEGORIES) \
		--units=$(UNITS) \
		--with-pretense-map=$(PMAP) \
		$${VALGRIND} --run-shrink \
		--with-timeout=`expr $(TIMEOUT) '*' 10`\
		$${SHELL_OPT} \
		$${THREADS_OPT} \
		$${SHOW_DIFF_OUTPUT}"; \
		 $${PROG} $${c} $(srcdir)/Units $${builddir}/Units

clean-units:
	$(SILENT) echo Cleaning test units
	$(SILENT) if test -d $$(pwd)/Units; then \
		$(SHELL) $(srcdir)/misc/units clean $$(pwd)/Units; \
	fi

#
# VALIDATE-INPUT Target
#
validate-input:
	$(V_RUN) \
	if test -n "$${ZSH_VERSION+set}"; then set -o SH_WORD_SPLIT; fi; \
	if test -n "$(VALIDATORS)"; then	\
		VALIDATORS="--validators=$(VALIDATORS)"; \
	fi; \
	c="$(srcdir)/misc/units validate-input \
		--categories=$(CATEGORIES) \
		$${VALIDATORS}"; \
	$(SHELL) $${c} $(srcdir)/Units $(srcdir)/misc/validators

#
# Test main part, not parsers
#
tmain: $(CTAGS_DEP) $(READTAGS_DEP) $(OPTSCRIPT_DEP)
	$(V_RUN) \
	if test -n "$${ZSH_VERSION+set}"; then set -o SH_WORD_SPLIT; fi; \
	if test x$(VG) = x1; then		\
		VALGRIND=--with-valgrind;	\
	fi;					\
	if ! test x$(CI) = x; then	\
		SHOW_DIFF_OUTPUT=--show-diff-output;		\
	fi;							\
	builddir=$$(pwd); \
	if ! test x$(PYTHON) = x; then	\
		PROG=$(PYTHON);		\
		SCRIPT=$(srcdir)/misc/units.py;	\
		if type cygpath > /dev/null 2>&1; then	\
			builddir=$$(cygpath -m "$$(pwd)");	\
			if ! test x$(SHELL) = x; then	\
				SHELL_OPT=--shell=$$(cygpath -m $(SHELL));	\
			fi;	\
		else	\
			if ! test x$(SHELL) = x; then	\
				SHELL_OPT=--shell=$(SHELL);	\
			fi;	\
		fi;	\
	else	\
		PROG=$(SHELL);		\
		SCRIPT=$(srcdir)/misc/units;	\
	fi;	\
	if ! test x$(THREADS) = x; then \
		THREADS_OPT=--threads=$(THREADS); \
	fi; \
	mkdir -p $${builddir}/Tmain && \
	\
	c="$${SCRIPT} tmain \
		--ctags=$(CTAGS_TEST) \
		--units=$(UNITS) \
		$${VALGRIND} \
		$${SHELL_OPT} \
		$${THREADS_OPT} \
		$${SHOW_DIFF_OUTPUT}"; \
		$${PROG} $${c} $(srcdir)/Tmain $${builddir}/Tmain

clean-tmain:
	$(SILENT) echo Cleaning main part tests
	$(SILENT) if test -d $$(pwd)/Tmain; then \
		$(SHELL) $(srcdir)/misc/units clean-tmain $$(pwd)/Tmain; \
	fi

tlib: $(MINI_GEANY_DEP)
	$(V_RUN) \
	builddir=$$(pwd); \
	mkdir -p $${builddir}/misc; \
	if test -s '$(MINI_GEANY_TEST)'; then \
		if $(SHELL) $(srcdir)/misc/tlib $(MINI_GEANY_TEST) \
			$(srcdir)/misc/mini-geany.expected \
			$${builddir}/misc/mini-geany.actual \
			$(VG); then \
			echo 'mini-geany: OK'; true; \
		else \
			echo 'mini-geany: FAILED'; false; \
		fi; \
	else \
		echo 'mini-geany: SKIP'; true; \
	fi
clean-tlib:
	$(SILENT) echo Cleaning libctags part tests
	$(SILENT) builddir=$$(pwd); \
		rm -f $${builddir}/misc/mini-geany.actual

#
# Test installation
#
tinst:
	$(V_RUN) \
	builddir=$$(pwd); \
	rm -rf $$builddir/$(TINST_ROOT); \
	$(SHELL) $(srcdir)/misc/tinst $(srcdir) $$builddir/$(TINST_ROOT)

#
# Test readtags
#
if USE_READCMD
roundtrip: $(READTAGS_DEP)
	$(V_RUN) \
	if ! test x$(CI) = x; then	\
		ROUNDTRIP_FLAGS=--minitrip;			\
	fi;							\
	builddir=$$(pwd); \
	$(SHELL) $(srcdir)/misc/roundtrip $(READTAGS_TEST) $${builddir}/Units $${ROUNDTRIP_FLAGS}
else
roundtrip:
endif

#
# Checking code in ctags own rules
#
codecheck: $(CTAGS_DEP)
	$(V_RUN) $(SHELL) misc/src-check

#
# Report coverage (usable only if ctags is built with "configure --enable-coverage-gcov".)
#
run-gcov:
	$(CTAGS_TEST) -o - $$(find ./Units -name 'input.*'| grep -v '.*b/.*') > /dev/null
	gcov $$(find -name '*.gcda')

clean-gcov:
	$(SILENT) echo Cleaning coverage reports
	$(SILENT) rm -f $(ALL_SRCS:.c=.gcda)
	$(SILENT) rm -f $(srcdir)/*.gcov

#
# Cppcheck
#
CPPCHECK_DEFS   = -DHAVE_LIBYAML -DHAVE_LIBXML -DHAVE_COPROC -DHAVE_DECL___ENVIRON
CPPCHECK_UNDEFS = -UDEBUG -UMIO_DEBUG -UCXX_DEBUGGING_ENABLED
CPPCHECK_FLAGS  = --enable=all

cppcheck:
	cppcheck $(CPPCHECK_DEFS) $(CPPCHECK_UNDEFS) $(CPPCHECK_FLAGS) \
		 $$(git  ls-files | grep '^\(parsers\|main\)/.*\.[ch]' )

#
# Testing examples in per-language man pages
#
man-test: $(CTAGS_DEP)
	$(V_RUN) \
	$(PYTHON) $(srcdir)/misc/man-test.py $(MAN_TEST_TMPDIR) $(CTAGS_TEST) $(srcdir)/man/ctags-lang-*.7.rst.in

clean-man-test:
	rm -rf $(MAN_TEST_TMPDIR)

# check if generated files are committed.
#   Note: "make -B" cannot be used here, since it reruns automake
chkgen_verbose = $(chkgen_verbose_@AM_V@)
chkgen_verbose_ = $(chkgen_verbose_@AM_DEFAULT_V@)
chkgen_verbose_0 = @echo CHKGEN "    $@";

cgok             =  echo "<ok>       $@:"
cgerr            =  echo "<ERROR>    $@:"
cgskip           =  echo "<skip>     $@:"

recover_side_effects = cg-force-optlib2c-srcs cg-force-txt2cstr-srcs cg-force-man-docs

# OPTLIB2C_SRCS : committed for win32 build
.PHONY: cg-clean-optlib2c-srcs cg-force-optlib2c-srcs check-genfile-optlib2c-srcs
cg-clean-optlib2c-srcs:
if BUILD_IN_GIT_REPO
	$(chkgen_verbose)rm -f $(OPTLIB2C_SRCS)
endif
cg-force-optlib2c-srcs: cg-clean-optlib2c-srcs
if BUILD_IN_GIT_REPO
	$(chkgen_verbose)$(MAKE) $(OPTLIB2C_SRCS)
endif
check-genfile-optlib2c-srcs: $(recover_side_effects) cg-force-optlib2c-srcs
if BUILD_IN_GIT_REPO
	$(chkgen_verbose)if ! git diff --exit-code $(OPTLIB2C_DIR); then \
		$(cgerr) "Files under $(OPTLIB2C_DIR) are not up to date." ; \
		$(cgerr) "If you change $(OPTLIB2C_DIR)/foo.ctags, don't forget to add $(OPTLIB2C_DIR)/foo.c to your commit." ; \
		exit 1 ; \
	else \
		$(cgok) "Files under $(OPTLIB2C_DIR) are up to date." ; \
	fi
endif

# TXT2CSTR_SRCS : committed for win32 build
.PHONY: cg-clean-txt2cstr-srcs cg-force-txt2cstr-srcs check-genfile-txt2cstr-srcs
cg-clean-txt2cstr-srcs:
if BUILD_IN_GIT_REPO
	$(chkgen_verbose)rm -f $(TXT2CSTR_SRCS)
endif
cg-force-txt2cstr-srcs: cg-clean-txt2cstr-srcs
if BUILD_IN_GIT_REPO
	$(chkgen_verbose)$(MAKE) $(TXT2CSTR_SRCS)
endif
check-genfile-txt2cstr-srcs: $(recover_side_effects) cg-force-txt2cstr-srcs
if BUILD_IN_GIT_REPO
	$(chkgen_verbose)if ! git diff --exit-code $(TXT2CSTR_DIR); then \
		$(cgerr) "Files under $(TXT2CSTR_DIR) are not up to date." ; \
		$(cgerr) "If you change $(TXT2CSTR_DIR)/foo.ps, don't forget to add $(TXT2CSTR_DIR)/foo.c to your commit." ; \
		exit 1 ; \
	else \
		$(cgok) "Files under $(TXT2CSTR_DIR) are up to date." ; \
	fi
endif

# man/*.in : committed for man pages to be genrated without rst2man
#   make clean-docs remove both man/*.in and docs/man/*.rst
.PHONY: cg-clean-man-docs cg-force-man-docs check-genfile-man-docs
cg-clean-man-docs:
if BUILD_IN_GIT_REPO
if HAVE_RST2MAN
	$(chkgen_verbose)$(MAKE) -C man clean-docs
endif
endif
cg-force-man-docs: cg-clean-man-docs
if BUILD_IN_GIT_REPO
if HAVE_RST2MAN
	$(chkgen_verbose)$(MAKE) -C man man-in
endif
endif
check-genfile-man-docs:  $(recover_side_effects) cg-force-man-docs
if BUILD_IN_GIT_REPO
if HAVE_RST2MAN
	$(chkgen_verbose)if ! git diff --exit-code -- man; then \
		$(cgerr) "Files under man/ are not up to date." ; \
		$(cgerr) "Please execute 'make -C man man-in' and commit them." ; \
		exit 1 ; \
	else \
		$(cgok) "Files under man are up to date." ; \
	fi
endif
endif

# docs/man/*.rst : committed for Read the Docs
.PHONY: cg-force-update-docs check-genfile-update-docs
cg-force-update-docs: check-genfile-man-docs
if BUILD_IN_GIT_REPO
if HAVE_RST2MAN
	$(chkgen_verbose)$(MAKE) -C man update-docs
endif
endif

check-genfile-update-docs: cg-force-update-docs $(recover_side_effects)
if BUILD_IN_GIT_REPO
if HAVE_RST2MAN
	$(chkgen_verbose)if ! git diff --exit-code -- docs/man; then \
		$(cgerr) "Files under docs/man/ are not up to date." ; \
		$(cgerr) "Please execute 'make -C man update-docs' and commit them." ; \
		exit 1 ; \
	else \
		$(cgok) "Files under docs/man are up to date." ; \
	fi
endif
endif

# win32/{ctags_vs2013.vcxproj*,peg_rule.mak} : committed for win32 build without POSIX tools
#   regenerate files w/o out-of-source build and w/ GNU make
.PHONY: cg-force-win32 check-genfile-win32
cg-force-win32:
if BUILD_IN_GIT_REPO
	$(chkgen_verbose)if test "$(top_srcdir)" = "$(top_builddir)" \
		&& ($(MAKE) --version) 2>/dev/null | grep -q GNU ; then \
		$(MAKE) -BC win32 ; \
	fi
endif
check-genfile-win32: cg-force-win32 $(recover_side_effects)
if BUILD_IN_GIT_REPO
	$(chkgen_verbose)if ! git diff --exit-code -- win32; then \
		if test "$(SKIP_CHECKGEN_WIN32)" = "yes"; then \
			$(cgskip) "Skip checking the files under win32." ; \
			exit 0 ; \
		else \
			$(cgerr) "Files under win32/ are not up to date." ; \
			$(cgerr) "Please execute 'make -BC win32' and commit them." ; \
			exit 1 ; \
		fi \
	else \
		$(cgok) "Files under win32 are up to date." ; \
	fi
endif

.PHONY: check-genfile-add-docs-man
check-genfile-add-docs-man: $(recover_side_effects)
	$(chkgen_verbose) {\
		(cd man; git ls-files .) | grep ctags-lang- | sed -e 's/\.in$$//' > TEMP-MAN-LS; \
		(cd docs/man; git ls-files .) | grep ctags-lang-  > TEMP-DOCS-MAN-LS; \
		if ! diff TEMP-MAN-LS TEMP-DOCS-MAN-LS; then \
			$(cgerr) 'See "<" lines above.'; \
			$(cgerr) 'docs/man/*rst genereated from man/*rst.in are not in the git repo'; \
			$(cgerr) 'Please add the genereated file to the git repo'; \
			rm TEMP-MAN-LS TEMP-DOCS-MAN-LS; \
			exit 1 ; \
		else \
			rm TEMP-MAN-LS TEMP-DOCS-MAN-LS; \
			$(cgok) 'All rst files under docs/man are in our git repo'; \
		fi; \
	}

.PHONY: check-genfile-docs-man-pages-rst
check-genfile-docs-man-pages-rst: $(recover_side_effects)
	$(chkgen_verbose) for f in $$( (cd docs/man; git ls-files .) | grep ctags-lang- ); do \
		if ! grep -q $$f docs/man-pages.rst; then \
			$(cgerr) "$$f is not found in docs/man-pages.rst"; \
			$(cgerr) "Please add $$f to docs/man-pages.rst"; \
			exit 1; \
		fi; \
	done; \
	$(cgok) "docs/man-pages.rst includes all ctags-lang-*.rst"

check-genfile: \
	check-genfile-optlib2c-srcs \
	check-genfile-txt2cstr-srcs \
	check-genfile-update-docs \
	check-genfile-add-docs-man \
	check-genfile-docs-man-pages-rst \
	check-genfile-win32

#
# Test installation
#
tutil: $(UTILTEST_DEP)
# See _VALGRIND_EXIT in misc/uints.py about 56.
	$(V_RUN) vg=; \
	if test x$(VG) = x1; then \
		vg="valgrind "; \
		vg="$$vg --leak-check=full"; \
		vg="$$vg --track-origins=yes"; \
		vg="$$vg --error-exitcode=56"; \
	fi; \
	\
	builddir=$$(pwd); \
	$$vg $$builddir/$(UTILTEST_TEST) -v
