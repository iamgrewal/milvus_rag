/*
*   Copyright (c) 2000-2001, Thaddeus Covert <sahuagin@mediaone.net>
*   Copyright (c) 2002 Matthias Veit <matthias_veit@yahoo.de>
*   Copyright (c) 2004 Elliott Hughes <enh@acm.org>
*
*   This source code is released for free distribution under the terms of the
*   GNU General Public License version 2 or (at your option) any later version.
*
*   This module contains functions for generating tags for Ruby language
*   files.
*/

#ifndef CTAGS_PARSER_RUBY_H
#define CTAGS_PARSER_RUBY_H

/*
*   INCLUDE FILES
*/
#include "general.h"  /* must always come first */

#include "subparser.h"

typedef struct sRubySubparser rubySubparser;

typedef enum {
	RUBY_CLASS_KIND,
	RUBY_METHOD_KIND,
	RUBY_MODULE_KIND,
	RUBY_SINGLETON_KIND,
	RUBY_CONST_KIND,
	RUBY_ACCESSOR_KIND,
	RUBY_ALIAS_KIND,
	RUBY_LIBRARY_KIND,
} rubyKind;

struct sRubySubparser {
	subparser subparser;
	/* Returning other than CORK_NIL means the string is consumed. */
	int (* lineNotify) (rubySubparser *s, const unsigned char **cp, int corkIndex);
	void (* enterBlockNotify) (rubySubparser *s, int corkIndex);
	void (* leaveBlockNotify) (rubySubparser *s, int corkIndex);
	/* Privately used in Ruby parser side. */
	int corkIndex;
};

/* Return true if it skips something. */
extern bool rubySkipWhitespace (const unsigned char **cp);

extern bool rubyCanMatchKeyword (const unsigned char** s, const char* literal);
extern bool rubyCanMatchKeywordWithAssign (const unsigned char** s, const char* literal);

/* rubyParseString() moves the *CP to the char just after the string literal started from
 * BOUNDARY (' or "). The string with no BOUNDARY is stored to VSTR.
 * You can use rubyParseString() just for skipping if you specify NULL as VSTR.

 * rubyParseString() returns false if **s is '\0' else true.
 * NOTE: even if the string is not terminated with BOUNDARY, the function moves *cp
 * and returns true. */
extern bool rubyParseString (const unsigned char** cp, unsigned char boundary, vString* vstr);
extern bool rubyParsePercentString (const unsigned char** cp, vString* vstr);

extern bool rubyParseMethodName (const unsigned char **cp, vString* vstr);
extern bool rubyParseModuleName (const unsigned char **cp, vString* vstr);

#endif
