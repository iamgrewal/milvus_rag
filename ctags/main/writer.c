/*
*   Copyright (c) 2016, Red Hat, Inc.
*   Copyright (c) 2016, Masatake YAMATO
*
*   This source code is released for free distribution under the terms of the
*   GNU General Public License version 2 or (at your option) any later version.
*
*/

#include "general.h"

#include "colprint_p.h"
#include "debug.h"
#include "entry_p.h"
#include "options_p.h"
#include "writer_p.h"

#include <string.h>

extern tagWriter uCtagsWriter;
extern tagWriter eCtagsWriter;
extern tagWriter etagsWriter;
extern tagWriter xrefWriter;
extern tagWriter jsonWriter;

static tagWriter *writerTable [WRITER_COUNT] = {
	[WRITER_U_CTAGS] = &uCtagsWriter,
	[WRITER_E_CTAGS] = &eCtagsWriter,
	[WRITER_ETAGS] = &etagsWriter,
	[WRITER_XREF]  = &xrefWriter,
	[WRITER_JSON]  = &jsonWriter,
	[WRITER_CUSTOM] = NULL,
};

static tagWriter *writer;

extern void setTagWriter (writerType wtype, tagWriter *customWriter)
{
	if (wtype != WRITER_CUSTOM)
		writer = writerTable [wtype];
	else
		writer = customWriter;
	writer->type = wtype;
}

extern void writerSetup (MIO *mio, void *clientData)
{
	writer->clientData = clientData;

	if (writer->preWriteEntry)
		writer->private = writer->preWriteEntry (writer, mio,
												 writer->clientData);
	else
		writer->private = NULL;
}

extern bool writerTeardown (MIO *mio, const char *filename)
{
	if (writer->postWriteEntry)
	{
		bool r;
		r = writer->postWriteEntry (writer, mio, filename,
									writer->clientData);
		writer->private = NULL;
		return r;
	}
	return false;
}

extern int writerWriteTag (MIO * mio, const tagEntryInfo *const tag)
{
	return writer->writeEntry (writer, mio, tag,
							   writer->clientData);
}

extern int writerWritePtag (MIO * mio,
					 const ptagDesc *desc,
					 const char *const fileName,
					 const char *const pattern,
					 const char *const parserName)
{
	if (writer->writePtagEntry == NULL)
		return -1;

	return writer->writePtagEntry (writer, mio, desc, fileName,
								   pattern, parserName,
								   writer->clientData);

}

extern void writerRescanFailed (unsigned long validTagNum)
{
	if (writer->rescanFailedEntry)
		writer->rescanFailedEntry(writer, validTagNum, writer->clientData);
}

extern bool ptagMakeCtagsOutputMode (ptagDesc *desc, langType langType CTAGS_ATTR_UNUSED,
									 const void *data CTAGS_ATTR_UNUSED)
{
	const char *mode ="";

	if (&uCtagsWriter == writer)
		mode = "u-ctags";
	else if (&eCtagsWriter == writer)
		mode = "e-ctags";

	return writePseudoTag (desc,
						   mode,
						   "u-ctags or e-ctags",
						   NULL);
}

extern const char *outputDefaultFileName (void)
{
	return writer->defaultFileName;
}

extern bool writerCanPrintPtag (void)
{
	return (writer->writePtagEntry)? true: false;
}

extern bool writerCanPrintNullTag (void)
{
	return writer->canPrintNullTag;
}
extern bool writerDoesTreatFieldAsFixed (int fieldType)
{
	if (writer->treatFieldAsFixed)
		return writer->treatFieldAsFixed (fieldType);
	return false;
}

#ifdef _WIN32
extern enum filenameSepOp getFilenameSeparator (enum filenameSepOp currentSetting)
{
	if (writer->overrideFilenameSeparator)
		return writer->overrideFilenameSeparator (currentSetting);
	return currentSetting;
}
#endif

extern bool ptagMakeCtagsOutputFilesep (ptagDesc *desc,
										langType language CTAGS_ATTR_UNUSED,
										const void *data)
{
	const char *sep = "slash";
#ifdef _WIN32
	const optionValues *opt = data;
	if (getFilenameSeparator (opt->useSlashAsFilenameSeparator)
		!= FILENAME_SEP_USE_SLASH)
		sep = "backslash";
#endif
	return writePseudoTag (desc, sep, "slash or backslash", NULL);
}

extern bool ptagMakeCtagsOutputExcmd (ptagDesc *desc,
									  langType language CTAGS_ATTR_UNUSED,
									  const void *data)
{
	const char *excmd;
	const optionValues *opt = data;
	switch (opt->locate)
	{
	case EX_MIX:
		excmd = "mixed";
		break;
	case EX_LINENUM:
		excmd = "number";
		break;
	case EX_PATTERN:
		excmd = "pattern";
		break;
	case EX_COMBINE:
		excmd = "combineV2";
		break;
	default:
		AssertNotReached ();
		excmd = "bug!";
		break;
	}
	return writePseudoTag (desc, excmd,
						   "number, pattern, mixed, or combineV2",
						   NULL);
}

extern void writerCheckOptions (bool fieldsWereReset)
{
	if (writer->checkOptions)
		writer->checkOptions (writer, fieldsWereReset);
}

extern bool writerPrintPtagByDefault (void)
{
	return writer->printPtagByDefault;
}

extern writerType getWrierForOutputFormat (const char *oformat)
{
	for (int i = 0; i < WRITER_CUSTOM; i++)
	{
		if (writerTable[i]->oformat == NULL)
			continue;

		if (strcmp(writerTable[i]->oformat, oformat) == 0)
		{
			if (writerTable[i]->writeEntry == NULL)
				return WRITER_UNAVAILABLE;
			else
				return i;
		}
	}

	return WRITER_UNKNOWN;
}

#define WRITER_COL_OFORMAT 0
#define WRITER_COL_AVAILABLE 1
#define WRITER_COL_NULLTAG 2

static int writerColprintCompareLines (struct colprintLine *a , struct colprintLine *b)
{
	const char *a_oformat  = colprintLineGetColumn (a, WRITER_COL_OFORMAT);
	const char *b_oformat  = colprintLineGetColumn (b, WRITER_COL_OFORMAT);

	return strcmp(a_oformat, b_oformat);
}

extern struct colprintTable * writerColprintTableNew (void)
{
	return colprintTableNew ("L:OFORMAT", "R:DEFAULT", "R:AVAILABLE", "R:NULLTAG", NULL);
}

extern void printOutputFormats (bool withListHeader, bool machinable, FILE *fp)
{
	struct colprintTable * table = writerColprintTableNew ();

	for (int i = 0; i < WRITER_COUNT; i++)
	{
		if (!writerTable[i])
			continue;
		if (!writerTable[i]->oformat)
			continue;

		struct colprintLine * line = colprintTableGetNewLine (table);
		colprintLineAppendColumnCString (line, writerTable[i]->oformat);

		colprintLineAppendColumnBool (line, i == WRITER_DEFAULT);
		colprintLineAppendColumnBool (line, writerTable[i]->writeEntry? true: false);
		colprintLineAppendColumnBool (line, writerTable[i]->canPrintNullTag);
	}

	colprintTableSort (table, writerColprintCompareLines);
	colprintTablePrint (table, 0, withListHeader, machinable, fp);
	colprintTableDelete (table);
}
