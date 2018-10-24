%{
#include <stdlib.h>
void yyerror(char *);
#include "y.tab.h"
%}

letra	[a-z|A-Z|_] 
numero	[0-9]
identificador	{letra}({letra}|{numero})*

%%


{numero}+	{ yylval.dval = atoi(yytext);
		  return NUMBER;
		}

int		{	yylval.dval = INT;
			return TYPE;
		}
float		{
			yylval.dval = FLOAT;
			return TYPE;
		}

PRINT		{	return PRINT; 
		}

{identificador}	{
			yylval.dval = (long) strdup(yytext);
			return ID;
		}	

[-+=(){};]	{	return *yytext; }




[ \t\n] 	; /* skip whitespace */
. 	yyerror("invalid character");
%%
int yywrap(void) {
return 1;
}
