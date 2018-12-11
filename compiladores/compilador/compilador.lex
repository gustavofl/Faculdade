%{
#include <stdlib.h>
void yyerror(char *);
#include "y.tab.h"
%}

letra	[a-z|A-Z|_] 
digito	[0-9]
identificador	{letra}({letra}|{digito})*

%%

programa	{ return PROGRAMA; }

{digito}+	{ 
				yylval = atoi(yytext);
				return NUM_INT;
			}

{digito}+\.{digito}+ {
				yylval = (int) strdup(yytext);
				return NUM_REAL;
			}

inteiro		{ return INT; }

real		{ return FLOAT; }

{identificador}	{}

ou			{}

e			{}

nao			{}

se			{}

senao		{}

enquanto	{}

funcao		{}

escreva		{}

leia		{}

cadeia		{}

\n 			{}


[-+*/%<>!=(){}]	{ return *yytext; }




[ \t] 	; /* skip whitespace */
. 	yyerror("invalid character");
%%
int yywrap(void) {
return 1;
}
