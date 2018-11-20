%{
#include <stdio.h>
#include "tabela.h"
#include "arvore.h"

int yylex(void);
void yyerror(char *);

pilha_contexto *pilha;

%}

%token PROGRAMA TIPO INT REAL NUM_INT NUM_REAL ID EXPR ATTR OU E NAO SE SENAO ENQUANTO FUNCAO ESCREVA LEIA CADEIA PULALINHA
%left '+' '-'
%left '*' '/' '%'
%%

program:
	program PROGRAMA 			{ printf("PROGRAMA\n"); }
	| program NUM_REAL			{ printf("numero real %.2f\n", $2); }
	| program NUM_INT				{ printf("numero inteiro %d\n", $2); }
	|
	;

%%

void yyerror(char *s) {
	fprintf(stderr, "%s\n", s);
}

int main(void) {
	pilha = NULL;
	yyparse();
	return 0;
}
