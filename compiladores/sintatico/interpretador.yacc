%{
#ifndef YYSTYPE
#define YYSTYPE long int
#endif
#include <stdio.h>
#include <stdlib.h>
#include "tabela.h"

int yylex(void);
void yyerror(char *);

pilha_contexto *pilha;

%}

%union {
	int dval;
	float fval;
}
%token TYPE INT FLOAT PRINT ID NUMBER
%type<dval> TYPE ID NUMBER expr
%left '+' '-'
%%


program:
			
	program bloco		{ }
	|
	; 	

bloco: 
	'{' 			{ tabela *contexto = criar_contexto(topo_pilha(pilha));
				  pilha = empilhar_contexto(pilha, contexto);

				 }
	decls stmts '}'		{ imprimir_contexto(topo_pilha(pilha));
				  desempilhar_contexto(&pilha); }
	;

decls: 
	decls decl		{ }
	|
	;
	
decl:
	TYPE	ID ';'		{	simbolo * s = criar_simbolo((char *) $2, $1); 
										inserir_simbolo(topo_pilha(pilha), s); }

	;

stmts: 
	stmts stmt
	| 	
	;

stmt:
	expr ';'		{	}
	| print
	| bloco
	| attr

	;

print:
	PRINT ID ';' {	 simbolo * s = localizar_simbolo(topo_pilha(pilha), (char *) $2);
								  if(s == NULL)
											yyerror("Identificador não declarado");
									else
											printf("[%s = %d]\n", s->lexema, s->val.dval);
								}
	;

attr: 
									/* localiza o símbolo e atualiza seu valor, utilizando o valor da expressão */
	ID '=' expr ';'		{ 	simbolo * s = localizar_simbolo(topo_pilha(pilha), (char *) $1);
							if(s == NULL)
								printf("Identificador %s nao definido.\n", (char *) $1);
							else{
								if(s->tipo == INT)
									s->val.dval = $3;
								else if(s->tipo == FLOAT)
									s->val.fval = $3;
							}
						}

expr:

	 NUMBER			{ $$ = yylval.dval; }
	| ID			{ simbolo * s = localizar_simbolo(topo_pilha(pilha), (char *) $1);
				  if(s == NULL)
						yyerror("Identificador não declarado");
				  else  {
							$$ = s->val.dval;
				  }
				}
	| expr '+' expr		{ $$ = $1 + $3; }
	| expr '-' expr		{ $$ = $1 - $3; }
	| '(' expr ')'		{ $$ = $2; }
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
