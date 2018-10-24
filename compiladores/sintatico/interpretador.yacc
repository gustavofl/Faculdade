%{
#include <stdio.h>
#include <stdlib.h>
#include "tabela.h"

int yylex(void);
void yyerror(char *);

pilha_contexto *pilha;

%}

%union
{
	long dval;
	float fval;
}
%token PRINT
%token <dval> TYPE ID NUMBER INT FLOAT
%token <fval> NUMBER_FLOAT
%type <dval> expr_int
%type <fval> expr_float
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
	TYPE	ID ';'		{	printf("teste\n"); simbolo * s = criar_simbolo((char *) $2, $1); 
							inserir_simbolo(topo_pilha(pilha), s); }

	;

stmts: 
	stmts stmt
	| 	
	;

stmt:
	expr_int ';'		{	}
	| print
	| bloco
	| attr

	;

print:
	PRINT ID ';' {	 simbolo * s = localizar_simbolo(topo_pilha(pilha), (char *) $2);
								  if(s == NULL)
										printf("Identificador \"%s\" não declarado.\n", (char *) $2);
									else
										if(s->tipo == INT)
											printf("[%s = %d]\n", s->lexema, s->val.dval);
										else if(s->tipo == FLOAT)
											printf("[%s = %f]\n", s->lexema, s->val.fval);
								}
	;

attr: 
		/* localiza o símbolo e atualiza seu valor, utilizando o valor da expressão */
	ID '=' expr_int ';'		{ 	simbolo * s = localizar_simbolo(topo_pilha(pilha), (char *) $1);
							if(s == NULL)
								printf("Identificador \"%s\" nao definido.\n", (char *) $1);
							else{
								if(s->tipo == INT)
									s->val.dval = $3;
								else if(s->tipo == FLOAT){
									printf(">>> %li\n", $3);
									s->val.fval = $3;
								}
							}
						}

expr_int:

	 NUMBER			{ $$ = $1; }
	| ID			{ simbolo * s = localizar_simbolo(topo_pilha(pilha), (char *) $1);
						if(s == NULL)
							printf("Identificador \"%s\" não declarado.\n", (char *) $1);
						else  {
							$$ = s->val.dval;
						}
					}
	| expr_int '+' expr_int		{ $$ = $1 + $3; }
	| expr_int '-' expr_int		{ $$ = $1 - $3; }
	| '(' expr_int ')'		{ $$ = $2; }
	; 

expr_float:
	NUMBER_FLOAT	{ $$ = yylval.fval; }
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
