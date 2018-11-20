all: compilador
compilador: y.tab.c lex.yy.c tabela.c tabela.h arvore.c arvore.h
		gcc -o compilador *.c
y.tab.c: compilador.yacc
		yacc -d compilador.yacc
lex.yy.c: compilador.lex
		lex compilador.lex
clean:
		rm y.tab.c y.tab.h lex.yy.c compilador
test: compilador source.ccc
		./compilador < source.ccc