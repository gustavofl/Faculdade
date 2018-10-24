all: interpretador
interpretador: y.tab.c lex.yy.c tabela.c tabela.h
		gcc -o interpretador *.c
y.tab.c: interpretador.yacc
		yacc -d interpretador.yacc
lex.yy.c: interpretador.lex
		lex interpretador.lex
clean:
		rm y.tab.c y.tab.h lex.yy.c interpretador
test: interpretador source.ccc
		./interpretador < source.ccc
bck: interpretador.yacc interpretador.lex tabela.c tabela.h source.ccc
	cp interpretador.lex interpretador.lex.bck
	cp interpretador.yacc interpretador.yacc.bck
	cp tabela.c tabela.c.bck
	cp tabela.h tabela.h.bck
	cp source.ccc source.ccc.bck