import math, time
import random
from random import randint

def distancia_euclidiana(p, q):
	distancia = 0
	p_carac = p[0]
	q_carac = q[0]
	
	for i in range(len(p_carac)):
		distancia += (p_carac[i] - q_carac[i])**2
	distancia = math.sqrt(distancia)

	return distancia

def distancia_hamming(p, q):
	distancia = 0
	p_carac = p[0]
	q_carac = q[0]

	for i in range(len(p_carac)):
		if(p_carac[i] != q_carac[i]):
			distancia += 1

	return distancia

def media_lista(lista):
	tam = len(lista)
	soma = 0
	for i in lista: soma += i

	return float(soma)/float(tam)

def desvio_padrao(lista):
	tam = len(lista)
	media = media_lista(lista)
	soma = 0
	for i in lista: soma += (i - media)**2

	desvio = math.sqrt(float(soma)/float(tam))

	return desvio

def ordenar_lista_maior(lista):
	if(len(lista) >= 2):
		for i in range(len(lista)-1,0,-1):
			if(lista[i][-1] > lista[i-1][-1]):
				lista[i],lista[i-1] = lista[i-1],lista[i]

	return lista

def moda_lista(lista, com_peso):
	categorias = {}

	for i in lista:
		pessoa = i[1]
		if(not pessoa in categorias):
			categorias[pessoa] = 0
		if(com_peso == False):
			categorias[pessoa] += 1
		else:
			if(i[-1] == 0): i[-1] = 0.000001

			valor = 1.0/float(i[-1])
			categorias[pessoa] += valor

	maior=0
	moda=""
	for i in categorias:
		if(categorias[i] > maior):
			moda = i
			maior = categorias[i]

	return moda

def separar_fold_cross_validation(dados, r):
	particoes = []
	tam = len(dados)

	tam_particao = int(tam/r)
	for i in range(r-1):
		particoes.append(dados[i*tam_particao:(i+1)*tam_particao])
	particoes.append(dados[(r-1)*tam_particao:])

	return particoes

def print_lista(lista):
	print('[')
	for i in lista:
		print('\t'+str(i))
	print(']')

def classificador_knn(treino, teste, k, funcao_distancia=distancia_euclidiana, com_peso=False):
	treino_original = treino
	qnt_total = 0
	qnt_sucesso = 0

	for i,p in enumerate(teste):
		
		lista_proximos = []
		treino = treino_original

		for j,q in enumerate(treino):
			distancia = funcao_distancia(p,q)

			if(len(lista_proximos) < k):
				# q.append(distancia)
				q[-1] = distancia
				lista_proximos.append(q)
			elif(distancia < lista_proximos[0][-1]):
				lista_proximos.pop()
				# q.append(distancia)
				q[-1] = distancia
				lista_proximos.append(q)
				ordenar_lista_maior(lista_proximos)

		resultado = lista_proximos[0][1]
		resultado = moda_lista(lista_proximos, com_peso)
		qnt_total += 1
		if(resultado == p[1]):
			qnt_sucesso += 1

	return float(qnt_sucesso)/float(qnt_total)

def classificador_naive_bayes(treino, teste):
	arvore_prob_caracs,prob_classes = calcular_probabilidades(treino)
	qnt_sucesso = 0
	qnt_total = 0

	for p in teste:
		maior_classe = get_maior_classe(prob_classes)
		maior_prob = 0

		for classe in prob_classes:

			probabilidade = 1
			probabilidade *= prob_classes[classe]
			for i,carac in enumerate(p[0]):
				try:
					probabilidade *= arvore_prob_caracs[classe][i][carac]
				except:
					probabilidade *= 0

			if(probabilidade > maior_prob):
				maior_prob = probabilidade
				maior_classe = classe

		if(maior_classe == p[1]):
			qnt_sucesso += 1
		qnt_total += 1

	return qnt_sucesso/float(qnt_total)

# preparacao para naive bayes
def calcular_probabilidades(lista):
	# quantidade de cada classe
	qnt_classes = {}
	for dado in lista:
		if(not dado[1] in qnt_classes.keys()):
			qnt_classes[dado[1]] = 0
		qnt_classes[dado[1]] += 1

	# quanticade de cada valor de cada caracteristica para cada classe
	arvore_probabilidades = {}
	for dado in lista:
		classe = dado[1]
		if(not classe in arvore_probabilidades.keys()):
			arvore_probabilidades[classe] = {}

		for i,carac in enumerate(dado[0]):
			if(not i in arvore_probabilidades[classe].keys()):
				arvore_probabilidades[classe][i] = {}
			if(not carac in arvore_probabilidades[classe][i].keys()):
				arvore_probabilidades[classe][i][carac] = 0

			arvore_probabilidades[classe][i][carac] += 1

	# calculo das probabilidades posterioris das caracteristicas
	for classe in arvore_probabilidades:
		for indice in arvore_probabilidades[classe]:
			for carac in arvore_probabilidades[classe][indice]:
				probabilidade = arvore_probabilidades[classe][indice][carac]/float(qnt_classes[classe])
				arvore_probabilidades[classe][indice][carac] = probabilidade

	# calculo das prioris das classes
	for classe in qnt_classes:
		qnt_classes[classe] /= float(len(lista))

	return arvore_probabilidades,qnt_classes

def get_maior_classe(dic):
	maior_classe = list(dic)[0]

	for classe in dic:
		if(dic[classe] > dic[maior_classe]):
			maior_classe = classe

	return maior_classe

def ler_arquivo(arquivo):
	arq = open(arquivo, "r")

	linhas = arq.readlines()

	caracteristicas = []

	caracs_irrelevante = [',','.','!','?','/','(',')','[',']','{','}',';',':','_','-','>','<']

	for i in range(1,len(linhas)):
		linha = linhas[i]

		linha = linha.replace("\n","")
		linha,rotulo = linha.split('\t')

		linha = linha.lower()

		for c in caracs_irrelevante:
			linha = linha.replace(c,' ')

		dados = linha.split()

		caracteristicas.append([dados,rotulo])

	return caracteristicas

def construir_dicionario(dados):
	dicionario = []

	for instance in dados:
		for carac in instance[0]:
			if(not carac in dicionario):
				dicionario.append(carac)

	return dicionario

def extracao_caracs(dados):
	dicionario = construir_dicionario(dados)

	dados_carac_bin = []

	for d in dados:
		caracs = []
		for carac in dicionario:
			if(carac in d[0]):
				caracs.append(1)
			else:
				caracs.append(0)
		dados_carac_bin.append([caracs,d[1]])

	return dados_carac_bin

def main():
	dados_amazon = ler_arquivo("sentiment_labelled_sentences/amazon_cells_labelled.txt")
	dados_imdb = ler_arquivo("sentiment_labelled_sentences/imdb_labelled.txt")
	dados_yelp = ler_arquivo("sentiment_labelled_sentences/yelp_labelled.txt")

	bases = [extracao_caracs(dados_amazon), extracao_caracs(dados_imdb), extracao_caracs(dados_yelp)]

	# add atributo para distancia
	for base in bases:
		for i in range(len(base)):
			base[i] = [base[i][0],base[i][1],0]

	k=1
	r=10
	repeticao=1

	for ind_base,base in enumerate(("AMAZON", "IMDB", "YELP")):
	#if(0):
		print("\n\n>>>> Base de dados "+base)

		dados = bases[ind_base]

		print("\n\nNaive Bayes")
		lista_acuracia_naive_bayes = []
		for i in range(repeticao):
			random.shuffle(dados)
			particoes = separar_fold_cross_validation(dados, r)

			for iteracao_fold_cross in range(r):
				treino = []
				teste = []

				for j in range(r):
					if(j == iteracao_fold_cross):
						teste = particoes[j]
					else:
						treino.extend(particoes[j])

				resultado_classificador = classificador_naive_bayes(treino, teste)
				lista_acuracia_naive_bayes.append(resultado_classificador)
				print("%d-Taxa de acertos (acuracia) %d-fold-cross-validation (particao %d): %.2f %%" % ((i*r+iteracao_fold_cross+1), r, (iteracao_fold_cross+1), resultado_classificador*100.0))

		print("\nDistancia de Hamming")
		lista_acuracia_hamming = []
		for i in range(repeticao):
			random.shuffle(dados)
			particoes = separar_fold_cross_validation(dados, r)

			for iteracao_fold_cross in range(r):
				treino = []
				teste = []

				for j in range(r):
					if(j == iteracao_fold_cross):
						teste = particoes[j]
					else:
						treino.extend(particoes[j])

				resultado_classificador = classificador_knn(treino, teste, k, funcao_distancia=distancia_hamming)
				lista_acuracia_hamming.append(resultado_classificador)
				print("%d-Taxa de acertos (acuracia) %d-fold-cross-validation (particao %d): %.2f %%" % ((i*r+iteracao_fold_cross+1), r, (iteracao_fold_cross+1), resultado_classificador*100.0))

		print("\n\nMEDIAS DAS TAXAS DE ACERTOS")
		print("Media acuracia com Naive Bayes: %.2f %%" % (media_lista(lista_acuracia_naive_bayes)*100.0))
		print("Media acuracia com distancia de Hamming: %.2f %%" % (media_lista(lista_acuracia_hamming)*100.0))


		print("\nDESVIO PADRAO DAS TAXAS DE ACERTOS")
		print("Desvio padrao da Naive Bayes: %.2f %%" % (desvio_padrao(lista_acuracia_naive_bayes)*100.0))
		print("Desvio padrao da distancia de Hamming: %.2f %%" % (desvio_padrao(lista_acuracia_hamming)*100.0))

	print('\n\n\nClassificando a base AMAZON com treinamento na classe YELP')
	resultado_classificador = classificador_naive_bayes(bases[2], bases[0])
	perc_erros = 1-resultado_classificador
	print('\nPercentual de erros: %.2f %%' % (perc_erros*100.0))

main()