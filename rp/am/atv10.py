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
		maior_prob = 0
		maior_classe = ''

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

def ler_arquivo(arquivo):
	arq = open(arquivo, "r")

	linhas = arq.readlines()

	caracteristicas = []

	for i in range(1,len(linhas)):
		linha = linhas[i]

		linha = linha.replace("\n","")

		dados = linha.split(",")

		rotulo = dados.pop(-1)

		caracteristicas.append([dados,rotulo])

	return caracteristicas

def wilcoxon_rank(l1, l2):
	rank = []

	assert len(l1) == len(l2)

	for i in range(len(l1)):
		diff = l2[i]-l1[i]
		rank.append([diff, abs(diff), 0.0])

	rank.sort(key=lambda x:x[1])

	valor_atual = rank[0]
	first_instance = 0
	for i,media in enumerate(rank):
		if(media[1] != valor_atual[1]):

			for j in range(first_instance,i):
				rank[j][2] = ((first_instance+1)+i)/2.0

			valor_atual = media
			first_instance = i

	for j in range(first_instance,i+1):
		rank[j][2] = ((first_instance+1)+(i+1))/2.0

	return rank

def wilcoxon(l1, l2):
	rank = wilcoxon_rank(l1,l2)

	rank.sort(key=lambda x:x[0])

	soma_media_positiva = 0
	soma_media_nula = 0
	soma_media_negativa = 0

	for media in rank:
		if(media[0] < 0): soma_media_negativa += media[2]
		elif(media[0] == 0): soma_media_nula += media[2]
		elif(media[0] > 0): soma_media_positiva += media[2]

	r_mais = soma_media_positiva + soma_media_nula/2.0
	r_menos = soma_media_negativa + soma_media_nula/2.0

	s = min(r_mais,r_menos)

	n = len(l1)

	z = (s-(1/4.0)*n*(n-1))/math.sqrt((1/24.0)*n*(n+1)*(2*n+1))

	return z

def main():
	dados = ler_arquivo("car.data")

	# add atributo para distancia
	for i in range(len(dados)):
		dados[i] = [dados[i][0],dados[i][1],0]

	k=1
	r=10
	repeticao=10

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

	print("\n\nTESTE DE WILCOXON (95%s de nivel de confianca)" % "%")
	z = wilcoxon(lista_acuracia_hamming,lista_acuracia_naive_bayes)
	
	# DEBUG
	#print("z =",z)
	
	if(abs(z) < 1.96):
		print("Assume-se com 95%s de certeza que ambos os classificadores apresentam o mesmo desempenho" % "%")
	else:
		print("Assume-se com 95%s de certeza os classificadores nao apresentam o mesmo desempenho" % "%")

main()