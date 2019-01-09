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

def ordenar_maior_face(lista):
	if(len(lista) >= 2):
		for i in range(len(lista)-1,0,-1):
			if(lista[i][-1] > lista[i-1][-1]):
				lista[i],lista[i-1] = lista[i-1],lista[i]

	return lista

def moda_faces(lista, com_peso):
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
	faces_moda=""
	for i in categorias:
		if(categorias[i] > maior):
			faces_moda = i
			maior = categorias[i]

	return faces_moda

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
				ordenar_maior_face(lista_proximos)

		resultado = lista_proximos[0][1]
		resultado = moda_faces(lista_proximos, com_peso)
		qnt_total += 1
		if(resultado == p[1]):
			qnt_sucesso += 1

	return float(qnt_sucesso)/float(qnt_total)

def ler_arquivo_csv(arquivo):
	arq = open(arquivo, "r")

	passageiros = arq.readlines()

	passageiros_dados = []

	for i in range(1,len(passageiros)):
		linha = passageiros[i]

		linha = linha.replace("\n","")
		linha = linha.split("\"")

		dados = []
		dados.extend(linha[0][:-1].split(","))
		dados.append(''.join(linha[1:-1]))
		dados.extend(linha[-1][1:].split(","))

		sobrevivente = dados.pop(1)

		passageiros_dados.append([dados,sobrevivente])

	return passageiros_dados

def tratar_dados_titanic(arquivo):
	resultado = []

	# Transformacao de dados categoricos para numericos
	for i,pessoa in enumerate(arquivo):
		dados_categ = pessoa[0]

		dados_num = []

		# ID
		dados_num.append(int(dados_categ[0])) 

		# Class
		classe = dados_categ[1]
		if(classe != ''):
			dados_num.append(int(classe))
		else:
			dados_num.append('#') # calcular media

		# NOME (Nao faz sentido usar o nome na classificacao)

		# Sex
		sex = dados_categ[3]
		if(sex == 'male'):
			dados_num.append(1)
		elif(sex == 'female'):
			dados_num.append(-1)
		else:
			dados_num.append('#') # calcular media

		# Age
		age = dados_categ[4]
		if(age != ''):
			dados_num.append(float(age))
		else:
			dados_num.append('#') # calcular media

		# sibsp
		sibsp = dados_categ[5]
		if(sibsp != ''):
			dados_num.append(int(sibsp))
		else:
			dados_num.append('#') # calcular media

		# parch
		parch = dados_categ[6]
		if(parch != ''):
			dados_num.append(int(parch))
		else:
			dados_num.append('#') # calcular media

		# Ticket
		ticket = dados_categ[7]
		if(ticket == 'LINE'):
			dados_num.append(0)
		elif(ticket != ''):
			ticket = ticket.split()
			try:
				dados_num.append(int(ticket[-1]))
			except:
				print(ticket)
		else:
			dados_num.append('#') # calcular media

		# Fare (tarifa)
		fare = dados_categ[8]
		if(fare != ''):
			dados_num.append(float(fare))
		else:
			dados_num.append('#') # calcular media

		# Cabin number
		cabin = dados_categ[9]
		if(cabin == ""):
			dados_num.append('#')
		else:
			# as vezes tem mais de uma cabine por pessoa
			cabin = cabin.split()

			# se tiver mais de uma, pegar a media
			media = 0
			for c in cabin:
				# obtem o numero da letra no alfabeto e multiplica por 200 (parece ter 200 cabines por letra)
				media += (ord(c[0])-65)*200

				# se possuir um numero, somar
				if(len(c) > 1): media += int(c[1:])
			media /= len(cabin)

			dados_num.append(media)

		# EMBARKED
		embarked = dados_categ[10]
		if(embarked == "S"):
			dados_num.append(1)
		elif(embarked == "C"):
			dados_num.append(2)
		elif(embarked == "Q"):
			dados_num.append(3)
		else:
			dados_num.append('#')

		resultado.append(dados_num)

	# tratamente atributos faltosos
	quantidades = [0 for i in range(len(resultado[0]))]
	medias = [0 for i in range(len(resultado[0]))]
	for dados in resultado:
		for i,carac in enumerate(dados):
			if(carac != '#'):
				quantidades[i] += 1
				medias[i] += carac
	for i in range(len(medias)):
		medias[i] = medias[i]/float(quantidades[i])

	for dados in resultado:
		for i in range(len(dados)):
			if(dados[i] == '#'):
				dados[i] = medias[i]

	# Padronizacao nos valores das caracteristicas
	caracteristicas = [[] for i in range(len(resultado[0]))]

	for dados in resultado:
		for i in range(len(dados)):
			caracteristicas[i].append(dados[i])

	medias = [media_lista(lista) for lista in resultado]
	desvios_padrao = [desvio_padrao(lista) for lista in resultado]

	for dados in resultado:
		for i in range(len(dados)):
			dados[i] = (dados[i]-medias[i])/desvios_padrao[i]

	return resultado

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
	dados = ler_arquivo_csv("train.csv")

	# add atributo para distancia
	for i in range(len(dados)):
		dados[i] = [dados[i][0],dados[i][1],0]

	k=1
	r=10

	print("\n\n10-fold-cross-validation (1-NN)")
	print("\nDistancia de Hamming")
	lista_acuracia_hamming = []
	for i in range(r):
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

	dados_num = tratar_dados_titanic(dados)

	# atualizar os atributos de categoricos para numericos
	for i in range(len(dados)):
		dados[i][0] = dados_num[i]

	print("\n\nDistancia Euclidiana")
	lista_acuracia_euclidiana = []
	for i in range(r):
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

			resultado_classificador = classificador_knn(treino, teste, k, funcao_distancia=distancia_euclidiana)
			lista_acuracia_euclidiana.append(resultado_classificador)
			print("%d-Taxa de acertos (acuracia) %d-fold-cross-validation (particao %d): %.2f %%" % ((i*r+iteracao_fold_cross+1), r, (iteracao_fold_cross+1), resultado_classificador*100.0))


	print("\n\nMEDIAS DAS TAXAS DE ACERTOS")
	print("Media acuracia com distancia de Hamming: %.2f %%" % (media_lista(lista_acuracia_hamming)*100.0))
	print("Media acuracia com distancia Euclidiana: %.2f %%" % (media_lista(lista_acuracia_euclidiana)*100.0))


	print("\nDESVIO PADRAO DAS TAXAS DE ACERTOS")
	print("Desvio padrao da distancia de Hamming: %.2f %%" % (desvio_padrao(lista_acuracia_hamming)*100.0))
	print("Desvio padrao da da distancia Euclidiana: %.2f %%" % (desvio_padrao(lista_acuracia_euclidiana)*100.0))

	print("\n\nTESTE DE WILCOXON (95%s de nivel de confianca)" % "%")
	z = wilcoxon(lista_acuracia_hamming,lista_acuracia_euclidiana)
	
	# DEBUG
	#print("z =",z)
	
	if(abs(z) < 1.96):
		print("Assume-se com 95%s de certeza que ambos os classificadores apresentam o mesmo desempenho" % "%")
	else:
		print("Assume-se com 95%s de certeza os classificadores nao apresentam o mesmo desempenho" % "%")

main()