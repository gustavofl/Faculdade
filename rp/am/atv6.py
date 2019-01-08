import math
from random import randint

def distancia_euclidiana(p, q):
	distancia = 0
	for i in range(4):
		distancia += (p[i] - q[i])**2
	distancia = math.sqrt(distancia)

	'''
	if(distancia == 0):
		print(p)
		print(q)
	'''

	return distancia

def indice_do_mais_proximo(treino, p):
	mais_proximo = 0
	distancia_mais_proximo = distancia_euclidiana(p,treino[0])

	for i,q in enumerate(treino[1:]):
		distancia = distancia_euclidiana(p,q)
		if(distancia < distancia_mais_proximo):
			mais_proximo = i
			distancia_mais_proximo = distancia

	return mais_proximo

def ler_arquivo_iris():
	dados = []

	arq = open('iris.data', 'r')
	texto = arq.readlines()
	for linha in texto :
		caracteristicas = linha.split(',')
		
		if(len(caracteristicas) == 1):
			break

		caracteristicas[0] = float(caracteristicas[0])
		caracteristicas[1] = float(caracteristicas[1])
		caracteristicas[2] = float(caracteristicas[2])
		caracteristicas[3] = float(caracteristicas[3])

		dados.append(caracteristicas)
	arq.close()

	return dados

def ordenar_maior(lista):
	if(len(lista) >= 2):
		for i in range(len(lista)-1,0,-1):
			if(lista[i] > lista[i-1]):
				lista[i],lista[i-1] = lista[i-1],lista[i]

	return lista

def ordenar_maior_iris(lista):
	if(len(lista) >= 2):
		for i in range(len(lista)-1,0,-1):
			if(lista[i][-1] > lista[i-1][-1]):
				lista[i],lista[i-1] = lista[i-1],lista[i]

	return lista

def moda_iris(lista, com_peso):
	categorias = {}

	for i in lista:
		iris = i[4]
		if(not iris in categorias):
			categorias[iris] = 0
		if(com_peso == False):
			categorias[iris] += 1
		else:
			if(i[-1] == 0): i[-1] = 0.000001

			valor = 1.0/float(i[-1])
			categorias[iris] += valor

	maior=0
	iris_moda=""
	for i in categorias:
		if(categorias[i] > maior):
			iris_moda = i
			maior = categorias[i]

	return iris_moda

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


def separar_treino_teste(dados, qnt_treino):
	treino = []
	teste = []
	
	# os 50 primeiros de cada classe são de treino, o resto é de teste

	tipo_iris_anterior = ""
	qnt_iris_atual=0
	for iris in dados:
		if(iris[4]==tipo_iris_anterior):
			if(qnt_iris_atual<qnt_treino):
				treino.append(iris)
				qnt_iris_atual+=1
			else:
				teste.append(iris)
		else:
			tipo_iris_anterior = iris[4]
			treino.append(iris)
			qnt_iris_atual = 1

	return treino,teste

def estrificar_aleatorio(dados):
	classe1 = dados[:50]
	classe2 = dados[50:100]
	classe3 = dados[100:]

	treino = []
	teste = []
	for classe in classe1,classe2,classe3:
		for i in range(25):
			indice = randint(0,len(classe)-1)
			treino.append(classe.pop(indice))
		teste.extend(classe)

	return treino, teste

def print_lista(lista):
	print('[')
	for i in lista:
		print('\t'+str(i))
	print(']')

def classificador_knn(treino, teste, k, com_peso=False):
	treino_original = treino
	qnt_total = 0
	qnt_sucesso = 0

	for p in teste:
		
		lista_proximos = []
		treino = treino_original

		for q in treino:
			distancia = distancia_euclidiana(p,q)
			if(len(lista_proximos) < k):
				q.append(distancia)
				lista_proximos.append(q)
			elif(distancia < lista_proximos[0][-1]):
				lista_proximos.pop()
				q.append(distancia)
				lista_proximos.append(q)
				ordenar_maior_iris(lista_proximos)

		resultado = moda_iris(lista_proximos, com_peso)
		qnt_total += 1
		if(resultado == p[4]):
			qnt_sucesso += 1

	return float(qnt_sucesso)/float(qnt_total)

k=50

dados_iris = ler_arquivo_iris()

print("\n50-NN")
treino,teste = separar_treino_teste(dados_iris, 25)

resultado_classificador = classificador_knn(treino, teste, k, com_peso=True)
acuracia_50nn_com_peso = resultado_classificador
print("Taxa de acertos (acuracia) %d-NN sem peso: %.2f %%" % (k, resultado_classificador*100.0))

resultado_classificador = classificador_knn(treino, teste, k)
acuracia_50nn_sem_peso = resultado_classificador
print("Taxa de acertos (acuracia) %d-NN com peso: %.2f %%" % (k, resultado_classificador*100.0))



print("\n\nHOLDOUT 50/50")
print("\nCOM PESO")
lista_acuracia_holdout_com_peso = []
for i in range(100):
	treino,teste = estrificar_aleatorio(dados_iris)
	resultado_classificador = classificador_knn(treino, teste, k, com_peso=True)
	lista_acuracia_holdout_com_peso.append(resultado_classificador)
	print("Taxa de acertos (acuracia) %d-NN com peso: %.2f %%" % (k, resultado_classificador*100.0))

print("\nSEM PESO")
lista_acuracia_holdout_sem_peso = []
for i in range(100):
	treino,teste = estrificar_aleatorio(dados_iris)
	resultado_classificador = classificador_knn(treino, teste, k)
	lista_acuracia_holdout_sem_peso.append(resultado_classificador)
	print("Taxa de acertos (acuracia) %d-NN sem peso: %.2f %%" % (k, resultado_classificador*100.0))



# print("Acuracia 50-NN com peso: %.2f %%" % (acuracia_50nn_com_peso*100.0))
# print("Acuracia 50-NN sem peso: %.2f %%" % (acuracia_50nn_sem_peso*100.0))


print("\n\nMEDIAS DAS TAXAS DE ACERTOS")
print("Media acuracia holdout com peso: %.2f %%" % (media_lista(lista_acuracia_holdout_com_peso)*100.0))
print("Media acuracia holdout sem peso: %.2f %%" % (media_lista(lista_acuracia_holdout_sem_peso)*100.0))


print("\nDESVIO PADRAO DAS TAXAS DE ACERTOS")
print("Desvio padrao da acuracia holdout com peso: %.2f %%" % (desvio_padrao(lista_acuracia_holdout_com_peso)*100.0))
print("Desvio padrao da acuracia holdout sem peso: %.2f %%" % (desvio_padrao(lista_acuracia_holdout_sem_peso)*100.0))

