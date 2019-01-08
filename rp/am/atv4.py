import math

def distancia_euclidiana(p, q):
	distancia = 0
	for i in range(4):
		distancia += (p[i] - q[i])**2
	distancia = math.sqrt(distancia)

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

def moda_iris(lista):
	categorias = {}

	for i in lista:
		iris = i[4]
		if(not iris in categorias):
			categorias[iris] = 0
		categorias[iris] += 1

	maior=0
	iris_moda=""
	for i in categorias:
		if(categorias[i] > maior):
			iris_moda = i

	return iris_moda

def separar_treino_teste(dados, qnt_treino):
	treino = []
	teste = []
	
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

def classificador_knn(dados, tam_treino, k):
	# treino = dados[:tam_treino]
	# teste = dados[tam_treino:]
	treino,teste = separar_treino_teste(dados, tam_treino)
	qnt_total = 0
	qnt_sucesso = 0

	for p in teste:
		
		lista_proximos = []
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

		resultado = moda_iris(lista_proximos)
		# qnt_total += 1
		if(resultado == p[4]):
			qnt_sucesso += 1

	return qnt_sucesso

print("classificacoes corretas 1-NN: %d" % classificador_knn(ler_arquivo_iris(), 25, 1))
print("classificacoes corretas 5-NN: %d" % classificador_knn(ler_arquivo_iris(), 25, 5))