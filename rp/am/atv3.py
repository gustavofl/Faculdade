import math

treino = [[5.1 , 3.5 , 1.4 , 0.2],
		 [4.7 , 3.2 , 1.3 , 0.2],
		 [5.8 , 2.7 , 3.9 , 1.2],
		 [6.0 , 2.7 , 3.9 , 1.2],
		 [6.3 , 2.9 , 5.6 , 1.8],
		 [6.5 , 3.0 , 5.8 , 2.2]]

classes = ["iris-setosa","iris-setosa","iris-versiculor","iris-versiculor","iris-virginica","iris-virginica"]

teste = [[5.4 , 3.0 , 4.5 , 1.5],
		 [7.1 , 3.0 , 5.9 , 2.1],
		 [4.9 , 3.0 , 1.4 , 0.2]]

def distancia_euclidiana(p, q):
	if(len(p) != len(q)):
		return 0

	distancia = 0
	for i in range(len(p)):
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


def classificador_1nn(treino, classes, teste):
	resultado_teste = []

	for p in teste:
		mais_proximo = indice_do_mais_proximo(treino, p)
		resultado_teste.append(classes[mais_proximo])

	return resultado_teste

print(classificador_1nn(treino, classes, teste))