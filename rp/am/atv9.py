import math, time, threading
from random import randint
from multiprocessing import Process, Array
import multiprocessing

def distancia_euclidiana(p, q, tabela_distancias):
	if(tabela_distancias[p[2]][q[2]] != -1):
		return tabela_distancias[p[2]][q[2]]

	distancia = 0
	p_carac = p[0]
	q_carac = q[0]
	
	for i in range(len(p_carac)):
		distancia += (p_carac[i] - q_carac[i])**2
	distancia = math.sqrt(distancia)

	tabela_distancias[p[2]][q[2]] = distancia
	tabela_distancias[q[2]][p[2]] = distancia

	return distancia

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

def diff_lista(l1, l2):
	resultado = []

	assert len(l1) == len(l2)

	for i in range(len(l1)):
		resultado.append(l1[i] - l2[i])

	return resultado

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

def estrificar_aleatorio(dados):
	classes = []
	
	#DEBUG
	for i in range(40):
	#for i in range(7):
		classe_i = []

		#DEBUG
		for j in range(10):
		#for j in range(6):
			classe_i.append(dados[i*10+j])

		classes.append(classe_i)

	treino = []
	teste = []
	for classe in classes:
		for i in range(int(len(classe)/2)):
			indice = randint(0,len(classe)-1)
			treino.append(classe.pop(indice))
		teste.extend(classe)

	return treino, teste

def print_lista(lista):
	print('[')
	for i in lista:
		print('\t'+str(i))
	print(']')

def thread_classificador_knn(treino, teste, k, tabela_distancias, indice_thread, lista_resultado, com_peso):
	treino_original = treino
	qnt_sucesso = 0

	for i,p in enumerate(teste):
		
		lista_proximos = []
		treino = treino_original

		for j,q in enumerate(treino):
			distancia = distancia_euclidiana(p,q, tabela_distancias)

			if(len(lista_proximos) < k):
				q.append(distancia)
				lista_proximos.append(q)
			elif(distancia < lista_proximos[0][-1]):
				lista_proximos.pop()
				q.append(distancia)
				lista_proximos.append(q)
				ordenar_maior_face(lista_proximos)

		resultado = lista_proximos[0][1]
		resultado = moda_faces(lista_proximos, com_peso)
		if(resultado == p[1]):
			qnt_sucesso += 1

	lista_resultado[indice_thread] = qnt_sucesso

def classificador_knn(treino, teste, k, tabela_distancias, com_peso=False):
	treino_original = treino
	qnt_total = 0
	qnt_sucesso = 0

	for i,p in enumerate(teste):
		
		lista_proximos = []
		treino = treino_original

		for j,q in enumerate(treino):
			distancia = distancia_euclidiana(p,q, tabela_distancias)

			if(len(lista_proximos) < k):
				q.append(distancia)
				lista_proximos.append(q)
			elif(distancia < lista_proximos[0][-1]):
				lista_proximos.pop()
				q.append(distancia)
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

		passageiros_dados.append(dados)

	return passageiros_dados

def tratar_dados_titanic(arquivo):
	tickets = []
	resultado = []

	# Transformacao de dados categoricos para numericos
	for dados in arquivo:
		passageiro = [[]]

		for i in range(len(dados)):
			if(dados[i] == ''):
				dados[i] = '0'

		is_teste = int(len(dados) == 11)

		passageiro.append(dados[3-is_teste]) # NOME
		passageiro.append(int(dados[1])) # Survived

		passageiro[0].append(int(dados[0])) # ID
		passageiro[0].append(int(dados[2-is_teste])) # Class
		passageiro[0].append(int(dados[4-is_teste] == "male")) # Sex
		passageiro[0].append(float(dados[5-is_teste])) # Age
		passageiro[0].append(int(dados[6-is_teste])) # sibsp
		passageiro[0].append(int(dados[7-is_teste])) # parch

		# Ticket
		ticket = dados[8-is_teste].split(" ")
		try: 
			int(ticket[0])
			ticket[0] = "numero inteiro"
		except: 
			pass
		if(not ticket[0] in tickets):
			tickets.append(ticket[0])			
		passageiro[0].append(tickets.index(ticket[0]))

		passageiro[0].append(float(dados[9-is_teste])) # Fare (tarifa)
		
		# Cabin number
		cabin = dados[10-is_teste]
		if(cabin == "0"):
			passageiro[0].append(0)
		else:
			cabin = cabin.split()

			media = 0
			for c in cabin:
				media += (ord(c[0])-65)*200
				if(len(c) > 1): media += int(c[1:])
			media /= len(cabin)

			passageiro[0].append(media)

		# EMBARKED
		if(dados[11-is_teste] == "C"):
			passageiro[0].append(1)
		elif(dados[11-is_teste] == "Q"):
			passageiro[0].append(2)
		elif(dados[11-is_teste] == "S"):
			passageiro[0].append(3)
		else:
			passageiro[0].append(0)

		resultado.append(passageiro)

	# Padronizacao no valores dos atributos
	escalas = [0 for i in range(12)]
	for dados in resultado:
		for i,atributo in enumerate(dados[0]):
			if(atributo > escalas[i]):
				escalas[i] = atributo

	maior_escala = max(escalas)
	for dados in resultado:
		for i in range(len(dados[0])):
			dados[0][i] *= maior_escala/float(escalas[i])

	# DEBUG
	for dados in resultado:
		print(dados[1])

	return resultado, tickets

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

def intervalo_confianca(vetor):
	media = media_lista(vetor)
	desvio = desvio_padrao(vetor)

	inicio_intervalo = media-1.96*desvio
	final_intervalo = media+1.96*desvio

	return [inicio_intervalo, final_intervalo], media

def intervalo_possui_zero(intervalo):
	return intervalo[0] <= 0 and intervalo[1] >= 0

def sobreposicao_intervalos(intervalo1, intervalo2):
	interseccao_inicio_interv1 = intervalo1[0] >= intervalo2[0] and intervalo1[0] <= intervalo2[1]
	interseccao_final_inter1 = intervalo1[1] >= intervalo2[0] and intervalo1[1] <= intervalo2[1]
	interseccao_inicio_interv2 = intervalo2[0] >= intervalo1[0] and intervalo2[0] <= intervalo1[1]
	interseccao_final_interv2 = intervalo2[1] >= intervalo1[0] and intervalo2[1] <= intervalo1[1]

	return interseccao_inicio_interv1 or interseccao_final_inter1 or interseccao_inicio_interv2 or interseccao_final_interv2

def main():
	tabela_distancias = [Array("d", [-1 for j in range(400)]) for i in range(400)]

	dados_faces = obter_dados_faces()

	k=1

	print("\n\nHOLDOUT 50/50")
	print("\n1-NN")
	lista_acuracia_holdout_1_NN = []
	for i in range(100):
		treino,teste = estrificar_aleatorio(dados_faces)

		lista_resultado = Array("i", [0 for i in range(4)])

		# DEBUG
		time1 = time.time()

		threads = []
		qnt_threads = 4
		size_teste = int(len(teste)/qnt_threads)
		for i in range(qnt_threads):
			t=Process(target=thread_classificador_knn, args=[treino, teste[size_teste*i:size_teste*(i+1)], k, tabela_distancias, i, lista_resultado, True,])
			t.start()
			threads.append(t)

		for t in threads:
			t.join()

		time2 = time.time()
		diff_time = float(time2 - time1)

		resultado_classificador = sum(lista_resultado)/len(teste)

		lista_acuracia_holdout_1_NN.append(resultado_classificador)
		print("Taxa de acertos (acuracia) %d-NN com peso: %.2f %%  (%.2f segundos)" % (k, resultado_classificador*100.0, diff_time))

	k = 3

	print("\n3-NN")
	lista_acuracia_holdout_3_NN = []
	for i in range(100):
		treino,teste = estrificar_aleatorio(dados_faces)

		# DEBUG
		time1 = time.time()
		resultado_classificador = classificador_knn(treino, teste, k, tabela_distancias, com_peso=True)
		time2 = time.time()
		diff_time = float(time2 - time1)

		lista_acuracia_holdout_3_NN.append(resultado_classificador)
		print("Taxa de acertos (acuracia) %d-NN com peso: %.2f %%  (%.2f segundos)" % (k, resultado_classificador*100.0, diff_time))

	print("\n\nMEDIAS DAS TAXAS DE ACERTOS")
	print("Media acuracia holdout 1-NN: %.2f %%" % (media_lista(lista_acuracia_holdout_1_NN)*100.0))
	print("Media acuracia holdout 3-NN: %.2f %%" % (media_lista(lista_acuracia_holdout_3_NN)*100.0))


	print("\nDESVIO PADRAO DAS TAXAS DE ACERTOS")
	print("Desvio padrao da acuracia holdout 1-NN: %.2f %%" % (desvio_padrao(lista_acuracia_holdout_1_NN)*100.0))
	print("Desvio padrao da acuracia holdout 3-NN: %.2f %%" % (desvio_padrao(lista_acuracia_holdout_3_NN)*100.0))

	print("\n\nTESTE DE WILCOXON (95%s de nivel de confianca)" % "%")
	z = wilcoxon(lista_acuracia_holdout_1_NN,lista_acuracia_holdout_3_NN)
	
	# DEBUG
	#print("z =",z)
	
	if(abs(z) < 1.96):
		print("Assume-se com 95%s de certeza que ambos os classificadores apresentam o mesmo desempenho" % "%")
	else:
		print("Assume-se com 95%s de certeza os classificadores nao apresentam o mesmo desempenho" % "%")

	print("\n\nTESTE INTERVALO DE CONFIANCA (95%s de nivel de confianca)" % "%")
	vetor_diferenca = diff_lista(lista_acuracia_holdout_1_NN, lista_acuracia_holdout_3_NN)
	intervalo,media = intervalo_confianca(vetor_diferenca)
	if(intervalo_possui_zero(intervalo)):
		print("Nao e possivel afirmar nada sobre as taxas de acertos dos classificadores.")
	else:
		print("A media de acerto dos classificadores sao diferentes.")
		if(media < 0):
			print("\tO 3-NN tem uma taxa de acerto significantemente maior!")
		else:
			print("\tO 1-NN tem uma taxa de acerto significantemente maior!")

	print("\n\nTESTE DE SOBREPOSICAO DE INTERVALO DE CONFIANCA (95%s de nivel de confianca)" % "%")
	intervalo_1nn = intervalo_confianca(lista_acuracia_holdout_1_NN)[0]
	intervalo_3nn = intervalo_confianca(lista_acuracia_holdout_3_NN)[0]
	
	# DEBUG
	#print("Intervalos:")
	#rint("\t1-NN:", intervalo_1nn)
	#print("\t3-NN:", intervalo_3nn)

	if(sobreposicao_intervalos(intervalo_1nn, intervalo_3nn)):
		print("Há sobreposicao de intervalos, logo nao e possivel afirmar nada sobre as taxas de acertos dos classificadores.")
	else:
		print("Nao ha sobreposicao de intervalos:")
		if(intervalo_3nn[1] > intervalo_1nn[1]):
			print("\tO 3-NN tem uma taxa de acerto maior!")
		else:
			print("\tO 1-NN tem uma taxa de acerto maior!")

# main()

tratar_dados_titanic(ler_arquivo_csv("titanic/train.csv"))