import sys

def ler_off(arquivo):
	arq = open(arquivo, "r")

	arq.readline() # OFF

	quantidades = arq.readline().split()

	vertices = []

	for i in range(int(quantidades[0])):
		dados = [float(d) for d in arq.readline().split()]
		vertices.append(dados)

	faces = []

	for i in range(int(quantidades[1])):
		dados = [int(d) for d in arq.readline().split()]
		faces.append(dados)

	arq.close()

	return vertices,faces

def construir_face_vertice(vertices, faces):

	tabela_vertices = [[v,[]] for v in vertices]

	tabela_faces = faces

	for i,f in enumerate(faces):
		for v in f:
			if(not i in tabela_vertices[v][1]):
				tabela_vertices[v][1].append(i)

	return tabela_vertices,tabela_faces

def construir_winged_edge(vertices, faces):

	lista_faces = faces
	lista_vertices = [[v,[]] for v in vertices]
	lista_arestas = []

	# construir lista de arestas
	for indice_face, f in enumerate(lista_faces):
		for indice_vertice_face, vertice_atual in enumerate(f):
			vertice_anterior = f[indice_vertice_face-1]
			
			# procurar se ja existe esta aresta
			existe = False
			aresta = None
			for a in lista_arestas:
				v0 = a[0][0]
				v1 = a[0][1]
				if (vertice_anterior == v0 and vertice_atual == v1) or (vertice_anterior == v1 and vertice_atual == v0):
					existe = True
					aresta = a
					break

			# se nao existir uma entrada na lista de aresta: add aresta
			if not existe:
				indice_proxima_aresta = len(lista_arestas)
				lista_vertices[vertice_anterior][1].append(indice_proxima_aresta)
				lista_vertices[vertice_atual][1].append(indice_proxima_aresta)
				aresta = [[vertice_anterior,vertice_atual], [indice_face], [None,None,None,None]]
				lista_arestas.append(aresta)
			else: # se existir basta adicionar a face
				aresta[1].append(indice_face)

	# indicar arestas proximas a cada aresta
	for indice_aresta,aresta in enumerate(lista_arestas):
		

	return lista_faces, lista_vertices, lista_arestas

def p(var):
	print(var)
	assert False

def main():
	arquivo = ""

	if (len(sys.argv)) > 1:
		arquivo = sys.argv[1]
	else:
		arquivo = "triangles.off"

	vertices,faces = ler_off(arquivo)
	faces,vertices,arestas = construir_winged_edge(vertices,faces)

	print("faces")
	for i in faces:
		print(i)
	
	print("\nvertices")
	for i in vertices:
		print(i)
	
	print("\narestas")
	for i in arestas:
		print(i)
	print()

main()