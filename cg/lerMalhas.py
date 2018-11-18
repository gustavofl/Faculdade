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

def main():
	arquivo = "hand-hybrid.off"

	vertices,faces = ler_off(arquivo)
	tabela_vertices,tabela_faces = construir_face_vertice(vertices,faces)

	print("faces")
	for i in tabela_faces:
		print(i)
	
	print("\nvertices")
	for i in tabela_vertices:
		print(i)

main()