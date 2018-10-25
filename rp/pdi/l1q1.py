import cv2 as cv

def obter_imagens():
	dados = []

	for indice_imagem in range(3):
		nome_arquivo = "imagens/img"+str(indice_imagem+1)+".pgm"
		dados.append(cv.imread(nome_arquivo, 0))

	return dados

def requantizar(img, novo_q):
	nova_img = img.copy()

	razao = int(256/novo_q)
	for i in range(nova_img.shape[0]):
		for j in range(nova_img.shape[1]):
			peso_pixel = int(nova_img[i][j]/int(256/novo_q))
			fator = int(256/(novo_q-1) - 1)

			# correcao de aproximação feita pelo python
			if(peso_pixel >= novo_q):
				peso_pixel -= 1

			nova_img[i][j] = peso_pixel*fator
	return nova_img

def main():
	imagens = obter_imagens()

	for i,img in enumerate(imagens):
		cv.imshow("Original", img)

		for k in range(1,4):
			req = requantizar(img, 2*k)
			cv.imwrite("imagens/img"+str(i+1)+"."+str(k)+".pgm", req)

main()