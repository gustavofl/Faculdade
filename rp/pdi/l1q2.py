import cv2 as cv

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

def requantizar_e_salvar(img_origem, local_destino, k):
	img = cv.imread(img_origem, 0)
	req = requantizar(img, 2*k)
	cv.imwrite(local_destino, req)

def main():
	for img in range(1,4):
		for k in range(1,4):
			requantizar_e_salvar("imagens/img"+str(img)+".pgm", "imagens/requantizadas/img"+str(img)+"."+str(k)+".pgm", k)

main()