import cv2 as cv
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def aplicar_convolucao(img, matriz_convolucao):
	conv = matriz_convolucao
	# tam = len(conv)
	tam = 3
	offset = int(tam/1)

	nova_img = img.copy()

	for i in range(len(img)):
		for j in range(len(img[i])):
			nova_img[i][j] = 0
			for k in range(tam):
				for l in range(tam):
					nova_img[i][j] += img[i+k-offset][j+l-offset]*conv[k][l]

	return nova_img

def main():
	img = cv.imread("imagens/img2.pgm", 0)

	cv.imshow("Imagem Original", img)

	matriz_convolucao = [[-1,-2,-1],
						[0,0,0],
						[1,2,1]]

	nova_img = aplicar_convolucao(img, matriz_convolucao)
	cv.imwrite("imagens/img2-conv1.pgm", nova_img)

	matriz_convolucao = [[-1,0,1],
						[-2,0,2],
						[-1,0,1]]

	nova_img = aplicar_convolucao(img, matriz_convolucao)
	cv.imwrite("imagens/img2-conv2.pgm", nova_img)

main()