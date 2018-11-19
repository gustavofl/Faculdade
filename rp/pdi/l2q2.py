import math
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def distancia(p1, p2):
	assert len(p1) == len(p2)

	d = 0
	for i in range(len(p1)):
		d += p1[i]**2 + p2[i]**2
	d = math.sqrt(d)

	return d

def butterworth_passa_baixa(tam_x, tam_y, corte, n):
	# Filtro Butterworth passa baixa
	filtro = np.zeros((tam_x,tam_y,2),np.float32)

	fator_x = 2*math.pi/tam_x
	fator_y = 2*math.pi/tam_y

	for i in range(tam_x):
		for j in range(tam_y):
			d = distancia( ((i-tam_x/2)*fator_x,(j-tam_y/2)*fator_y) , (0,0) )
			filtro[i][j] = 1/float(1 + ( d/float(corte) )**(2*n))

	return filtro

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

def aplicar_filtro(img, filtro):
	# transformando a imagem para o dominio da frequencia
	dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

	### FILTRO ###########################
	
	fshift = dft_shift * filtro

	######################################
	
	# transformando de volta para o dominio do espaco (tempo)
	f_ishift = np.fft.ifftshift(fshift)
	img_back = cv.idft(f_ishift)
	img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])

	return img_back

def aplicar_sobel(img):
	matriz_convolucao = [[-1,0,1],
						[-2,0,2],
						[-1,0,1]]

	img_dx = aplicar_convolucao(img, matriz_convolucao)

	matriz_convolucao_dy = [[-1,-2,-1],
							[ 0, 0, 0],
							[ 1, 2, 1]]

	img_dy = aplicar_convolucao(img, matriz_convolucao_dy)

	img_gradiente = img.copy()

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			img_gradiente[i][j] = math.sqrt( img_dx[i][j]**2 + img_dy[i][j]**2 )

	return img_gradiente

def reescale(img):
	maior = 0

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if(img[i][j] > maior):
				maior = img[i][j]

	resultado = img.copy()

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			resultado[i][j] = int( ( img[i][j] / float(maior) ) * 255 )

	return resultado

def main(frequencia_de_corte):
	# img = cv.imread("imagens/palacios.jpg", 0)
	img = cv.imread("imagens/img1.pgm", 0)

	filtro_butterworth = butterworth_passa_baixa(img.shape[0], img.shape[1], frequencia_de_corte, 2)
	img_borrada = aplicar_filtro(img, filtro_butterworth)
	img_borrada = reescale(img_borrada)

	img_bordas = aplicar_sobel(img_borrada)

	### PLOTANDO ##########################################
	# plt.imshow(filtro[:,:,0], cmap="gray")
	# plt.imshow(img_bordas, cmap="gray"), plt.xticks([]), plt.yticks([])
	# plt.subplot(121), plt.imshow(img, cmap="gray")
	# plt.subplot(122), plt.imshow(img_back, cmap="gray")
	# plt.subplot(121), plt.imshow(img_borrada, cmap="gray")
	# plt.subplot(122), plt.imshow(img_bordas, cmap="gray")
	# plt.subplot(122), plt.imshow(dy, cmap="gray")
	# plt.show()
	# plt.imshow(img_back, cmap="gray"), plt.xticks([]), plt.yticks([])
	# plt.subplot(121), plt.imshow(img, cmap="gray")
	# plt.subplot(122), plt.imshow(img_soma, cmap="gray")
	# plt.show()

	cv.imwrite("imagens/img-"+str(frequencia_de_corte)+".jpg", img_bordas)

for corte in (math.pi/8, math.pi/4, 3*math.pi/8):
	main(corte)