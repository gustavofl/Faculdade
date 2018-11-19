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

def ILPF(tam_x, tam_y, corte):
	# Ideal Lowpass Filter
	filtro = np.zeros((tam_x,tam_y,2),np.uint8)

	fator_x = 2*math.pi/tam_x
	fator_y = 2*math.pi/tam_y

	for i in range(tam_x):
		for j in range(tam_y):
			if(distancia( ((i-tam_x/2)*fator_x,(j-tam_y/2)*fator_y) , (0,0) ) < corte):
				filtro[i][j] = 1

	return filtro

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

def butterworth_passa_alta(tam_x, tam_y, corte, n):
	# Filtro Butterworth passa baixa
	filtro = np.zeros((tam_x,tam_y,2),np.float32)

	fator_x = 2*math.pi/tam_x
	fator_y = 2*math.pi/tam_y

	for i in range(tam_x):
		for j in range(tam_y):
			d = distancia( ((i-tam_x/2)*fator_x,(j-tam_y/2)*fator_y) , (0,0) )
			filtro[i][j] = 1 - 1/float(1 + ( d/float(corte) )**(2*n))

	return filtro

def main(frequencia_de_corte):
	# lendo imagem
	img = cv.imread('imagens/palacios.jpg',0)

	# transformando a imagem para o dominio da frequencia
	dft = cv.dft(np.float32(img),flags = cv.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))

	### FILTRO ###########################

	# frequencia_de_corte = 3*math.pi/8

	# filtro passa baixa ideal
	# filtro = ILPF(img.shape[0], img.shape[1], frequencia_de_corte)

	# filtro Butterworth passa baixa
	# filtro = butterworth_passa_baixa(img.shape[0], img.shape[1], frequencia_de_corte, 2)

	# filtro Butterworth passa alta
	filtro = butterworth_passa_alta(img.shape[0], img.shape[1], frequencia_de_corte, 20)
	
	fshift = dft_shift * filtro

	# aplicar agucamento
	fshift = dft_shift + fshift

	######################################
	
	# transformando de volta para o dominio do espaco (tempo)
	f_ishift = np.fft.ifftshift(fshift)
	img_back = cv.idft(f_ishift)
	img_back = cv.magnitude(img_back[:,:,0],img_back[:,:,1])

	### PLOTANDO ##########################################
	# plt.imshow(filtro[:,:,0], cmap="gray")
	plt.imshow(img_back, cmap="gray"), plt.xticks([]), plt.yticks([])
	# plt.subplot(121), plt.imshow(img, cmap="gray")
	# plt.subplot(122), plt.imshow(img_back, cmap="gray")
	# plt.subplot(121), plt.imshow(img, cmap="gray")
	# plt.subplot(122), plt.imshow(img_back, cmap="gray")
	plt.show()
	# plt.imshow(img_back, cmap="gray"), plt.xticks([]), plt.yticks([])
	# plt.subplot(121), plt.imshow(img, cmap="gray")
	# plt.subplot(122), plt.imshow(img_soma, cmap="gray")
	# plt.show()

for corte in (math.pi/8, math.pi/4, 3*math.pi/8):
	main(corte)