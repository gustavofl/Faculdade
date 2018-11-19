import math
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def main(var):
	img = cv.imread("imagens/img3.pgm", 0)

	print(var)
	bordas = cv.Canny(img, 140, 180, apertureSize=var, L2gradient=False)

	# filtro_butterworth = butterworth_passa_baixa(img.shape[0], img.shape[1], frequencia_de_corte, 2)
	# img_borrada = aplicar_filtro(img, filtro_butterworth)
	# img_borrada = reescale(img_borrada)

	# img_bordas = aplicar_sobel(img_borrada)

	### PLOTANDO ##########################################
	# plt.imshow(filtro[:,:,0], cmap="gray")
	plt.imshow(bordas, cmap="gray"), plt.xticks([]), plt.yticks([])
	# plt.subplot(121), plt.imshow(img, cmap="gray")
	# plt.subplot(122), plt.imshow(img_back, cmap="gray")
	# plt.subplot(121), plt.imshow(img_borrada, cmap="gray")
	# plt.subplot(122), plt.imshow(img_bordas, cmap="gray")
	# plt.subplot(122), plt.imshow(dy, cmap="gray")
	# plt.show()
	# plt.imshow(img_back, cmap="gray"), plt.xticks([]), plt.yticks([])
	# plt.subplot(121), plt.imshow(img, cmap="gray")
	# plt.subplot(122), plt.imshow(img_soma, cmap="gray")
	plt.show()

	# cv.imwrite("imagens/img-"+str(var)+".jpg", bordas)

for i in range(3,8,2):
	main(i)