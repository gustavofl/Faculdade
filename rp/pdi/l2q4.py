import math
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

def preencher_buraco(img, kernel, x0):
	fila = []
	fila.append(x0)

def verificar_vizinhos(img, x, y):
	resultado = []
	vizinhos = [[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]]

	if(img[x][y] == 255):
		return resultado

	for v in vizinhos:
		try: 
			celula = img[x+v[0]][y+v[1]]
			if(not celula in (0,255)):
				if(not celula in resultado):
					resultado.append(celula)
		except:
			pass

	return resultado

def detectar_objetos(img):
	img_componentes = img.copy()
	prox_componente = 1
	equivalencia = {}

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			if(img[i][j] == 0):
				continue

			vizinhos = verificar_vizinhos(img, i, j)
			if(len(vizinhos) == 0):
				img_componentes[i][j] = prox_componente
				equivalencia[prox_componente] = [prox_componente]
				prox_componente += 1
			elif(len(vizinhos) == 1):
				img_componentes[i][j] = vizinhos[0]
			else:
				img_componentes[i][j] = vizinhos[0]
				while(len(vizinhos) > 1):
					if(vizinhos[0] in equivalencia[vizinhos[1]]):
						vizinhos.pop(0)
					else:
						for v in equivalencia[vizinhos[1]]:
							pass

def main():
	img = cv.imread("imagens/xicara.jpeg", 0)

	bordas = cv.Canny(img, 80, 220, apertureSize=3, L2gradient=False)
	
	detectar_objetos(bordas)

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

	# cv.imwrite("imagens/img-result.jpg", bordas)

main()