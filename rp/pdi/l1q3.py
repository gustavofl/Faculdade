import cv2 as cv
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def encontrar_intervalo_maior_densidade(array):
	matriz_densidade = [[0 for i in range(256)] for j in range(256)]

	intervalo_minimo = 128

	# calcular densidades
	for i in range(len(array)-intervalo_minimo):
		for j in range(i+intervalo_minimo, len(array)):
			soma = 0
			for k in range(i,j+1):
				soma += array[k][0]
			matriz_densidade[i][j] = soma / (float(j-i+1)/float(256))

	# encontrar maior densidade
	maior,inicio_intervalo,fim_intervalo = 0,0,0
	for i in range(len(matriz_densidade)):
		for j in range(len(matriz_densidade[i])):
			if(matriz_densidade[i][j] > maior):
				inicio_intervalo,fim_intervalo = i,j
				maior = matriz_densidade[i][j]

	# print("teste")
	# interval = 40
	# for i in range(130,180):
	# 	print(i,i+interval,matriz_densidade[i][i+interval])

	return inicio_intervalo,fim_intervalo

def normalizar_histograma(hist, altura, largura):
	qnt_pixels = altura*largura
	soma = 0

	for i in range(len(hist)):
		hist[i][0] /= float(qnt_pixels)
		soma += hist[i][0]

	media = soma / float(len(hist))

	# remover picos
	for i in range(len(hist)):
		if(hist[i][0] > 10*media):
			hist[i][0] = media

	return hist

def normalizar_array(array):
	for i in range(len(array)):
		array[i] /= float(max(array))

	return array

def get_transformacao(L, param_linha_1, param_linha_2, inicio_intervalo_densidade, final_intervalo_densidade):
	# apelidos
	inicio = inicio_intervalo_densidade
	fim = final_intervalo_densidade

	transformacao = []
	l_baixo = get_linha([0,0],[255,param_linha_1])
	l_cima = get_linha([0,param_linha_2],[255,255])
	l_meio = get_linha( [inicio,get_y(l_baixo,inicio)] , [fim,get_y(l_cima,fim)])
	for i in range(L):
		if(i < inicio):
			transformacao.append(int(get_y(l_baixo,i)))
		elif(i <= fim):
			transformacao.append(int(get_y(l_meio,i)))
		else:
			transformacao.append(int(get_y(l_cima,i)))

	return transformacao

def get_y(linha, x):
	# linha = [a,b]
	return x*linha[0]+linha[1]

def get_linha(p1, p2):
	# p = [x,y]
	m = (p2[1]-p1[1])/(p2[0]-p1[0])
	a = m
	b = -m*p1[0] + p1[1]

	return [a,b]

def aplicar_transformacao(img, transformacao):
	img = img.copy()

	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			img[i][j] = transformacao[img[i][j]]

	return img

def main():
	img = cv.imread("imagens/img1.pgm", 0)
	# img_original = img.copy()

	hist = cv.calcHist([img],[0],None,[256],[0,256])
	# plt.plot(hist)
	hist = normalizar_histograma(hist, img.shape[0], img.shape[1])

	inicio,fim = encontrar_intervalo_maior_densidade(hist)

	transformacao = get_transformacao(len(hist),100,155,inicio,fim)

	nova_img = aplicar_transformacao(img, transformacao)
	
	hist_transf = cv.calcHist([nova_img],[0],None,[256],[0,256])
	plt.plot(hist_transf)
	hist_transf = normalizar_histograma(hist_transf, img.shape[0], img.shape[1])

	# plt.plot(hist)
	# plt.plot(transformacao)
	# plt.plot(hist_transf)
	transformacao = normalizar_array(transformacao)
	# plt.plot(transformacao)
	plt.show()

	cv.imshow("Imagem Original", img)
	cv.imshow("Imagem Transformada", nova_img)
	cv.waitKey(0)
	cv.destroyAllWindows()

	cv.imwrite("imagens/img-transformada.pgm", nova_img)
main()
