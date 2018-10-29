import cv2 as cv
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def normalizar_histograma(hist, altura, largura):
	qnt_pixels = altura*largura
	soma = 0

	for i in range(len(hist)):
		hist[i][0] /= float(qnt_pixels)
		soma += hist[i][0]

	media = soma / float(len(hist))

	return hist

def normalizar_array(array):
	for i in range(len(array)):
		array[i] /= float(max(array))

	return array

def get_transformacao(a, b, c, d, L=256):
	# verificar valores dos argumentos
	assert a <= c

	transformacao = []
	l1 = get_linha( [0,0] , [a,b] )
	l2 = get_linha( [a,b] , [c,d] )
	l3 = get_linha( [c,d] , [255,255] )
	for i in range(L):
		if(i < a):
			transformacao.append(int(get_y(l1,i)))
		elif(i <= c):
			transformacao.append(int(get_y(l2,i)))
		else:
			transformacao.append(int(get_y(l3,i)))
		print(transformacao[-1])

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

def get_valores():
	a = int(input("Digite o valor de a: "))
	b = int(input("Digite o valor de b: "))
	c = int(input("Digite o valor de c: "))
	d = int(input("Digite o valor de d: "))

	return a,b,c,d

def main():
	img = cv.imread("imagens/img1.pgm", 0)

	hist = cv.calcHist([img],[0],None,[256],[0,256])
	# plt.plot(hist)
	hist = normalizar_histograma(hist, img.shape[0], img.shape[1])

	a,b,c,d = get_valores()
	transformacao = get_transformacao(a,b,c,d)

	nova_img = aplicar_transformacao(img, transformacao)
	
	hist_transf = cv.calcHist([nova_img],[0],None,[256],[0,256])
	# plt.plot(hist_transf)
	hist_transf = normalizar_histograma(hist_transf, img.shape[0], img.shape[1])

	# plt.plot(hist)
	# plt.plot(transformacao)
	# plt.plot(hist_transf)
	# transformacao = normalizar_array(transformacao)
	plt.plot(transformacao)
	plt.show()

	cv.imshow("Imagem Original", img)
	cv.imshow("Imagem Transformada", nova_img)
	cv.waitKey(0)
	cv.destroyAllWindows()

	# cv.imwrite("imagens/img-transformada.pgm", nova_img)

main()