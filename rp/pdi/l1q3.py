import cv2 as cv
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def teste_hist():
	img = cv.imread("imagens/img3.pgm", 0)
	hist = cv.calcHist([img],[0],None,[256],[0,256])

	plt.hist(img, 3, facecolor='green')

	plt.show()

def main():
	img = cv.imread("imagens/img3.pgm", 0)
	hist = cv.calcHist([img],[0],None,[256],[0,256])
	print(hist)

teste_hist()
