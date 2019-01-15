import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import glob
import numpy
import matplotlib.pyplot as plt

# criterio da loss function
criterion = nn.MSELoss()


class Net(nn.Module):

	def __init__(self):
		super(Net, self).__init__()

		self.fc1 = nn.Linear(4, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))

		return x


def ler_arquivo(path):
	dados = []

	arq = open(path, 'r')
	texto = arq.readlines()
	for linha in texto :
		caracteristicas = linha.split(',')
		
		if(len(caracteristicas) == 1):
			break

		caracteristicas[0] = float(caracteristicas[0])
		caracteristicas[1] = float(caracteristicas[1])
		caracteristicas[2] = float(caracteristicas[2])
		caracteristicas[3] = float(caracteristicas[3])

		rotulo = caracteristicas[4].replace('\n','')

		dados.append([caracteristicas, rotulo])
	arq.close()

	return dados

# def carregar_dados(path) -> 3 tensorList (treino,teste,validacao)
def carregar_dados():
	# ler arquivo
	dados = ler_arquivo("iris.data")

	# treino. validacao, teste
	bases = [None, None, None]

	bases[0] = []
	bases[2] = []

	count = 0
	for ind_classe in range(3):
		for ind_base in (0,2):
			for i in range(25):
				bases[ind_base].append(dados[count])
				count += 1

	# converter as bases em tensores
	for ind_base in range(len(bases)):
		if(bases[ind_base] != None):
			bases[ind_base] = torch.cat(bases[ind_base], 0)

	return particoes

# def gerar_rotulos_esperados(???) -> tensorList

# def criar_cnn() -> net_data
# def treinar_cnn(net_data, inputTreino, expectedTreino, inputValidacao, expectedValidacao) -> net_data
def criar_cnn(particoes, net_dados=None):
	if(net_dados == None):
		net = Net()
		print(net)

		optimizer = optim.SGD(net.parameters(), lr=0.01)

		ultimaValidacao = 1
		validacaoUnderFitting = 0.5

		listaLoss = []
		listaLossValid = []

		melhor_cnn_dados = {'epoch':1,
							'model_state_dict': None,
							'optimizer_state_dict': None,
							'loss': 0.5}
		melhor_cnn = Net()

		epoch = 0
	else:
		net = Net()
		net.load_state_dict(net_dados['model_state_dict'])
		print(net)

		optimizer = optim.SGD(net.parameters(), lr=0.01)
		optimizer.load_state_dict(net_dados['optimizer_state_dict'])

		ultimaValidacao = net_dados['loss']
		validacaoUnderFitting = net_dados['loss']+0.01

		try:
			listaLoss = net_dados['listaLoss']
			listaLossValid = net_dados['listaLossValid']
		except:
			listaLoss = []
			listaLossValid = []

		melhor_cnn_dados = net_dados

		melhor_cnn = net

		epoch = net_dados['epoch']

	# arq = open('continuar.data','r')
	# continuar = int(arq.readline())
	# arq.close()

	# while(continuar):
	for i in range(30):
		optimizer.zero_grad()   # zero the gradient buffers

		output = net(particoes['treino']['imagens'])
		loss = criterion(output, particoes['treino']['expected'])
		valueLoss = loss.data.tolist()
		listaLoss.append(valueLoss)

		outputValidacao = net(particoes['validacao']['imagens'])
		validLoss = criterion(outputValidacao, particoes['validacao']['expected'])
		valueValidLoss = validLoss.data.tolist()
		listaLossValid.append(valueValidLoss)

		if(epoch == 0):
			ultimaValidacao = valueValidLoss

		if(valueValidLoss > ultimaValidacao):
			print('over fitting')
			# break

		print('epoca=%d\t\tloss=%.7f\tvalidacao=%.7f\tDiferenca=%.7f' % ((epoch+1), valueLoss, valueValidLoss, (ultimaValidacao-valueLoss)))

		ultimaValidacao = valueValidLoss

		loss.backward()
		optimizer.step()	# Does the update

		if(epoch % 10 == 10-1):
			criterioUnderFitting = 0.007
			if(ultimaValidacao < 0.13):
				criterioUnderFitting = 0.000001
			if(validacaoUnderFitting - ultimaValidacao < criterioUnderFitting):
				print('under fitting')
				# break
			validacaoUnderFitting = ultimaValidacao

		if(ultimaValidacao < melhor_cnn_dados['loss']):
			melhor_cnn_dados['epoch'] = epoch
			melhor_cnn_dados['model_state_dict'] = net.state_dict()
			melhor_cnn_dados['optimizer_state_dict'] = optimizer.state_dict()
			melhor_cnn_dados['loss'] = ultimaValidacao
			melhor_cnn.load_state_dict(net.state_dict())

		# arq = open('continuar.data','r')
		# continuar = int(arq.readline())
		# arq.close()

		epoch += 1

	salvar = input('Salvar rede neural? (s/n) ')
	if(salvar == 's'):
		nome_arq = input('Digite o nome do arquivo: ')
		melhor_cnn_dados['listaLoss'] = listaLoss
		melhor_cnn_dados['listaLossValid'] = listaLossValid
		torch.save(melhor_cnn_dados, nome_arq+'.pth')

	return melhor_cnn,melhor_cnn_dados,[listaLoss,listaLossValid]

# def mostrar_curva_aprendizagem(net_data) -> void

# def testar_cnn(net_data, inputTeste, expectedTeste) -> erro
def testar_cnn(net, aprendizado, net_dados, particoes):
	epoca = net_dados['epoch']

	# calcular os hash e vetor de caracteristicas de todas as imagens
	base_imagens = torch.cat([particoes['treino']['imagens'], particoes['validacao']['imagens']], 0)

	base_classes = torch.cat([particoes['treino']['expected'], particoes['validacao']['expected']], 0)

	net(base_imagens)

	base_classes = base_classes.data.tolist()

	base_hash = net.get_hash()

	base_vetor_carac = net.get_vetor_carac()

	base_treino = []
	for i in range(len(base_imagens)):
		base_treino.append(Imagem(base_imagens[i], base_classes[i], base_hash[i], base_vetor_carac[i]))

	net(particoes['teste']['imagens'])

	teste_hash = net.get_hash()

	teste_vetor_carac = net.get_vetor_carac()

	base_teste = []
	for i in range(len(particoes['teste']['imagens'])):
		base_teste.append(Imagem(particoes['teste']['imagens'][i], particoes['teste']['expected'][i], teste_hash[i], teste_vetor_carac[i]))

	# for img in base_teste:
	# 	proximos = get_proximos_limiar(base_treino, img, limiar)
	# print(len(proximos))

	ind_img_busca = 0

	proximos = get_proximos_limiar(base_treino, base_teste[ind_img_busca], limiar)

	semelhantes = get_k_proximos(base_treino, base_teste[ind_img_busca], k)

	plt.plot(aprendizado[0][:epoca])
	plt.plot(aprendizado[1][:epoca])
	plt.show()

	img = base_teste[ind_img_busca].imagem
	img = img[0,:]
	img = img.detach().numpy()
	plt.subplot2grid((1, 1), (0, 0)).imshow(img, cmap="gray")
	plt.show()

	for i in range(2):
		for j in range(3):
			ind_img = i*3+j

			img = semelhantes[ind_img].imagem
			img = img[0,:]
			img = img.detach().numpy()
			plt.subplot2grid((2, 3), (i, j)).imshow(img, cmap="gray")
	plt.show()

def main():
	global lista_rotulos

	###### PARA GERAR NOVOS ROTULOS
	lista_rotulos = get_labels('output/*.jpg', 3)
	salvar_rotulos(lista_rotulos)

	###### PARA USAR ROTULOS GRAVADOS
	# lista_rotulos = carregar_rotulos()

	particoes = carregar_bases()

	###### CARREGAR REDE
	# net_dados = torch.load('rede_240_1.pth')

	###### INICIAR NOVA REDE
	net_dados = None

	net,net_dados,aprendizado = criar_cnn(particoes, net_dados)

	testar_cnn(net, aprendizado, net_dados, particoes)

main()