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


class Base():
	def __init__(self):
		self.inputTreino = None
		self.expectedTreino = None
		self.inputValidacao = None
		self.expectedValidacao = None
		self.inputTeste = None
		self.expectedTeste = None


# def ler_arquivo(path) -> lista de [vetor_caracteristica , rotulo]
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

		dados.append([caracteristicas[:4], rotulo])
	arq.close()

	return dados

# def carregar_dados(path) -> base
def carregar_dados(classe_1 = 0):
	# ler arquivo
	dados = ler_arquivo("iris.data")

	# treino. validacao, teste
	base = Base()

	base.inputTreino = []
	base.inputTeste = []

	base.expectedTreino = []
	base.expectedTeste = []

	count = 0
	for ind_classe in range(3):
		for i in range(25):
			base.inputTreino.append(dados[count][0])
			if(ind_classe == classe_1):
				base.expectedTreino.append([1.0])
			else:
				base.expectedTreino.append([0.0])
			count += 1

		for i in range(25):
			base.inputTeste.append(dados[count][0])
			if(ind_classe == classe_1):
				base.expectedTeste.append([1.0])
			else:
				base.expectedTeste.append([0.0])
			count += 1

	# converter lista em Tensor
	base.inputTreino = torch.Tensor(base.inputTreino)
	base.inputTeste = torch.Tensor(base.inputTeste)
	base.expectedTreino = torch.Tensor(base.expectedTreino)
	base.expectedTeste = torch.Tensor(base.expectedTeste)

	return base

# def gerar_rotulos_esperados(???) -> tensorList

# def criar_cnn() -> net_data
def criar_cnn(learning_rate):
	net = Net()
	print(net)

	optimizer = optim.SGD(net.parameters(), lr=learning_rate)

	net_data = {'epoch':1,
				'model_state_dict': net.state_dict(),
				'learning_rate': learning_rate,
				'optimizer_state_dict': optimizer.state_dict(),
				'loss': 0.5,
				'listaLoss': [],
				'listaLossValid': []}

	return net_data

# def treinar_cnn(net_data, base) -> net_data
def treinar_cnn(net_data, base, epoca_limite=None):
	net = Net()
	net.load_state_dict(net_data['model_state_dict'])

	optimizer = optim.SGD(net.parameters(), lr=net_data['learning_rate'])
	optimizer.load_state_dict(net_data['optimizer_state_dict'])

	ultimaValidacao = net_data['loss']
	validacaoUnderFitting = net_data['loss']+0.01

	listaLoss = net_data['listaLoss']
	listaLossValid = net_data['listaLossValid']

	melhor_cnn = net

	epoch = net_data['epoch']

	valueLoss = 0

	arq = open('continuar.data','r')
	continuar = int(arq.readline())
	arq.close()

	while((epoca_limite == None and continuar) or (epoca_limite != None and epoch < epoca_limite)):
		optimizer.zero_grad()   # zero the gradient buffers

		output = net(base.inputTreino)
		loss = criterion(output, base.expectedTreino)
		valueLoss = loss.data.tolist()
		listaLoss.append(valueLoss)

		# verificar se havera validacao
		if(base.inputValidacao != None and base.expectedValidacao != None):
			outputValidacao = net(base.inputValidacao)
			validLoss = criterion(outputValidacao, base.expectedValidacao)
			valueValidLoss = validLoss.data.tolist()
			listaLossValid.append(valueValidLoss)

			if(epoch == 1):
				ultimaValidacao = valueValidLoss

			if(valueValidLoss > ultimaValidacao):
				print('over fitting')
				# break

			# print('epoca=%d\t\tloss=%.7f\tvalidacao=%.7f\tDiferenca=%.7f' % (epoch, valueLoss, valueValidLoss, (ultimaValidacao-valueLoss)))
	
			ultimaValidacao = valueValidLoss
		else:			
			# print('epoca=%d\t\tloss=%.7f\t' % (epoch, valueLoss))
			pass

		loss.backward()
		optimizer.step()	# Does the update

		if(epoch % 1000 == 1000-1):
			criterioUnderFitting = 0.0000001
			if(validacaoUnderFitting - valueLoss < criterioUnderFitting):
				print('under fitting - epoca %d - taxa de erro = %.9f' % (epoch, valueLoss))
				break
			validacaoUnderFitting = valueLoss

		if(valueLoss < net_data['loss']):
			net_data['epoch'] = epoch
			net_data['model_state_dict'] = net.state_dict()
			net_data['optimizer_state_dict'] = optimizer.state_dict()
			net_data['loss'] = valueLoss
			net_data['listaLoss'] = listaLoss
			net_data['listaLossValid'] = listaLossValid
			melhor_cnn.load_state_dict(net.state_dict())

		arq = open('continuar.data','r')
		continuar = int(arq.readline())
		arq.close()

		epoch += 1

	print('Treinamento da rede neural finalizada: epoca %d - taxa de erro = %.9f' % (epoch, valueLoss))

	return net_data

# def mostrar_curva_aprendizagem(net_data) -> void
def mostrar_curva_aprendizagem(net_data):
	listaLoss = net_data['listaLoss']
	listaLossValid = net_data['listaLossValid']

	plt.plot(listaLoss)
	plt.plot(listaLossValid)
	plt.show()

# def testar_cnn(net_data, base) -> erro
def testar_cnn(net_data, base):
	net = Net()
	net.load_state_dict(net_data['model_state_dict'])

	output = net(base.inputTeste)
	loss = criterion(output, base.expectedTeste)
	print(">>> TESTE:",loss.data.tolist(),"\n\n")

	mostrar_curva_aprendizagem(net_data)    

def main():
	base_A = carregar_dados(classe_1=0)
	base_B = carregar_dados(classe_1=2)

	for lr in (0.01, 1.0, 10.0):
		net = criar_cnn(learning_rate=lr)

		net_A = treinar_cnn(net, base_A)
		testar_cnn(net_A, base_A)

		net_B = treinar_cnn(net, base_B, epoca_limite=100)
		testar_cnn(net_B, base_B)

main()