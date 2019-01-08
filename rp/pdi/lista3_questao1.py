import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import glob, os
import matplotlib.pyplot as plt
import numpy as np
import math

# camadas de saida
out1=2
out2=2

# tamanho dos filtros
f1=2
f2=3

# tamanho dos MaxPoolings
m1=1
m2=1

# tamanho da imagem na CNN
tam_img=90

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # conv layers
        self.conv1 = nn.Conv2d(3, out1, f1)
        self.conv2 = nn.Conv2d(out1, out2, f2)

        # fully connected layers
        # self.fc1 = nn.Linear((90-2*f+2) * (90-2*f+2) * out2, 200)
        tam1 = self.get_tam_fc() * self.get_tam_fc() * out2
        tam2 = self.get_tam_fc_2(tam1)
        self.fc1 = nn.Linear(tam1, tam2)
        # self.fc1 = nn.Linear(4 * 4 * 12, 20)
        self.fc2 = nn.Linear(tam2, 2)

        self.primeiraCamada = None
        self.segundaCamada = None

    def forward(self, x):
        x = F.relu(self.conv1(x))

        self.primeiraCamada = x

        # x = F.max_pool2d(x, m1)
        x = F.relu(self.conv2(x))

        self.segundaCamada = x

        # x = F.max_pool2d(x, m2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_tam_fc(self):
        return int((int((tam_img-f1+1)/m1)-f2+1)/m2)

    def get_tam_fc_2(self, num):
        # return int(math.sqrt(num))
        return 200


def carregar_e_separar_conjuntos(path, expected):
    fileList = glob.glob(path)

    trainSize = int(len(fileList)*0.8)
    for infile in fileList[:trainSize]:
        im = Image.open(infile)
        img_tensor = preprocess(im)
        img_tensor.unsqueeze_(0)
        if(img_tensor.size()[1] == 3):
            tensorTrain.append(img_tensor)
            trainExpected.append(torch.Tensor([expected]))

    validSize = int(len(fileList)*0.1)
    for infile in fileList[trainSize:trainSize+validSize]:
        im = Image.open(infile)
        img_tensor = preprocess(im)
        img_tensor.unsqueeze_(0)
        if(img_tensor.size()[1] == 3):
            tensorValid.append(img_tensor)
            validExpected.append(torch.Tensor([expected]))

    testSize = int(len(fileList)*0.1)
    for infile in fileList[trainSize+validSize+testSize:]:
        im = Image.open(infile)
        img_tensor = preprocess(im)
        img_tensor.unsqueeze_(0)
        if(img_tensor.size()[1] == 3):
            tensorTest.append(img_tensor)
            testExpected.append(torch.Tensor([expected]))

def mostrar_primeira_camada():
    np_array=net.primeiraCamada
    np_array=np_array[0,:,:,:]
    np_array=np_array.detach().numpy()
    for i in range(1):
        for j in range(out1):
            plt.subplot2grid((1, out1), (i, j)).imshow(np_array[i+j])

    plt.show()

def mostrar_ultima_camada():
    np_array=net.segundaCamada
    np_array=np_array[0,:,:,:]
    np_array=np_array.detach().numpy()
    for i in range(1):
        for j in range(out2):
            plt.subplot2grid((1, out2), (i, j)).imshow(np_array[i+j])

    plt.show()

net = Net()
print(net)

criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   # transforms.Resize(256),
   transforms.Resize(100),
   # transforms.Resize(60),
   # transforms.CenterCrop(240),
   transforms.CenterCrop(90),
   # transforms.CenterCrop(50),
   transforms.ToTensor(),
   normalize
])

tensorTrain = []
trainExpected = []

tensorValid = []
validExpected = []

tensorTest = []
testExpected = []

carregar_e_separar_conjuntos("101_ObjectCategories/garfield/*.jpg", [1.,0.])
carregar_e_separar_conjuntos("101_ObjectCategories/camera/*.jpg", [0.,1.])

input_net = torch.cat(tensorTrain, 0)
target = torch.cat(trainExpected, 0)

inputValidacao = torch.cat(tensorValid, 0)
targetValidacao = torch.cat(validExpected, 0)

inputTeste = torch.cat(tensorTest, 0)
targetTeste = torch.cat(testExpected, 0)

ultimaValidacao = 1

listaLoss = []
listaLossValid = []

epoch = 0
proxima_parada = 20
while(True):
    # in your training loop:
    optimizer.zero_grad()   # zero the gradient buffers

    output = net(input_net)
    loss = criterion(output, target)
    valueLoss = loss.data.tolist()
    listaLoss.append(valueLoss)

    outputValidacao = net(inputValidacao)
    validLoss = criterion(outputValidacao, targetValidacao)
    valueValidLoss = validLoss.data.tolist()
    listaLossValid.append(valueValidLoss)

    print((epoch+1), ', loss=', valueLoss, ', validacao=', valueValidLoss)

    if(epoch == 0):
        ultimaValidacao = valueValidLoss

    if(valueValidLoss > ultimaValidacao):
        break

    ultimaValidacao = valueValidLoss

    loss.backward()
    optimizer.step()    # Does the update

    if(epoch % proxima_parada == proxima_parada-1):
        continuar=input('continuar? ')
        if(continuar == 'n'):
            break
        try:
            n = int(continuar)
            proxima_parada = n
        except:
            pass

    epoch += 1

output = net(inputTeste)
loss = criterion(output, targetTeste)
print("\n>>> TESTE:",loss.data,"\n")

plt.plot(listaLoss)
plt.plot(listaLossValid)
plt.show()

############################################################
## TESTE SOBRE TODOS OS DADOS PARA A MATRIZ DE CONFUSAO

tensorTest.extend(tensorValid)
tensorTest.extend(tensorTrain)
input_net = torch.cat(tensorTest, 0)

testExpected.extend(validExpected)
testExpected.extend(trainExpected)
expected = torch.cat(testExpected, 0)

output = net(input_net)

output = output.data.tolist()
expected = expected.data.tolist()
matriz_confusao = [[0,0],[0,0]]

for i in range(len(output)):
    if(expected[i][0] == 1):
        if(output[i][0] > 0.5):
            matriz_confusao[0][0] += 1
        else:
            matriz_confusao[0][1] += 1
    elif(expected[i][1] == 1):
        if(output[i][1] > 0.5):
            matriz_confusao[1][1] += 1
        else:
            matriz_confusao[1][0] += 1

print(matriz_confusao)
# print('resultado test:', output)
# print('resultado esperado:', expected)

############################################################

mostrar = input('mostrar neuronios? ')
if(mostrar == 's'):
    im = Image.open("101_ObjectCategories/garfield.jpg")
    img_tensor = preprocess(im)
    img_tensor.unsqueeze_(0)
    print("predict test: ", net(img_tensor).data[0])

    mostrar_primeira_camada()
    mostrar_ultima_camada()


    im = Image.open("101_ObjectCategories/camera.jpg")
    img_tensor = preprocess(im)
    img_tensor.unsqueeze_(0)
    print("predict test: ", net(img_tensor).data[0])

    mostrar_primeira_camada()
    mostrar_ultima_camada()