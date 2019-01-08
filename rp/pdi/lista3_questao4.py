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
import copy

# camadas de saida
out1=2
out2=2
out3=4
out4=2

# tamanho dos filtros
f1=2
f2=2
f3=3
f4=3

# tamanho das matrizes MaxPolling
m1=2
m2=2
m3=2
m4=2

# tamanho da imagem na CNN
tam_img=240

# transformacoes que serao aplicadas as imagens
normalize = transforms.Normalize(
   mean=[0.485, 0.456, 0.406],
   std=[0.229, 0.224, 0.225]
)
preprocess = transforms.Compose([
   # transforms.Resize(256),
   transforms.Resize(tam_img+10),
   # transforms.Resize(60),
   # transforms.CenterCrop(240),
   transforms.CenterCrop(tam_img),
   # transforms.CenterCrop(50),
   transforms.ToTensor(),
   normalize
])

# criterio da loss function
criterion = nn.MSELoss()

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # conv layers
        self.conv1 = nn.Conv2d(3, out1, f1)
        self.conv2 = nn.Conv2d(out1, out2, f2)
        self.conv3 = nn.Conv2d(out2, out3, f3)
        self.conv4 = nn.Conv2d(out3, out4, f4)

        # fully connected layers
        tam1 = self.get_tam_fc_1() * self.get_tam_fc_1() * out4
        tam2 = self.get_tam_fc_2(tam1)
        self.fc1 = nn.Linear(tam1, tam2)
        self.fc2 = nn.Linear(tam2, 2)

        self.camadas = [None for i in range(8)]

    def forward(self, x):
        x = F.relu(self.conv1(x))

        self.camadas[0] = x

        x = F.max_pool2d(x,m1)

        self.camadas[1] = x

        x = F.relu(self.conv2(x))

        self.camadas[2] = x

        x = F.max_pool2d(x,m2)

        self.camadas[3] = x

        x = F.relu(self.conv3(x))

        self.camadas[4] = x

        x = F.max_pool2d(x,m3)

        self.camadas[5] = x

        x = F.relu(self.conv4(x))

        self.camadas[6] = x

        x = F.max_pool2d(x,m4)

        self.camadas[7] = x

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def mostrar_camadas(self):
        for camada in self.camadas:
            np_array=camada
            np_array=np_array[0,:,:,:]
            np_array=np_array.detach().numpy()
            for i in range(1):
                for j in range(camada[0].size()[0]):
                    plt.subplot2grid((1, camada[0].size()[0]), (i, j)).imshow(np_array[i+j])

            plt.show()

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def get_tam_fc_1(self):
        return int((int((int((int((tam_img-f1+1)/m1)-f2+1)/m2)-f3+1)/m3)-f4+1)/m4)

    def get_tam_fc_2(self, num):
        # return int(math.sqrt(num))
        return 40

def carregar_imagens(path):
    fileList = glob.glob(path)
    tensorList = []

    for infile in fileList:
        im = Image.open(infile)
        img_tensor = preprocess(im)
        img_tensor.unsqueeze_(0)
        if(img_tensor.size()[1] == 3):
            tensorList.append(img_tensor)

    return tensorList

def separar_conjunto(conjunto, expected):
    tensorTrain = []
    trainExpected = []

    tensorValid = []
    validExpected = []

    tensorTest = []
    testExpected = []

    tensorTotal = []
    totalExpected = []

    tam_conjunto = len(conjunto)
    lista = conjunto

    trainSize = int(tam_conjunto*0.8)
    for img in lista[:trainSize]:
        tensorTrain.append(img)
        trainExpected.append(torch.Tensor([expected]))

    validSize = int(tam_conjunto*0.1)
    for img in lista[trainSize:trainSize+validSize]:
        tensorValid.append(img)
        validExpected.append(torch.Tensor([expected]))

    testSize = int(tam_conjunto*0.1)
    for img in lista[trainSize+validSize+testSize:]:
        tensorTest.append(img)
        testExpected.append(torch.Tensor([expected]))

    tensorTotal.extend(tensorTrain)
    tensorTotal.extend(tensorValid)
    tensorTotal.extend(tensorTest)

    totalExpected.extend(trainExpected)
    totalExpected.extend(validExpected)
    totalExpected.extend(testExpected)

    tensorTrain = torch.cat(tensorTrain, 0)
    trainExpected = torch.cat(trainExpected, 0)
    tensorValid = torch.cat(tensorValid, 0)
    validExpected = torch.cat(validExpected, 0)
    tensorTest = torch.cat(tensorTest, 0)
    testExpected = torch.cat(testExpected, 0)
    tensorTotal = torch.cat(tensorTotal, 0)
    totalExpected = torch.cat(totalExpected, 0)

    return tensorTrain,trainExpected,tensorValid,validExpected,tensorTest,testExpected,tensorTotal,totalExpected

def carregar_bases():
    garfield = carregar_imagens("101_ObjectCategories/garfield/*.jpg")
    camera = carregar_imagens("101_ObjectCategories/camera/*.jpg")

    dados1 = separar_conjunto(garfield, [1.,0.])
    dados2 = separar_conjunto(camera, [0.,1.])

    inputTrain = torch.cat([dados1[0],dados2[0]], 0)
    targetTrain = torch.cat([dados1[1],dados2[1]], 0)

    inputValidacao = torch.cat([dados1[2],dados2[2]], 0)
    targetValidacao = torch.cat([dados1[3],dados2[3]], 0)

    inputTeste = torch.cat([dados1[4],dados2[4]], 0)
    targetTeste = torch.cat([dados1[5],dados2[5]], 0)

    inputTotal = torch.cat([dados1[6],dados2[6]], 0)
    targetTotal = torch.cat([dados1[7],dados2[7]], 0)

    return inputTrain,targetTrain,inputValidacao,targetValidacao,inputTeste,targetTeste,inputTotal,targetTotal

def calcular_matriz_confusao(net, base, expected):
    ############################################################
    ## TESTE SOBRE TODOS OS DADOS PARA A MATRIZ DE CONFUSAO
    
    output = net(base)

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

    return matriz_confusao

def criar_cnn(inputTrain, targetTrain, inputValidacao, targetValidacao):
    net = Net()
    print(net)
    print('m1 =',m1,'; m2 =',m2,'m3 =',m3,'; m4 =',m4)

    optimizer = optim.SGD(net.parameters(), lr=0.01)

    ultimaValidacao = 1
    validacaoUnderFitting = 0.5

    listaLoss = []
    listaLossValid = []

    melhor_cnn = Net()
    melhor_epoca = 1
    melhor_loss = 0.5

    epoch = 0
    # proxima_parada = 1
    # for i in range(10):
    while(True):
        # in your taining loop:
        optimizer.zero_grad()   # zero the gradient buffers

        output = net(inputTrain)
        loss = criterion(output, targetTrain)
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
            print('over fitting')
            # break

        ultimaValidacao = valueValidLoss

        loss.backward()
        optimizer.step()    # Does the update

        if(epoch % 10 == 10-1):
            criterioUnderFitting = 0.00008
            if(ultimaValidacao < 0.13):
                criterioUnderFitting = 0.000001
            if(validacaoUnderFitting - ultimaValidacao < criterioUnderFitting):
                print('under fitting')
                # break
            validacaoUnderFitting = ultimaValidacao

        if(ultimaValidacao < melhor_loss):
            melhor_cnn.load_state_dict(net.state_dict())
            # melhor_cnn = copy.deepcopy(net)
            melhor_loss = ultimaValidacao
            melhor_epoca = epoch

        arq = open('continuar.data','r')
        continuar = int(arq.readline())
        arq.close()

        if(not continuar):
            break

        epoch += 1

    # return net,ultimaValidacao,[listaLoss,listaLossValid]
    return melhor_cnn,melhor_loss,melhor_epoca,[listaLoss,listaLossValid]

def testar_cnn(net, inputTeste, targetTeste, inputTotal, targetTotal, aprendizado, epoca):
    output = net(inputTeste)
    loss = criterion(output, targetTeste)
    print("\n>>> TESTE:",loss.data,"\n")

    plt.plot(aprendizado[0][:epoca])
    plt.plot(aprendizado[1][:epoca])
    plt.show()

    print(calcular_matriz_confusao(net, inputTotal, targetTotal))

    mostrar = input('mostrar neuronios? ')
    if(mostrar == 's'):
        im = Image.open("101_ObjectCategories/garfield.jpg")
        img_tensor = preprocess(im)
        img_tensor.unsqueeze_(0)
        print("predict test: ", net(img_tensor).data[0])

        net.mostrar_camadas()


        im = Image.open("101_ObjectCategories/camera.jpg")
        img_tensor = preprocess(im)
        img_tensor.unsqueeze_(0)
        print("predict test: ", net(img_tensor).data[0])

        net.mostrar_camadas()

def main():
    global out1,out2,out3,out4,f1,f2,f3,f4,m1,m2,m3,m4

    inputTrain,targetTrain,inputValidacao,targetValidacao,inputTeste,targetTeste,inputTotal,targetTotal = carregar_bases()

    arq = open('melhor.data','r')
    melhor = float(arq.readline())
    arq.close()
    # melhor = 0.5
    # m1=a
    # out1=b
    # f1=c
    # m2=d
    # out2=e
    # f2=f
    # m3=g
    # out3=h
    # f3=p
    # m4=j
    # out4=k
    # f4=l
    net,loss,epoca,aprendizado = criar_cnn(inputTrain, targetTrain, inputValidacao, targetValidacao)

    if(loss < melhor):
        melhor = loss
        
        arq = open('melhor.data','w')
        print(melhor,file=arq)
        arq.close()

        arq = open('desempenho.data','a')
        print('\n>>> LOSS: ',loss,file=arq)
        print(net,file=arq)
        print('m1 =',m1,'; m2 =',m2,'m3 =',m3,'; m4 =',m4,'\n',file=arq)
        arq.close()

    if(loss <= 0.1 or True):
        testar_cnn(net, inputTeste, targetTeste, inputTotal, targetTotal, aprendizado, epoca)

main()