import numpy as np
import matplotlib.pyplot as plt

def plot(train_loss,train_acc, test_loss, test_acc):
    x = np.linspace(1,10,10)

    plt.figure(figsize=(20,10))
    plt.subplot(121)
    plt.rcParams['font.family'] = 'SimHei'

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(x, train_loss, label='train_loss', linestyle='--', color='tomato')
    plt.plot(x, test_loss, label='test_loss', linestyle='-.', color='blue')
    plt.legend()

    plt.subplot(122)
    plt.rcParams['font.family'] = 'SimHei'

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.plot(x, train_acc, label='train_accuracy', linestyle='--', color='tomato')
    plt.plot(x, test_acc, label='test_accuracy', linestyle='-.', color='blue')
    plt.legend()
    plt.show()

# lenet =  [0.9472531847133758, 0.9720342356687898, 0.9790007961783439, 0.9817874203821656, 0.9829816878980892, 0.9838773885350318, 0.9837778662420382, 0.9840764331210191, 0.9847730891719745, 0.9868630573248408]
# alexnet =  [0.9635748407643312, 0.9798964968152867, 0.9856687898089171, 0.9874601910828026, 0.9892515923566879, 0.9905453821656051, 0.9895501592356688, 0.990047770700637, 0.9902468152866242, 0.9911425159235668]
# vgg =  [0.973328025477707, 0.9901472929936306, 0.9897492038216561, 0.9895501592356688, 0.9909434713375797, 0.9922372611464968, 0.9924363057324841, 0.9934315286624203, 0.9932324840764332, 0.9920382165605095]
# googlenet =  [0.6418192675159236, 0.9383957006369427, 0.9716361464968153, 0.9827826433121019, 0.9839769108280255, 0.986265923566879, 0.9888535031847133, 0.988156847133758, 0.9878582802547771, 0.9888535031847133]
# resnet = [0.9817874203821656, 0.9827826433121019, 0.9898487261146497, 0.9896496815286624, 0.9914410828025477, 0.9931329617834395, 0.9920382165605095, 0.9938296178343949, 0.9912420382165605, 0.9925358280254777]
#
# x = np.linspace(1,10,10)
#
# plt.figure(figsize=(20,10))
# plt.rcParams['font.family'] = 'SimHei'
#
# plt.xlabel('epoch')
# plt.ylabel('test accuracy')
# plt.plot(x, lenet, label='LeNet', color='tomato')
# plt.plot(x, alexnet, label='AlexNet', color='blue')
# plt.plot(x, vgg, label='VGG-Net', color='gold')
# plt.plot(x, googlenet, label='GoogLeNet', color='darkcyan')
# plt.plot(x, resnet, label='ResNet', color='green')
# plt.legend()
# plt.show()