import matplotlib.pyplot as plt
import random
import math

in_file = open('dataset/dataset.csv')


def read_data(in_file):
    file = in_file.readlines()
    data = []
    cluster = []
    for i in file:
        split_i = i.split(',')
        split_i[0] = float(split_i[0])
        split_i[1] = float(split_i[1])
        data.append(split_i[:2])
        cluster.append(float(split_i[2].strip()))
    return data, cluster


def sigmoid(x_input):
    return 1 / ((math.e ** -x_input) + 1)


data = read_data(in_file)
points = data[0]
clusters = data[1]
x = [float(x1[0]) for x1 in points]
y = [float(x1[1]) for x1 in points]
plt.scatter(x, y, c=clusters)
plt.show()
num_of_tests = int(len(points) * 0.75)
train_data = points[0:num_of_tests]
test_data = points[num_of_tests:]
params = [0, random.randint(1, 5), random.randint(1, 5)]
num_of_epochs = 1000
learning_rate = 0.001
grad = []
for i in range(num_of_epochs):
    grad = []
    for j in range(len(params)):
        grad.append(0)
    for k in range(len(train_data)):
        network_output = sigmoid((params[0]) + (train_data[k][0] * params[1]) + (train_data[k][1] * params[2]))
        cost = (network_output - clusters[k]) ** 2
        grad[0] += 2 * (network_output - clusters[k]) * network_output * (1 - network_output)
        grad[1] += 2 * (network_output - clusters[k]) * network_output * (1 - network_output) * train_data[k][0]
        grad[2] += 2 * (network_output - clusters[k]) * network_output * (1 - network_output) * train_data[k][1]
    for j in range(len(params)):
        params[j] = params[j] - (learning_rate * grad[j] / len(test_data))
print(params)
cost = 0
cluster_out = []
correct = 0
for i in range(len(test_data)):
    network_output = sigmoid((params[0]) + (train_data[i][0] * params[1]) + (train_data[i][1] * params[2]))
    cost += (network_output - clusters[300 + i]) ** 2
    if network_output < 0.1:
        cluster_out.append(0)
    else:
        cluster_out.append(1)
    if cluster_out[i] == clusters[300 + i]:
        correct += 1
print(correct / len(test_data))
x = [float(x1[0]) for x1 in test_data]
y = [float(x1[1]) for x1 in test_data]
plt.scatter(x, y, c=cluster_out)
plt.show()
num_of_epochs = 10000
learning_rate = 100
# b0, w0 , w1, b1, u0, u1, b2, v0, vi
params = [0, random.randint(1, 9), random.randint(1, 9), 0, random.randint(1, 9), random.randint(1, 9), 0,
          random.randint(1, 9), random.randint(1, 9)]
grad = []
for i in range(num_of_epochs):
    grad = []
    for j in range(len(params)):
        grad.append(0)
    for k in range(len(train_data)):
        a0 = sigmoid((params[0]) + (train_data[k][0] * params[1]) + (train_data[k][1] * params[2]))
        a1 = sigmoid((params[3]) + (train_data[k][0] * params[4]) + (train_data[k][1] * params[5]))
        network_output = sigmoid((params[6]) + (a0 * params[7]) + (a1 * params[8]))
        cost = (network_output - clusters[k]) ** 2
        grad[0] += 2 * (network_output - clusters[k]) * network_output * (1 - network_output) * params[7] * a0 * (
                1 - a0)
        grad[1] += 2 * (network_output - clusters[k]) * network_output * (1 - network_output) * train_data[k][0] * \
                   params[7] * a0 * (1 - a0)
        grad[2] += 2 * (network_output - clusters[k]) * network_output * (1 - network_output) * train_data[k][1] * \
                   params[7] * a0 * (1 - a0)
        grad[3] += 2 * (network_output - clusters[k]) * network_output * (1 - network_output) * params[8] * a1 * (
                    1 - a1)
        grad[4] += 2 * (network_output - clusters[k]) * network_output * (1 - network_output) * train_data[k][0] * \
                   params[8] * a1 * (1 - a1)
        grad[5] += 2 * (network_output - clusters[k]) * network_output * (1 - network_output) * train_data[k][1] * \
                   params[8] * a1 * (1 - a1)
        grad[6] += 2 * (network_output - clusters[k]) * network_output * (1 - network_output)
        grad[7] += 2 * (network_output - clusters[k]) * network_output * (1 - network_output) * train_data[k][0]
        grad[8] += 2 * (network_output - clusters[k]) * network_output * (1 - network_output) * train_data[k][1]
    for j in range(len(params)):
        params[j] = params[j] - (learning_rate * grad[j] / len(test_data))
cost = 0
cluster_out = []
correct = 0
for i in range(len(test_data)):
    a0 = sigmoid((params[0]) + (test_data[i][0] * params[1]) + (test_data[i][1] * params[2]))
    a1 = sigmoid((params[3]) + (test_data[i][0] * params[4]) + (test_data[i][1] * params[5]))
    network_output = sigmoid((params[6]) + (a0 * params[7]) + (a1 * params[8]))
    if network_output < 0.1:
        cluster_out.append(0)
    else:
        cluster_out.append(1)
    if cluster_out[i] == clusters[300 + i]:
        correct += 1
print(correct)
print(correct / len(test_data))
x = [float(x1[0]) for x1 in test_data]
y = [float(x1[1]) for x1 in test_data]
plt.scatter(x, y, c=cluster_out)
plt.show()
