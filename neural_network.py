import matplotlib.pyplot as plt

in_file = open('dataset/dataset.csv')


def read_data(in_file):
    file = in_file.readlines()
    data = []
    cluster = []
    for i in file:
        split_i = i.split(',')
        data.append(split_i[:2])
        cluster.append(float(split_i[2].strip()))
    return data, cluster


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
print(len(points), len(train_data), len(test_data))
