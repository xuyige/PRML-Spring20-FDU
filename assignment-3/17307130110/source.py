import numpy as np
from matplotlib import pyplot as plt
from handout.data import generate_2D_data, generate_3D_data,draw_2D_data
from handout.EM import kmeans, myEM


K, data, label = generate_2D_data(1000)
km_pi, km_mean, km_cov, km_label = kmeans(data, K, max_epoch=1000)
em_pi, em_mean, em_cov, em_label = myEM(data, K, max_epoch=1000)

draw_2D_data(data, label, 1, "Real")
draw_2D_data(data, km_label, 2, "Kmeans")
draw_2D_data(data, em_label, 3, "EM")

print("\n")

print("2D------ Kmeans ------")
print("km_pi is ")
print(km_pi)
print("km_mean is ")
print(km_mean)
print("km_cov is ")
print(km_cov)

print("\n")

print("2D------ EM ------")
print("em_pi is ")
print(em_pi)
print("em_mean is ")
print(em_mean)
print("em_cov is ")
print(em_cov)

print("\n")


K, data, label = generate_3D_data(1000)
km_pi, km_mean, km_cov, km_label = kmeans(data, K, max_epoch=1000)
em_pi, em_mean, em_cov, em_label = myEM(data, K, max_epoch=1000)

print("\n")

print("3D------ Kmeans ------")
print("km_pi is ")
print(km_pi)
print("km_mean is ")
print(km_mean)
print("km_cov is ")
print(km_cov)

print("\n")

print("3D------ EM ------")
print("em_pi is ")
print(em_pi)
print("em_mean is ")
print(em_mean)
print("em_cov is ")
print(em_cov)


plt.show()

