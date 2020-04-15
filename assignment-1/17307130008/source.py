from handout import *
import numpy as np
import matplotlib as plot

mean = np.array([[-5, 0], [0, 5], [5, -5]])
cov = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]])
sample = normal_distribution_generate(mean, cov, 3, [100, 100, 100])
# outlier = np.array([])
save_dataset(sample, 'data.data')

model = LeastSquareDiscriminantModel()
model.train('one-vs-other')
model.display()
#
model2 = PerceptronModel()
model2.train()
model2.display()

model3 = LogisticDiscriminantModel()
loss_record = model3.train()
model3.display(loss_record)
print('model3 test accuracy:', model3.test())

model4 = LogisticGenerativeModel()
loss_record = model4.train()
model4.display(loss_record)
print('model4 test accuracy:', model4.test())
