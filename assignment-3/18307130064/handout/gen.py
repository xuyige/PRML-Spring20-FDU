import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random

def gendata ():
    mean1 = (1,1)
    cov1 = [[1,0],[0,1]]
    mean2 = (-2,3)
    cov2 = [[1,0],[0,1]]
    mean3 = (4,4)
    cov3 = [[1,0],[0,1]]

    num1 = 200
    num2 = 200
    num3 = 30

    dist1 = np.random.multivariate_normal(mean1,cov1,(num1))
    dist2 = np.random.multivariate_normal(mean2,cov2,(num2))
    dist3 = np.random.multivariate_normal(mean3,cov3,(num3))

    #print (dist3.shape)
    #for i in range(200):
        #print ( dist3[i][0] , np.exp(dist3[i][0]))
        #dist3[i][0] = np.exp(dist3[i][0])
        #dist3[i][1] = np.exp(dist3[i][1])
    
    out1 = dist1.transpose()
    out2 = dist2.transpose()
    out3 = dist3.transpose()

    plt.plot(out1[0],out1[1],'o')
    plt.plot(out2[0],out2[1],'o')
    plt.plot(out3[0],out3[1],'o')
    plt.show ()

    output = []
    for i in dist1:
        output.append ( (i[0],i[1]) )
    for i in dist2:
        output.append ( (i[0],i[1]) )
    for i in dist3:
        output.append ( (i[0],i[1]) )

    random.shuffle ( output )

    fle = open ("data.data" , "w")
    fle.write ( str(num1+num2+num3) + '\n' )
    for i in output:
        fle.write ( str(i[0]) + " " + str(i[1]) + "\n" )
    fle.close () 
