import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import random

def gendata ():
    mean1 = (1,2)
    cov1 = [[1,0],[0,1]]
    mean2 = (5,5)
    cov2 = [[1,0],[0,1]]
    mean3 = (-2,3)
    cov3 = [[1,0],[0,1]]

    num1 = 150
    num2 = 150
    num3 = 150

    dist1 = np.random.multivariate_normal(mean1,cov1,(num1))
    dist2 = np.random.multivariate_normal(mean2,cov2,(num2))
    dist3 = np.random.multivariate_normal(mean3,cov3,(num3))
    
    out1 = dist1.transpose()
    out2 = dist2.transpose()
    out3 = dist3.transpose()

    plt.plot(out1[0],out1[1],'o')
    plt.plot(out2[0],out2[1],'o')
    plt.plot(out3[0],out3[1],'o')
    plt.show ()

    output = []
    for i in dist1:
        output.append ( (i[0],i[1],1) )
    for i in dist2:
        output.append ( (i[0],i[1],2) )
    for i in dist3:
        output.append ( (i[0],i[1],3) )

    random.shuffle ( output )

    fle = open ("data.data" , "w")
    output = []
    for i in dist1:
        output.append ( (i[0],i[1],1) )
    for i in dist2:
        output.append ( (i[0],i[1],2) )
    for i in dist3:
        output.append ( (i[0],i[1],3) )
    random.shuffle ( output )
    fle.write ( str(num1+num2+num3) + '\n' )
    for i in output:
        fle.write ( str(i[0]) + " " + str(i[1]) + " " + str(i[2]) + "\n" )
    fle.close () 

def getdata ():
    fle = open ( "data.data" , "r" )
    lines = int(fle.readline())
    #print ( lines )
    retx = np.empty ( (lines,2) )
    rety = np.empty ( (lines,1) )
    for i in range ( lines ):
        (retx[i][0],retx[i][1],rety[i]) = fle.readline().split()
    rety=rety.astype ( int )

    fle.close ()
    return (lines,retx,rety)

'''
Generative model
'''

def cal ( n , x , y ):
    pi = np.zeros ( (3) )
    mu = np.zeros ( (3,2) )
    sigma = np.zeros ( (2,2) )
    for i in range ( n ):
        pi[y[i]-1] += 1
        mu[y[i]-1] += x[i]
    for i in range ( 3 ):
        mu[i] /= pi[i]
        pi[i] /= n
    for i in range ( n ):
        sigma += np.dot ( (x[i]-mu[y[i]-1]).transpose() , (x[i]-mu[y[i]-1]) )
    sigma /= n
    return (pi,mu,sigma)

def predictgen ( nowx , pi , mu , sigma ):
    prob = np.zeros ( 3 )
    sum = 0.0
    for i in range ( 3 ):
        prob[i] = stats.multivariate_normal ( mu[i] , sigma ).pdf ( nowx ) * pi[i]
        sum += prob[i]
    prob /= sum
    maxx = -1.0
    maxi = 0
    for i in range ( 3 ):
        if maxx < prob[i]:
            maxx=prob[i]
            maxi = i
    return maxi + 1


def testgen ( n , x , y , pi , mu , sigma ):
    prd = np.zeros ( n )
    acc = 0
    for i in range ( n ):
        prd[i] = predictgen ( x[i] , pi , mu , sigma )
        if prd[i] == y[i]: acc += 1

    print ( "Generative model" )
    print ( "Accuracy:" , acc / n , "\n" )

    out = x.transpose ()
    col = ['ro','bo','go']
    wrong = ['rx','bx','gx']
    plt.figure(figsize=(10,10))
    for i in range ( n ):
        if y[i][0] == int(prd[i]):
            plt.plot ( out[0][i] , out[1][i] , col[y[i][0]-1] )
        else:
            plt.plot ( out[0][i] , out[1][i] , wrong[y[i][0]-1] )
    plt.show ()

    print ( "Wrong cases:" )
    for i in range ( n ):
        if y[i][0] != int(prd[i]):
            print ( x[i] , "real:" , y[i][0] , "predict:" , int(prd[i]) )
    print ()

    return


'''
Discriminative model
'''

def sigmoid ( x ):
    return 1.0/(1+np.exp(-x))

def predict_sgd ( j , nowx , w , w0 ):
    #print ( nowx , w[j] , np.dot ( nowx , w[j] ) )
    return sigmoid (np.dot ( nowx , w[j] ) + w0[j])

def sgd ( n , x , y , batch , epoch , alpha ):
    w = np.zeros ( (3,2) )
    w0 = np.zeros ( (3,1) )
    prd = np.zeros ( (3,n) ) 
    dec = alpha / epoch

    zp = list(zip(x,y))
    x,y = zip (*zp)

    now = 0
    for ep in range ( epoch ):
        random.shuffle ( zp )
        for i in range ( batch ):
            now += 1
            if now == n:
                now = 0
                break
            dw = np.zeros ( (3,2) )
            dw0 = np.zeros ( (3,1) )
            for j  in range ( 3 ):
                prd[j][now] = predict_sgd ( j , x[now] , w , w0 )
                dw[j] += ((1 if y[now]==j+1 else 0)-prd[j][now]) * x[now]
                dw0[j] += ((1 if y[now]==j+1 else 0)-prd[j][now])
            #print ( dw )
            #print ( dw0 )
            w += dw/batch * alpha
            w0 += dw0/batch * alpha
        alpha -= dec
    return (w,w0)


def predictdis ( nowx , w , w0 ):
    prob = np.zeros ( 3 )
    sum = 0.0
    for i in range ( 3 ):
        prob[i] = sigmoid(np.dot ( w[i] , nowx ) + w0[i])
        sum += prob[i]
    prob /= sum
    maxx = -1.0
    maxi = 0
    for i in range ( 3 ):
        if maxx < prob[i]:
            maxx=prob[i]
            maxi = i
    return maxi + 1


def testdis ( n , x , y , w , w0 ):
    prd = np.zeros ( n )
    acc = 0
    for i in range ( n ):
        prd[i] = predictdis ( x[i] , w , w0 )
        if prd[i] == y[i]: acc += 1

    print ( "Discriminative model" )
    print ( "Accuracy:" , acc / n , "\n" )

    out = x.transpose ()
    col = ['ro','bo','go']
    wrong = ['rx','bx','gx']
    plt.figure(figsize=(10,10))
    for i in range ( n ):
        if y[i][0] == int(prd[i]):
            plt.plot ( out[0][i] , out[1][i] , col[y[i][0]-1] )
        else:
            plt.plot ( out[0][i] , out[1][i] , wrong[y[i][0]-1] )
    plt.show ()
    
    print ( "Wrong cases:" )
    for i in range ( n ):
        if y[i][0] != int(prd[i]):
            print ( x[i] , "real:" , y[i][0] , "predict:" , int(prd[i]) )
    print ()

    return

gendata()

(n,x,y) = getdata ()

(pi,mu,sigma) = cal ( n , x , y )

testgen ( n , x , y , pi , mu , sigma )

epoch = 2000
batch = 10
alpha = 1e0
(w,w0) = sgd ( n , x , y , batch , epoch , alpha )

testdis ( n , x , y , w , w0 )