from handout import *

def getdata ():
    fle = open ( "data.data" , "r" )
    lines = int(fle.readline())
    #print ( lines )
    retx = np.empty ( (lines,2) )
    for i in range ( lines ):
        (retx[i][0],retx[i][1]) = fle.readline().split()

    fle.close ()
    return (lines,retx)

def train ( GMM ):
    i = 0
    tim = 0
    #for k in range ( 100 ):
    while True:
        GMM.train ()

        if i % 10 == 0: print ( "Epoch %d" % i )
        now = GMM.output ( 1 if i % 10 == 0 else 0 )
        i += 1
        
        cnt = 0
        if i != 1:
            for j in range ( GMM.n ):
                if now[j] != last[j]: cnt += 1
        last = now

        if cnt <= GMM.n / 150: tim += 1
        if tim >= 20: break
    
    print ( "Final prediction:" )
    now = GMM.output ( 1 )

#np.random.seed ( 53 )

gendata ()

(n,point) = getdata ()
GMM = model ( 3 , n , point )

train ( GMM )