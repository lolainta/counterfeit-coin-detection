from sklearn.linear_model import Lasso,LassoCV
import matplotlib.pyplot as plt
from threading import Thread
from queue import Queue
from tqdm import tqdm
import numpy as np
import warnings
import config

def regression(phi,res,cntft):
    alphas=LassoCV(cv=5,n_jobs=-1).fit(phi,res).alphas_
    for alpha in np.flip(alphas):
        ret=Lasso(alpha=alpha,max_iter=1000000).fit(phi,res).coef_
        ret[np.abs(ret)<0.2]=0
        if len(np.nonzero(ret)[0])>=cntft:
            break
    return ret

def sim(cntft,eq):
    phi=np.random.rand(eq,config.NUM_COINS)
    secrete=np.zeros(config.NUM_COINS)
    for i in np.random.choice(range(config.NUM_COINS),cntft,replace=False):
        secrete[i]=1
    res=phi@secrete
    guess=regression(phi,res,cntft)
    return np.array_equal(np.nonzero(secrete)[0],np.nonzero(guess)[0])

def run(cntft,pos):
    for eq in tqdm(config.N,position=pos):
        cnt=0
        for exp in tqdm(range(config.EXP_NUM),leave=False,position=pos+1):
            cnt+=sim(cntft,eq)
        queue.put((cntft,eq,cnt/config.EXP_NUM))

def plot(res):
    fig,ax=plt.subplots()
    ax.set_xlabel("Number of weighings", fontsize=12)
    ax.set_ylabel("Success rate", fontsize=12)
    ax.set_xlim([5,40])
    ax.set_ylim([-0.01,1.01])
    fig.set_figwidth(12) 
    fig.set_figheight(5)

    y=np.transpose(np.array([list(result.values()) for result in res.values()]))
    ax.plot(config.N,y,'o',linestyle=':')
    plt.savefig('result.png')

def main():
    threads=list()
    ans=dict()
    for i,cntft in enumerate(config.COUNTERFEIT):
        ans[cntft]=dict()
        threads.append(Thread(target=run,args=(cntft,i*2,)))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    while not queue.empty():
        res=queue.get()
        ans[res[0]][res[1]]=res[2]
    print(ans)
    plot(ans)


queue=Queue()
warnings.filterwarnings(action='ignore')
if __name__ == '__main__':
    main()
