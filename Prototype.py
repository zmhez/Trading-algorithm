import quandl
quandl.ApiConfig.api_key = 'bxpKukDn_rGbG_ygvhgV'


from pylab import plot,show
from numpy import vstack,array
from numpy.random import rand
import numpy as np
from scipy.cluster.vq import kmeans,vq
import datetime as dt
import pandas as pd
import pandas_datareader.data as pdd
from math import sqrt
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt

dct={'sma':['sma_ft','sma_tp'],
     'pattern':['me_star']

    }

ticker='AAPL'
start=dt.datetime(2013,1,1)
end=dt.datetime(2023,12,31)
df=pdd.DataReader(ticker, "quandl",start,end,api_key = 'u8RPsWkhvPm8DZHKzWPH')
df=df.sort_values(by='Date',ascending=True)


close='AdjClose'

op='AdjOpen'
iop=df.columns.get_loc('AdjOpen')

high='AdjHigh'
ihigh=df.columns.get_loc('AdjHigh')

low='AdjLow'
ilow=df.columns.get_loc('AdjLow')

close='AdjClose'
iclose=df.columns.get_loc('AdjClose')

vol='AdjVolume'
ivol=df.columns.get_loc('AdjVolume')

sway=0.01
ee=df.shape[1]

#Indicators
#sma
def sma(df,window,sway=0.01):
    smaa=np.zeros(df.shape[0],int)
    df['sma'] = df.AdjClose.rolling(window).mean()
    df['sma']=df['sma'].fillna(0)

    for i in range(window+1,df.shape[0]-window):
        if min(df[op][i-1],df[close][i-1])<df['sma'][i-1]*(1+sway) and max(df[op][i-1],df[close][i-1])>df['sma'][i]*(1-sway):

            if df[close][i-2]>df['sma'][i-2] and max(df[op][i-1],df[close][i-1])<df[close][i] and df[close][i]>df['sma'][i]:
                smaa[i]=1
    
            elif df[close][i-2]<df['sma'][i-2] and min(df[op][i-1],df[close][i-1])>df[close][i] and df[close][i]<df['sma'][i]:
                smaa[i]=-1

        if max(df[op][i-1],df[close][i-1])<(df['sma'][i-1]*(1+sway)) and df[close][i]>df['sma'][i] and max(df[op][i],df[close][i])>max(df[op][i-1],df[close][i-1]):
            smaa[i]=1
    
        elif min(df[op][i-1],df[close][i-1])>(df['sma'][i-1]*(1-sway)) and df[close][i]<df['sma'][i] and min(df[op][i],df[close][i])<min(df[op][i-1],df[close][i-1]):
            smaa[i]=-1

    df['sma']=smaa
    return None

#Pattern
#Morning Star/Evening Star
def me_star(df,sway=0.3):
    mes=np.zeros(df.shape[0],int)
    csize1=0
    csize2=0
    csize3=0
    for i in range(2,df.shape[0]):
        csize1=df[close][i-2]-df[op][i-2]
        csize2=df[close][i-1]-df[op][i-1]
        csize3=df[close][i]-df[op][i]

        if (csize1<0 and csize3>0 and max(df[close][i-1],df[op][i-1])<(df[close][i-2]-csize1*sway) and abs(csize2)<abs(sway*csize1) 
            and df[close][i]>(df[close][i-2]-0.5*csize1)):
            mes[i]=1
            if (df[vol][i-2]*(1-sway))<df[vol][i-1] and (df[vol][i-1]*(1-sway))<df[vol][i]:
                mes[i]=2
    
        if (csize1>0 and csize3<0 and min(df[close][i-1],df[op][i-1])>(df[close][i-2]-csize1*sway) and abs(csize2)<abs(sway*csize1) 
            and df[close][i]<(df[close][i-2]-0.5*csize1)):
            mes[i]=-2
            if (df[vol][i-2]*(1-sway))<df[vol][i-1] and (df[vol][i-1]*(1-sway))<df[vol][i]:
                mes[i]=-2

    df['mes']=mes
    return None

#DeMark 9
def dm(df,start):
    direction=-1
    if df[close][start-8]>df[close][start-12]:
        direction=1
    if direction==-1:
        for i in range(8):
            if df[close][start-i]>df[close][start-i-4]:
                return 0
        return direction   
    if direction==1:
        for i in range(8):
            if df[close][start-i]<df[close][start-i-4]:
                return 0
        return direction
        
def dm9(df):
    dm9=np.zeros(df.shape[0],int)
    for i in range(12,df.shape[0]):
        dm9[i]=dm(df,i)*(-1)

    df['dm9']=dm9
    df.drop(columns=['dm9'])    
    return None

#TST!!!!!!!!!!!

def tst(df,window=5,profit=0.05,cap=10000,sp=99,sl=0.02,pos=0.25,kelly=True,cms=0):
    count=0
    correct=0
    pt=0
    cs=cap
    pftl=[]
    for i in range(0,(df.shape[0]-window)):
        j=df.iloc[i][ee:].sum()
        if j != 0:
            count+=1
            if j>0:

                if df.iloc[i+1:i+window+1,ihigh].max()>(df[close][i]*(1+profit)):
                    pt+=1

                if df[close][i+1]>df[close][i]>0:
                    correct+=1
                    pftl.append((df[close][i+1]-df[close][i])/df[close][i])

            if j<0:
                if df.iloc[i+1:i+window+1,ilow].min()<(df[close][i]*(1-profit)):
                    pt+=1

                if df[close][i+1]<df[close][i]:
                    correct+=1
                    pftl.append((df[close][i]-df[close][i+1])/df[close][i])                
            
    if count==0:
        print('Data pool is too small')
    else:  
        ND_prob=correct/count
        DT_prob=pt/count  
        print('ND prob=',correct,'/',count,'=',ND_prob)
        print('DT prob=',pt,'/',count,'=',DT_prob)

    pft=sum(pftl)/len(pftl)
    buyin=0
    cap=cs
    ret=np.zeros(df.shape[0],int)
    #if SR=True and Kelly=True:
    if True:
        p=ND_prob
        q=1-p
        a=sl
        b=pft
        kly=(p/a-q/b)
        if kly>1:
            kly=pos
        for i in range(0,df.shape[0]-1):
            buyin=kly*cap
            j=df.iloc[i][ee:].sum()
            if kly<=0:
                print('not worth investing')
                break
            if j>0:
                cap-=buyin-buyin/(df[close][i])*(df[close][i+1])
                cap-=cms*2
            elif j<0:
                cap+=2*buyin-buyin/(df[close][i])*(df[close][i+1])  
                cap-=cms*2 
            if cap<cms:
                print('cap(SR,Kelly)=Bankrupt')
                break                
        print('cap(SR,Kelly)=',cap)

    buyin=0
    cap=cs
    ret=np.zeros(df.shape[0],int)
    tt=np.zeros(df.shape[0],int)
    #elif SR=True and Kelly!=True:
    if True:
        for i in range(0,df.shape[0]-1):
            j=df.iloc[i][ee:].sum()
            if j>0:
                cap=cap/(df[close][i])*(df[close][i+1])-cms*2
            elif j<0:
                cap=2*cap-cap/(df[close][i])*(df[close][i+1])-cms*2
            if cap<cms:
                print('cap(SR,noKelly)=Bankrupt')
                break
        print('cap(SR,noKelly)=',cap)

    buyin=0
    cap=cs
    ret=np.zeros(df.shape[0],int)
    #elif SR!=True and Kelly=True:
    if True:
        p=DT_prob
        q=1-p
        a=sl
        b=profit
        kly=(p/a-q/b)
        if kly>1:
            kly=pos

        for i in range(0,df.shape[0]-1-window):
            cap+=ret[i]
            buyin=kly*cap-cms
            j=df.iloc[i][ee:].sum()
            if kly<=0:
                print('not worth investing')
                break
            if cap>=buyin:
                if j>0:
                    cap-=buyin+cms          
                    if df.iloc[i+1:i+window+1,ihigh].max()>(df[close][i]*(1+profit)) or df.iloc[i+1:i+window+1,ilow].min()<(df[close][i]*(1-sl)):
                        for k in range(i+1,i+window+1):
                            if df[high][k]>(df[close][i]*(1+profit)):
                                ret[k]+=buyin*(1+profit)-cms*2
                                break
                            elif df[low][k]<(df[close][i]*(1-sl)):
                                ret[k]+=buyin*(1-sl)-cms*2
                                break
                    else:
                        ret[i+window]+=buyin/df[close][i]*df[close][i+window]-cms*2

                elif j<0: 
                    cap-=buyin+cms
                    if df.iloc[i+1:i+window+1,ilow].min()<(df[close][i]*(1-profit)) or df.iloc[i+1:i+window+1,ihigh].max()>(df[close][i]*(1+sl)):
                        for k in range(i+1,i+window+1):
                            if df[low][k]<(df[close][i]*(1-profit)):
                                ret[k]+=buyin*(1+profit)-cms*2
                                break
                            elif df[high][k]>(df[close][i]*(1+sl)):
                                ret[k]+=buyin*(1-sl)-cms*2
                                break
                    else:
                        ret[i+window]+=2*buyin-buyin/df[close][i]*df[close][i+window]-cms*2
            
        cap+=ret[df.shape[0]-1:].sum()                  
        print('cap(noSR,Kelly)=',cap)             
        
    buyin=0
    cap=cs
    ret=np.zeros(df.shape[0],int)
    tt=np.zeros(df.shape[0],int)
    #elif SR!=True and Kelly!=True:
    if True:
        for i in range(0,df.shape[0]-1):
            cap+=ret[i]
            j=df.iloc[i][ee:].sum()
            if cap!=0:
                if j>0: 
                    if df.iloc[i+1:i+window+1,ihigh].max()>(df[close][i]*(1+profit)) or df.iloc[i+1:i+window+1,ilow].min()<(df[close][i]*(1-sl)):
                        for k in range(i+1,i+window+1):
                            if df[high][k]>(df[close][i]*(1+profit)):
                                ret[k]+=cap*(1+profit)-cms*2
                                break
                            elif df[low][k]<(df[close][i]*(1-sl)):
                                ret[k]+=cap*(1-sl)-cms*2
                                break
                            
                    else:
                        ret[i+window]+=cap/df[close][i]*df[close][i+window]-cms*2
                    cap=0
                    
                elif j<0: 
                    if df.iloc[i+1:i+window+1,ilow].min()<(df[close][i]*(1-profit)) or df.iloc[i+1:i+window+1,ihigh].max()>(df[close][i]*(1+sl)):
                        for k in range(i+1,i+window+1):
                            if df[low][k]<(df[close][i]*(1-profit)):
                                ret[k]+=cap*(1+profit)-cms*2
                                break
                            elif df[high][k]>(df[close][i]*(1+sl)):
                                ret[k]+=cap*(1-sl)-cms*2
                                break
                    else:
                        ret[i+window]+=2*cap-cap/df[close][i]*df[close][i+window]-cms*2
                    cap=0          
        cap+=ret[df.shape[0]-1:].sum() 
        print('cap(noSR,noKelly)=',cap)     
    return None

#INPUT!!!!!!!!!!!!   
fd=pd.DataFrame()
sma(df,13)
me_star(df)
tst(df,5,0.02)
print(df.tail(7))




