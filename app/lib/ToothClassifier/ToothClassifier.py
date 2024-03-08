import numpy as np
import joblib
from config import RANDOM_FOREST_MODEL

loaded_model = joblib.load(RANDOM_FOREST_MODEL)
NUM_LABELS = 8


class ToothClassifier:
    """
    Tooth shoould be [xywh_box]
    """
    """
    Tooth shoould be [xywh_box]
    """
    def __init__(self,w,h,tooth):
        self.len=len(tooth)
        """ Number of tooth"""
        self.cls=[-1 for i in range(self.len)]
        """ Classes of tooth"""


        ## 生成 features
        tx_postive=[t[0]-w/2 for t in tooth if t[0]>w/2]
        tx_negative=[w/2-t[0] for t in tooth if t[0]<=w/2]

        sorted_tx_postive = sorted(tx_postive)
        sorted_tx_negative = sorted(tx_negative)
        
        ranks_tx_postive = {val:i for i,val in enumerate(sorted_tx_postive)} ## 原始數值 -> 排名

        ranks_tx_negative = {val:i for i,val in enumerate(sorted_tx_negative)} ## 原始數值 -> 排名


        tw=[t[2] for t in tooth]
        sorted_tw=sorted(tw) # 大到小
        ranks_tw = {val: i/(self.len-1) for i, val in enumerate(sorted_tw)}# i:排名, val:原始值

        th=[t[3] for t in tooth]
        sorted_th = sorted(th)
        ranks_th = {val:i/(self.len-1) for i,val in enumerate(sorted_th)}




        self.features=[[ # from xywh
            t[0]/w, ## x
            t[1]/h, ## y
            t[2]/w, ## w
            t[3]/h, ## h
            w/h, ## 長寬比

            ranks_tx_postive[t[0]-w/2] if t[0]-w/2 in ranks_tx_postive else -1,

            ranks_tx_negative[w/2-t[0]] if w/2-t[0] in ranks_tx_negative else -1,

            ranks_tw[t[2]],# Rank w 牙齒寬度排名
            ranks_th[t[3]] # Rank h 牙齒高度排名

        ] for i,t in enumerate(tooth)]
        


        self.predicts=loaded_model.predict_proba(self.features)

        self.analysis()



    
    def analysis(self):
        
        self._now=self.predicts.copy()
        """
        每顆牙的分類機率
        """
        self._result=[[] for i in range(NUM_LABELS)]

        for i in range(self.len):
            self.add(i)

        self.cls=[-1 for i in range(self.len)]

        for i in range(len(self._result)):
            for t in self._result[i]:
                self.cls[t]=i

        

    
    def add(self,ti):


        cls=np.argmax(self._now[ti]) ## 機率最大的類別


        if cls>7:
            print('Not a tooth')
            return
        
        if cls==0:  ## lower 先去upper
            if self._now[ti][cls]>0.6:
                self._result[cls].append(ti)
                return
            cls=np.argmax(self._now[ti][1:]) ## 機率最大的類別


        if cls in (6,7) and self._now[ti][cls-2]!=0: ## 可能是犬齒
            cls-=2
        
        
        self._result[cls].append(ti)

        cls_num = len(self._result[cls])




        
        match cls:
            case 1:
                if cls_num<3: ## 上顎門牙未滿
                    return
            case 2|3|4|5:
                if cls_num<2: ## 上顎側門牙犬齒未滿
                    return
            case _:
                return 

            
            


        
        
        
        ## 滿人找出最不可能的剔掉
        member=np.array(self._result[cls])
        argpmin=np.argmin([self._now[t][cls] for t in member])
        pmin=member[argpmin]
        self._result[cls][argpmin]=self._result[cls][-1]
        self._result[cls].pop()

        adder=self._now[pmin][cls]
        self._now[pmin][cls]=0
        adder=adder/(sum([int(i>0) for i in self._now[pmin]])-1)
        for i in range(len(self._now[pmin])-1):
            if self._now[pmin][i+1]>0:
                self._now[pmin][i+1]+=adder
        self._now[pmin][0]*=1.1
        #print(f'gg,{cls=},{pmin=},{result[cls]=}')
        #END+=1

        self.add(pmin)

    

    
