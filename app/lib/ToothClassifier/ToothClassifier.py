import numpy as np
import joblib
from config import RANDOM_FOREST_MODEL

loaded_model = joblib.load(RANDOM_FOREST_MODEL)
NUM_LABELS = 7


class ToothClassifier:
    def __init__(self,w,h,tooth):
        self.features=[[
            t[0]/w,
            t[1]/h,
            t[2]/w,
            t[3]/h,
            w/h
        ] for t in tooth]
        self.predicts=loaded_model.predict_proba(self.features)
        self._now=self.predicts.copy()
        self._result=[[] for i in range(NUM_LABELS)]

        for i in range(len(tooth)):
            self.add(i)

        self.cls=[-1 for i in range(len(tooth))]

        for i in range(len(self._result)):
            for t in self._result[i]:
                self.cls[t]=i

    
    def add(self,ti):


        cls=np.argmax(self._now[ti])

        if cls>6:
            print('class error')
            return
        
        self._result[cls].append(ti)
        if cls in (0,6) :
            return
        elif cls == 1 and len(self._result[1])<3:
            return
        elif cls in (2,3,4,5) and len(self._result[cls])<2:
            return
        
        
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




        #print(f'gg,{cls=},{pmin=},{result[cls]=}')
        #END+=1

        self.add(pmin)
