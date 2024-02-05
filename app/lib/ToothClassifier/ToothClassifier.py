import numpy as np
import joblib
from config import RANDOM_FOREST_MODEL

loaded_model = joblib.load(RANDOM_FOREST_MODEL)


class ToothClassifier:
    def __init__(self,w,h,tooth):
        self.features=[[
            t[0]/w,
            t[1]/h,
            t[2]/w,
            t[3]/h
        ] for t in tooth]
        self.predicts=loaded_model.predict_proba(self.features)
        self._now=self.predicts.copy()
        self._result=[[] for i in range(6)]

        for i in range(len(tooth)):
            self.add(i)

        self.cls=[-1 for i in range(len(tooth))]

        for i in range(len(self._result)):
            for t in self._result[i]:
                self.cls[t]=i

    
    def add(self,ti):


        cls=np.argmax(self._now[ti])

        if cls>5:
            print('class error')
            return
        self._result[cls].append(ti)
        if cls in (0,5) or len(self._result[cls])<3:
            #print(f'{tooth=},{cls=}')
            return
        
        member=np.array(self._result[cls])
        argpmin=np.argmin([self._now[t][cls] for t in member])
        pmin=member[argpmin]
        self._result[cls][argpmin]=self._result[cls][-1]
        self._result[cls].pop()
        self._now[pmin][cls]=0

        #print(f'gg,{cls=},{pmin=},{result[cls]=}')
        #END+=1

        self.add(pmin)
