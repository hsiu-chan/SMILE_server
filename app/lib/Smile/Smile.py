from ultralytics import YOLO
import matplotlib.pyplot as plt

import cv2
import mediapipe as mp
import os
import numpy as np
from config import OUTPUT_FOLDER, MODEL_PATH
from lib.ToothClassifier import ToothClassifier,FDI_MAP


Model = YOLO(MODEL_PATH)
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2)


mouse=[62,96,89,179,86,15,316,403,319,325,292,407,272,271,268,12,38,41,42,183]#嘴巴
lip=[78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]#嘴唇
mid_line = [10,9,8,6,5,4,1,0]# 中線

class SMILE:
    def __init__(self,input_path, 
                 device='cpu',
                 filter=0.9, 
                 output_path=f"{OUTPUT_FOLDER}output.png" ):
        self.device=device

        self.error=[]

        #####Input/output#####
        self.input_path=input_path 

        self.output_path=output_path
        """圖片輸出路徑"""



        #####Image normalize#####
        img=cv2.imread(input_path)
        h, w ,d= img.shape
        self.img=cv2.resize(img, (1024, int(1024*h/w)), interpolation=cv2.INTER_AREA)
        self.shape=self.img.shape
        """原圖shape"""



        #####Find mouse#####
        self.mouse=[]
        self.box=[]
        """boximg切割的座標 lrud"""
        self.expand=10
        self.boximg=[]
        """裁切只保留嘴巴img"""
        

        self.rotation_matrix=[]
        """旋轉矩陣"""
        self.rotated_img=''
        """轉完的矩陣"""
        


        #####Result#####
        self.filter=filter
        self.tooth=[]
        """
        [center_x,center_y,w,h]
        """
        self.tooth_cls=[]
        """牙齒類別"""

        self.ntooth=0
        """牙齒數"""

        self.smile_info={
            'test':'vault'
        }

        print('SMILE_init')


        #### set_model###


        pass


    def find_all_tooth(self):
        try:
            self.find_mouse()
            print (f'found mouth')
        except:
            return False


        ## YOLO predict
        result = Model.predict(
            source=self.output_path,
            mode="predict",
            device=self.device
        )
        
        boxes = result[0].boxes
        
        self.tooth=[]
        for box in boxes:
            if int(box.cls!=1):
                continue
            if box.conf[0]>=self.filter:
                self.tooth.append(box.xywh.tolist()[0] )

        



        h,w,d=self.boximg.shape

        self.tooth_cls=ToothClassifier(w,h,self.tooth).cls

        self.ntooth=len(self.tooth_cls)
        self.smile_info['tooth_num']=self.ntooth

        self.caculate_variables()




        
        # OUTPUT
        currentAxis = plt.gca()
        img=plt.imread(self.output_path)
        plt.imshow(img)

        colors=plt.cm.hsv(np.linspace(0, 1, self.ntooth)).tolist()   


        ## draw tooth
        for t, cl,c in zip(self.tooth, self.tooth_cls,colors):
            x,y,w,h=tuple(t)
            
            try:
                currentAxis.text(x,y, FDI_MAP[cl],bbox={'facecolor': c, 'alpha': 0.5})
            except:
                print(cl)
            currentAxis.add_patch(plt.Rectangle((x-w/2,y-h/2),w,h,fill=False,edgecolor=c,linewidth=2))
        
        ## draw mouth
        X=[x[0]-self.box[0] for x in self.mouse]
        Y=[y[1]-self.box[2] for y in self.mouse]
        print(f'{X=},{Y=}')

        plt.plot(X,Y)



        plt.axis('off')        
        plt.savefig(self.output_path,bbox_inches='tight',pad_inches=0.0, dpi=200)
        #plt.show()
        plt.clf()
        print('All tooth found and img output')
        return True


    
    
    def find_mouse(self): # 裁切嘴巴區域，存到 output_path
        h, w, d = self.img.shape
        
    
        
    
        #########################openCV辨識嘴 #########################
        RGBim = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(RGBim)

        
        ## 取得臉中線
        midline_x=[]
        midline_y=[]

        if results.multi_face_landmarks:
            for face_landmarks in  results.multi_face_landmarks:
                for index in mid_line:
                    midline_x.append(int(face_landmarks.landmark[index].x * w))
                    midline_y.append(int(face_landmarks.landmark[index].y * h))

        self.get_rotate_matrix(midline_x,midline_y)

        self.rotated_img = cv2.warpAffine(self.img, self.rotation_matrix[:2], (w, h), flags=cv2.INTER_LINEAR)
        #self.img

        
        ## 辨識

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                #for index in mouse:
                for index in lip:
                    x = int(face_landmarks.landmark[index].x * w)
                    y = int(face_landmarks.landmark[index].y * h)
                    roxy=np.dot(self.rotation_matrix,[[x],[y],[1]])
                    #self.mouse.append([x,y])
                    self.mouse.append([
                        int(roxy[0]),
                        int(roxy[1])
                    ])
        

                

        
        

        self.mouse=np.array(self.mouse)

        umos=min(self.mouse[:,1])#嘴上緣
        dmos=max(self.mouse[:,1])#嘴下緣
        lmos=min(self.mouse[:,0])#嘴左緣
        rmos=max(self.mouse[:,0])#嘴右緣
        wmos=rmos-lmos#嘴寬
        hmos=dmos-umos#嘴高
        mmos=[int((lmos+rmos)/2),int((umos+dmos)/2)]#嘴中心

        self.smile_info['mouth_width']=wmos

        
        
        ######輸出#####
        self.box=np.array([lmos-self.expand,rmos+self.expand,umos-self.expand,dmos+self.expand])
        

    

        self.boximg=self.rotated_img[self.box[2]:self.box[3],self.box[0]:self.box[1]]


        self.smile_info['mouse_box']=[10,rmos-self.box[0],10, dmos-self.box[2]] #lrud 
        





        cv2.imwrite(self.output_path,self.boximg)

        return self.box
    
    def get_rotate_matrix(self,x:list,y:list): 
        """
        x,y 為中線上的點，取回歸直線後旋轉成垂直
        """
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        m=np.abs(m)
        print(f'臉中線斜率{m=}')

            ## 回歸直線
        w,h,d=self.shape

        cy,cx= w/2, h/2

        t1=np.array([[1,0,-cx],
                    [0,1,-cy],
                    [0,0,1]])
        lenth = np.sqrt(1+m*m)

        r=np.array([[m,1,0],
            [-1,m,0],
            [0,0,lenth]])/lenth
        
        t2=np.array([[1,0,cx],
                    [0,1,cy],
                    [0,0,1]])

        self.rotation_matrix = np.dot(t2,np.dot(r, t1))
    
    def caculate_variables(self):
        tooth_map={}

        for i, c in enumerate(self.tooth_cls):
            try:
                tooth_map[c].append(i)
            except:
                tooth_map[c]=[i]
            

        ## Arc ratio
        try:
            incisor_lower_border = max([self.tooth[i][1]+ ## cy
                                        self.tooth[i][3]/2 ## h/2
                                        for i in tooth_map[1]])
            
            canine_lower=[]
            for i in [4,5]:
                try:
                    canine_lower.append(self.tooth[tooth_map[i]][1]+ ## cy
                                    self.tooth[tooth_map[i]][3]/2) ## h/2
                except:
                    pass
            if len(canine_lower)>0:
                intercanine_line = max(canine_lower)
            else:
                self.error.append('intercanine_line not found')
            
            
        except:
            self.error.append('incisor_lower_border not found')
        
        

        
