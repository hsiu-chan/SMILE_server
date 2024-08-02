from ultralytics import YOLO
import matplotlib.pyplot as plt

import cv2
import mediapipe as mp
import os
import numpy as np
from config import OUTPUT_FOLDER, MODEL_PATH, ESPCN_PATH
from lib.ToothClassifier import ToothClassifier,FDI_MAP
from datetime import datetime


class YOLO_model:
    _instance = None
    def __new__(cls, *args, **kwargs):
        
        if not cls._instance:
            cls._instance = YOLO(MODEL_PATH)
        return cls._instance








# 使用 OpenCV 的 DNN 模型進行超解析度
sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(ESPCN_PATH)
sr.setModel("espcn", 3)  # 使用 ESPCN 模型，倍率為 3



mouse=[62,96,89,179,86,15,316,403,319,325,292,407,272,271,268,12,38,41,42,183]#嘴巴
lip=[78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191]#嘴唇
mid_line = [10,9,8,6,5,4,1,0]# 中線

class SMILE:
    def __init__(self,input_path, 
                 device='cpu',
                 filter=0.9, 
                 output_path=""):
        

        
        self.device=device

        self.current_time=datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        #####Input/output#####
        if not isinstance(input_path, str):
            raise ValueError("input_path must be an str")
        self.input_path=input_path

        file_type=input_path.split('.')[-1]
        

        self.output_path=f"{OUTPUT_FOLDER}{self.current_time}.{file_type}"
        print(f'output_path={self.output_path}')



        """圖片輸出路徑"""



        #####Image normalize#####
        img=cv2.imread(input_path)

        h, w ,d= img.shape
        self.img=img
        """原圖"""

        if (h>2000 or w>2000):
           self.img=cv2.resize(img, (1024, int(1024*h/w)), interpolation=cv2.INTER_AREA)
        if (w<500):
            self.img=sr.upsample(img)
            print("using ESPCN_x3")

        
        self.shape=self.img.shape
        """原圖shape: h, w ,d"""



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
            "image":{
                "width":self.shape[1],
                "height":self.shape[0]
            },
            "date":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "error":[]
            

        }
        """輸出"""

        print('SMILE_init')


        #### set_model###


        pass
    
    def add_error(self,e:str)->None:
        self.smile_info["error"].append(e)

    
    def is_face(self)->bool:
        """
        進行人臉檢測
        """
        
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        results = face_detection.process(self.img)
        
        return bool(results.detections)
    



    def find_all_tooth(self):
        """
        找所有牙齒
        """
        
        if not self.is_face():
            self.add_error('face not found')
            return False


        
        self.find_mouse()

        ## YOLO predict
        result = YOLO_model().predict(
            source=self.output_path,
            mode="predict",
            device=self.device
        )
        
        boxes = result[0].boxes
        
        self.tooth=[]
        self.smile_info['tooth_boxes']=[]
        """xywh, 0-1"""

        x0=self.box[0]
        y0=self.box[2]

        for box in boxes:
            if int(box.cls!=0):
                continue
            if box.conf[0]>=self.filter:
                tooth_box=box.xywh.tolist()[0]
                self.tooth.append(tooth_box)

                float_box=[
                        (x0+tooth_box[0])/self.shape[1],
                        (y0+tooth_box[1])/self.shape[0],
                        tooth_box[2]/self.shape[1],
                        tooth_box[3]/self.shape[0]
                    ]
                self.smile_info['tooth_boxes'].append(list(map(lambda num: round(num, 5), float_box)))
        
        # 無牙齒情況
        if (len(self.tooth))==0:
            self.add_error("tooth not found")
            return False
        



        



        h,w,d=self.boximg.shape
        # 牙齒類別
        self.tooth_cls=ToothClassifier(w,h,self.tooth).cls
        self.smile_info['tooth_cls']=[FDI_MAP[c] for c in self.tooth_cls]
        


        

        # 牙齒數量
        self.ntooth=len(self.tooth_cls)
        self.smile_info['tooth_num']=self.ntooth

        self.caculate_variables()




        
        
        print('All tooth found and img output')
        return True


    
    
    def find_mouse(self): # 裁切嘴巴區域，存到 output_path
        h, w, d = self.img.shape
        
    
        
    
        #########################openCV辨識嘴 #########################
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh=mp_face_mesh.FaceMesh(
                    max_num_faces=1,       # 一次偵測最多幾個人臉
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)



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

        self.smile_info['mouth'] = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                #for index in mouse:
                for index in lip:
                    x = int(face_landmarks.landmark[index].x * w)
                    y = int(face_landmarks.landmark[index].y * h)
                    roxy=np.dot(self.rotation_matrix,[[x],[y],[1]])
                    self.mouse.append([x,y])
                    
                    self.smile_info['mouth'].append([round(face_landmarks.landmark[index].x,5),round(face_landmarks.landmark[index].y,5)])

                    #self.mouse.append([
                    #    int(roxy[0]),
                    #    int(roxy[1])
                    #])
        

                

        
        re_mouth=np.array(self.smile_info['mouth'])

        self.smile_info['mouth_box']=[
            min(re_mouth[:,0]),
            max(re_mouth[:,0]),
            min(re_mouth[:,1]),
            max(re_mouth[:,1]),
        ]


        self.mouse=np.array(self.mouse)

        

        umos=min(self.mouse[:,1])#嘴上緣
        dmos=max(self.mouse[:,1])#嘴下緣
        lmos=min(self.mouse[:,0])#嘴左緣
        rmos=max(self.mouse[:,0])#嘴右緣
        wmos=rmos-lmos#嘴寬
        hmos=dmos-umos#嘴高
        mmos=[int((lmos+rmos)/2),int((umos+dmos)/2)]#嘴中心


        
        
        ######輸出#####
        self.box=np.array([lmos-self.expand,rmos+self.expand,umos-self.expand,dmos+self.expand])
        

    

        self.boximg=self.img[self.box[2]:self.box[3],self.box[0]:self.box[1]]


        

        
        
        cv2.imwrite(self.output_path,self.boximg)
        print('boximg ok')

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

        r=np.array([[m,-1,0],
            [1,m,0],
            [0,0,lenth]])/lenth
        
        t2=np.array([[1,0,cx],
                    [0,1,cy],
                    [0,0,1]])

        self.rotation_matrix = np.dot(t2,np.dot(r, t1))
    
    def caculate_variables(self):
        """
        計算牙齒參數
        """
        ## 初始化 cls2tooth 作為字典
        
        cls2tooth = {} # 牙齒類別->tooth 編號

        for i, cls in enumerate(self.tooth_cls):
            if cls != 0:
                if cls not in cls2tooth:
                    cls2tooth[cls] = []
                cls2tooth[cls].append(i)

        self.smile_info['cls2tooth'] = cls2tooth
        
        ## maxillary teeth exposed
        maxillary_teeth_exposed=0
        for cls in self.tooth_cls:
            if cls!=0:
                maxillary_teeth_exposed+=1
        
        self.smile_info['maxillary_teeth_exposed']=maxillary_teeth_exposed


        teeth= self.smile_info["tooth_boxes"]

        ## incisor_lower_border
        try:
            incisor_lower_border = max([ teeth[i][1]+ ## cy
                                        teeth[i][3]/2 ## h/2
                                        for i in cls2tooth[1]])
            self.smile_info["incisor_lower_border"]=incisor_lower_border
        except:
            self.add_error("incisor not found")


        ## intercanine_line
        canine_id=list(cls2tooth[4])+list(cls2tooth[5])
        try:
            intercanine_line = max([
                teeth[i][1]+ ## cy
                teeth[i][3]/2 ## h/2
                for i in canine_id
            ])
            self.smile_info["intercanine_line"]=intercanine_line
        except:
            self.add_error('canine not found')
        
        
        ## Arc ratio
        #try:
        mouse_lower_border = np.max(np.array(self.smile_info["mouth"])[:,1])

        print(f'{incisor_lower_border-intercanine_line=}')
        print(f'{mouse_lower_border-intercanine_line=}')

        arc_ratio=(incisor_lower_border-intercanine_line)/(mouse_lower_border-intercanine_line)

        self.smile_info['arc_ratio']=round(arc_ratio,3)
        
        

        #except Exception as e:
        #     print(f"Arc ratio error: {e}")
        


        ## buccal corridor

        all_teeth_width = max([ t[0]+t[2]/2 for t in teeth])- min([ t[0]-t[2]/2 for t in teeth])

        mouth_box=self.smile_info['mouth_box']

        buccal_corridor=1-all_teeth_width/(mouth_box[1]-mouth_box[0])
        
        self.smile_info['buccal_corridor']=round(buccal_corridor,3)

    def draw_mouth(self):
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
        
        
        



        plt.axis('off')        
        plt.savefig(self.output_path,bbox_inches='tight',pad_inches=0.0, dpi=200)
        #plt.show()
        plt.clf()
    def draw_result(self):
        """
        畫完整人臉，輸出到 output 
        """


        # 將相對坐標轉換為絕對坐標
        def relative_to_absolute(coords, width, height):
            return [(x * width, y * height) for x, y in coords]
        # 顏色映射字典
        color_map = {
            'L': 'cyan',
            '1': 'blue',
            '12': 'green',
            '22': 'magenta',
            '13': 'yellow',
            '23': 'red',
            'pm_R': 'orange',
            'pm_L': 'purple'
        }

        background_image = plt.imread(self.input_path)


        smile_info=self.smile_info
        image_height = background_image.shape[0]
        image_width = background_image.shape[1]
        
        


        # 畫背景
        fig, ax = plt.subplots(figsize=(image_width / 100, image_height / 100), dpi=100)

        # 顯示背景圖片
        plt.imshow(background_image, extent=[0, image_width,  image_height, 0], origin='upper')

        # 繪製口腔輪廓
        
        try:
            mouth_points = smile_info['mouth']
            mouth_points_abs = relative_to_absolute(mouth_points, image_width, image_height)
            mouth_points_abs.append(mouth_points_abs[0])
            mouth_x, mouth_y = zip(*mouth_points_abs)
            ax.plot(mouth_x, mouth_y, marker='', linestyle='-', color='gray', alpha=0.4)
        except:
            pass

        # 繪製牙齒並根據牙齒類別上色

        try:

            tooth_boxes = smile_info['tooth_boxes']
            tooth_cls = smile_info['tooth_cls']
            tooth_boxes_abs = relative_to_absolute([box[:2] for box in tooth_boxes], image_width, image_height)
            tooth_sizes = [box[2:4] for box in tooth_boxes]
            handles = []
            labels = []

            for (x, y), (w, h), cls in zip(tooth_boxes_abs, tooth_sizes, tooth_cls):
                color = color_map.get(cls, 'gray')  # 默認顏色為灰色
                rect = plt.Rectangle((x - w * image_width / 2, y - h * image_height / 2), w * image_width, h * image_height, 
                                    linewidth=1, edgecolor=color, facecolor='none', alpha=0.4)
                ax.add_patch(rect)
                if cls not in labels:
                    handles.append(rect)
                    labels.append(cls)

            # 添加圖例
            ax.legend(handles, labels, title='Tooth Classes')
        except:
            pass

        # 設置圖形的範圍和標籤
        ax.set_xlim(0, image_width)
        ax.set_ylim(image_height, 0)

        ax.set_title('Mouth and Teeth Visualization')
        ax.set_xlabel('Width')
        ax.set_ylabel('Height')

        plt.axis('off')        
        plt.savefig(self.output_path,bbox_inches='tight',pad_inches=0.0, dpi=200)
        plt.clf()








        
