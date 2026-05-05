# 声明：本代码由大模型辅助生成
from ultralytics import YOLO
import torch

def main():
    
       print(">>> 正在加载 YOLOv8 架构...")
       model = YOLO("yolov8n.pt") 

       print(">>> 启动拟合...")
       results = model.train(
           data="./NEU-DET/data.yaml",
           epochs=100,                  # 训练轮数
           imgsz=800,                  # 图像缩放尺寸
           batch=16,                   # 每次吞吐的图片数量 (4060 8G 推荐值)
           device=0,                   # 调用第一张 GPU
           workers=2,                  # 限制数据加载线程，防止内存爆炸
           optimizer='auto',           # 自动选择优化器
       
           copy_paste=0.2,              
           mosaic=0.5,                  
           erasing=0.4,                 

           project="Defect_Results",   # 训练结果保存的主文件夹名
           name="v5_train"             # 本次训练的子文件夹名
       )

       print(">>> 训练结束。")

if __name__ == '__main__':
       main()
       