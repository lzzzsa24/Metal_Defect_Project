# 声明：本代码由大模型辅助生成，已根据本地 RTX 4060 环境进行超参数调优。
from ultralytics import YOLO
import torch

def main():
       # 验证 CUDA 是否可用 (测试你的显卡是否上线)
       print(f"GPU 是否可用: {torch.cuda.is_available()}")
       if torch.cuda.is_available():
           print(f"当前使用的显卡: {torch.cuda.get_device_name(0)}")

       print(">>> 正在加载 YOLOv8 架构...")
       model = YOLO("yolov8n.pt") 

       print(">>> 启动拟合...")
       results = model.train(
           data="./NEU-DET/data.yaml", # 指向你刚才修改的配置文件
           epochs=50,                  # 训练轮数
           imgsz=640,                  # 图像缩放尺寸
           batch=16,                   # 每次吞吐的图片数量 (4060 8G 推荐值)
           device=0,                   # 调用第一张 GPU
           workers=2,                  # 限制数据加载线程，防止内存爆炸
           project="Defect_Results",   # 训练结果保存的主文件夹名
           name="v1_train"             # 本次训练的子文件夹名
       )

       print(">>> 训练结束。")

if __name__ == '__main__':
       main()
       