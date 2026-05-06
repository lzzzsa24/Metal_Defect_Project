from ultralytics import YOLO

def main():
    print(">>> 正在装载best.pt...")
    # 路径指向你刚刚训练出来的最优权重
    model = YOLO("./runs/detect/Defect_Results/v8_train/weights/best.pt") 
    
    # 请去你的验证集里随便挑一张图片的完整名字，替换掉下面的 'xxx.jpg'
    # 例如：'crazing_1.jpg' 或者 'patches_5.jpg'
    test_image_path = "./NEU-DET/valid/images/crazing_242_jpg.rf.b0c322fa5c246914a54dd6dd77e4cd24.jpg" 
    
    print(f">>> 正在对目标图像 {test_image_path} 开启扫描...")
    
    # 执行预测：save=True 会把画好框的图片保存下来
    results = model.predict(source=test_image_path, save=True)
    
    print(">>> 扫描完毕！请前往 Defect_Results 目录查看最新生成的 predict 文件夹。")

if __name__ == "__main__":
    main()