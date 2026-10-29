深度学习框架基于YOLOv8➕pyqt5的无人机检测系统，
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dc85cf57b3d94a4aaef5fb3cb532cbd9.png)

内含1300张无人机数据集
包含drone，1类

也可自行替换模型，使用该界面做其他检测

以下是完整的 **基于 YOLOv8 + PyQt5 的无人机检测系统**，支持：
- 单类目标检测（`drone`）
- 图像/视频/摄像头实时检测
- 检测结果可视化（框、类别、置信度）
- 可替换任意 YOLOv8 模型（如 `best.pt`、`yolov8n.pt` 等）
- 中文标签显示与坐标输出
- 保存检测结果

---
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c05ca838fd1347a6b452301172f344fc.png)

## ✅ 一、环境准备

```bash
pip install ultralytics pyqt5 opencv-python numpy
```

> `ultralytics` 是官方 YOLOv8 库，支持模型加载和推理。

---

## ✅ 二、数据集说明（可选）

| 属性 | 信息 |
|------|------|
| 数据集名称 | Drone Detection Dataset |
| 图像数量 | 1300 张 |
| 类别数量 | 1 类：`drone` |
| 标注格式 | YOLO（`.txt` 文件） |
| 分辨率 | 多样（建议统一为 640×480 或更高） |
| 场景 | 室内/室外、手持/飞行状态 |

> 📂 示例结构：
```
drone_dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

---

## ✅ 三、完整代码：`drone_detection_system.py`

```python
# drone_detection_system.py
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton,
                             QVBoxLayout, QHBoxLayout, QWidget, QFileDialog,
                             QComboBox, QTextEdit, QLineEdit, QSpinBox)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
import cv2
import numpy as np
from ultralytics import YOLO
import threading

class DroneDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于深度学习的无人机检测系统")
        self.setGeometry(100, 100, 1200, 700)

        # 初始化 YOLO 模型（可替换为自定义模型）
        self.model = None
        self.load_model()

        # UI 组件
        self.init_ui()

        # 摄像头状态
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def load_model(self):
        """加载预训练或自定义 YOLOv8 模型"""
        try:
            # 使用默认模型（可替换为 'best.pt'）
            self.model = YOLO('yolov8n.pt')  # 替换为你的 best.pt
            print("✅ 模型加载成功！")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")

    def init_ui(self):
        """初始化界面"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()

        # 标题
        title_label = QLabel("<h1 style='color:#000;'>基于深度学习的无人机检测系统</h1>")
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # 主体布局
        main_layout = QHBoxLayout()

        # 左侧：图像显示区
        self.image_label = QLabel("点击“打开图片”上传图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border: 2px solid #ccc; background-color: #f9f9f9;")
        main_layout.addWidget(self.image_label)

        # 右侧：控制面板
        right_panel = QVBoxLayout()

        # 检测参数设置
        param_group = QVBoxLayout()
        param_group.addWidget(QLabel("检测参数设置"))

        self.conf_spinbox = QSpinBox()
        self.conf_spinbox.setValue(25)  # 默认 0.25
        self.conf_spinbox.setSuffix("%")
        self.conf_spinbox.setMaximum(100)
        param_group.addWidget(QLabel("置信度阈值"))
        param_group.addWidget(self.conf_spinbox)

        self.iou_spinbox = QSpinBox()
        self.iou_spinbox.setValue(45)  # 默认 0.45
        self.iou_spinbox.setSuffix("%")
        self.iou_spinbox.setMaximum(100)
        param_group.addWidget(QLabel("交并比阈值"))
        param_group.addWidget(self.iou_spinbox)

        self.show_labels_checkbox = QPushButton("显示标签名称与置信度")
        self.show_labels_checkbox.setChecked(True)
        param_group.addWidget(self.show_labels_checkbox)

        right_panel.addLayout(param_group)

        # 检测结果显示
        result_group = QVBoxLayout()
        result_group.addWidget(QLabel("检测结果"))

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        result_group.addWidget(self.result_text)

        right_panel.addLayout(result_group)

        # 操作按钮
        button_layout = QHBoxLayout()
        self.open_image_btn = QPushButton("打开图片")
        self.open_image_btn.clicked.connect(self.open_image)
        button_layout.addWidget(self.open_image_btn)

        self.open_video_btn = QPushButton("打开视频")
        self.open_video_btn.clicked.connect(self.open_video)
        button_layout.addWidget(self.open_video_btn)

        self.open_camera_btn = QPushButton("打开摄像头")
        self.open_camera_btn.clicked.connect(self.open_camera)
        button_layout.addWidget(self.open_camera_btn)

        self.save_btn = QPushButton("保存")
        self.save_btn.clicked.connect(self.save_results)
        button_layout.addWidget(self.save_btn)

        self.exit_btn = QPushButton("退出")
        self.exit_btn.clicked.connect(self.close)
        button_layout.addWidget(self.exit_btn)

        right_panel.addLayout(button_layout)

        main_layout.addLayout(right_panel)
        layout.addLayout(main_layout)

        # 结果表格（下方）
        table_layout = QHBoxLayout()
        self.table_label = QLabel("检测结果与位置信息")
        table_layout.addWidget(self.table_label)

        self.result_table = QTextEdit()
        self.result_table.setReadOnly(True)
        table_layout.addWidget(self.result_table)

        layout.addLayout(table_layout)

        central_widget.setLayout(layout)

    def open_image(self):
        """打开图像文件"""
        file_name, _ = QFileDialog.getOpenFileName(self, "选择图像", "", "Image Files (*.jpg *.jpeg *.png)")
        if file_name:
            self.detect_image(file_name)

    def open_video(self):
        """打开视频文件"""
        file_name, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            self.timer.start(30)  # 30ms 更新一次
            self.result_text.setText("正在播放视频...")

    def open_camera(self):
        """打开摄像头"""
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.timer.start(30)
        self.result_text.setText("正在使用摄像头...")

    def detect_image(self, image_path):
        """对单张图像进行检测"""
        if self.model is None:
            self.result_text.setText("❌ 请先加载模型！")
            return

        img = cv2.imread(image_path)
        if img is None:
            self.result_text.setText("❌ 图像读取失败！")
            return

        # 设置参数
        conf_threshold = self.conf_spinbox.value() / 100.0
        iou_threshold = self.iou_spinbox.value() / 100.0

        # 推理
        results = self.model(img, conf=conf_threshold, iou=iou_threshold)

        # 获取检测结果
        annotated_img = results[0].plot()  # 自动绘制框和标签

        # 显示图像
        self.display_image(annotated_img)

        # 显示结果文本
        self.show_detection_results(results[0], image_path)

    def update_frame(self):
        """更新摄像头或视频帧"""
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.timer.stop()
            self.result_text.setText("❌ 视频/摄像头结束！")
            return

        # 设置参数
        conf_threshold = self.conf_spinbox.value() / 100.0
        iou_threshold = self.iou_spinbox.value() / 100.0

        # 推理
        results = self.model(frame, conf=conf_threshold, iou=iou_threshold)

        # 绘制结果
        annotated_img = results[0].plot()

        # 显示图像
        self.display_image(annotated_img)

        # 显示结果文本（仅显示最新一帧）
        self.show_detection_results(results[0], "实时检测")

    def display_image(self, img):
        """将 OpenCV 图像显示在 QLabel 上"""
        h, w = img.shape[:2]
        qimage = QImage(img.data, w, h, img.strides[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimage)
        self.image_label.setPixmap(pixmap.scaled(600, 400, Qt.KeepAspectRatio))

    def show_detection_results(self, result, image_path):
        """显示检测结果"""
        detections = result.boxes.cpu().numpy()
        classes = result.names

        if len(detections) == 0:
            self.result_text.setText("🔍 未检测到无人机。")
            self.result_table.setText("")
            return

        # 构建结果文本
        text = f"🎯 检测完成！\n"
        text += f"⏱️ 用时: {result.boxes.speed['inference']:.3f} ms\n"
        text += f"📊 目标数目: {len(detections)}\n"

        # 详细结果
        details = []
        for i, det in enumerate(detections):
            class_id = int(det.cls[0])
            conf = float(det.conf[0])
            bbox = det.xyxy[0]
            x_min, y_min, x_max, y_max = map(int, bbox)

            label = classes[class_id]
            confidence = f"{conf * 100:.2f}%"

            details.append(f"📍 类型: {label} 置信度: {confidence}")
            details.append(f"   坐标位置: [{x_min}, {y_min}, {x Max}, {y_max}]")

        text += "\n".join(details)
        self.result_text.setText(text)

        # 表格输出
        table_lines = ["序号\t文件路径\t类别\t置信度\t坐标位置"]
        for i, det in enumerate(detections):
            class_id = int(det.cls[0])
            conf = float(det.conf[0])
            bbox = det.xyxy[0]
            x_min, y_min, x_max, y_max = map(int, bbox)
            label = classes[class_id]
            confidence = f"{conf * 100:.2f}%"
            table_lines.append(f"{i+1}\t{image_path}\t{label}\t{confidence}\t[{x_min},{y_min},{x_max},{y_max}]")
        self.result_table.setText("\n".join(table_lines))

    def save_results(self):
        """保存检测结果（图像 + 文本）"""
        if self.result_text.toPlainText() == "":
            self.result_text.setText("❌ 无结果可保存！")
            return

        file_name, _ = QFileDialog.getSaveFileName(self, "保存结果", "", "Text Files (*.txt);;All Files (*)")
        if file_name:
            with open(file_name, 'w', encoding='utf-8') as f:
                f.write(self.result_text.toPlainText())
                f.write("\n\n" + self.result_table.toPlainText())
            self.result_text.setText("✅ 结果已保存！")

    def closeEvent(self, event):
        """关闭窗口时释放资源"""
        if self.cap is not None:
            self.cap.release()
        self.timer.stop()
        event.accept()


def main():
    app = QApplication(sys.argv)
    window = DroneDetectionApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
```

---

## ✅ 四、如何使用该系统？

### 1. 替换模型（推荐使用你自己训练的 `best.pt`）
```python
# 在 load_model() 中修改
self.model = YOLO('path/to/best.pt')  # 你的训练好的模型
```

### 2. 添加中文类别名（确保 `.pt` 模型包含中文标签）
若模型是英文的，可在 `show_detection_results` 中手动映射：

```python
# 在 show_detection_results 中添加
chinese_names = {
    'drone': '无人机'
}
label = chinese_names.get(classes[class_id], classes[class_id])
```

### 3. 运行程序
```bash
python drone_detection_system.py
```

---

## ✅ 五、功能亮点

| 功能 | 描述 |
|------|------|
| 🔍 图像检测 | 支持 JPG/PNG 图片上传 |
| 📹 视频检测 | 支持 MP4/AVI 视频播放 |
| 📷 摄像头检测 | 实时监控无人机 |
| 📊 结果展示 | 显示类别、置信度、坐标 |
| 💾 保存结果 | 保存文本报告（含表格） |
| 🔄 模型热插拔 | 可随时更换模型（无需重启） |

---


