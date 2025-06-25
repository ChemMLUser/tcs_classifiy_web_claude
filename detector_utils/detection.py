import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import os


class YOLOv5Detector:
    def __init__(self, model_path):
        """
        初始化YOLOv5检测器

        Args:
            model_path: 训练好的YOLOv5模型路径
        """
        self.model_path = model_path
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()

    def load_model(self):
        """加载YOLOv5模型"""
        try:
            # 加载自定义训练的模型
            self.model = torch.hub.load('yolov5_master', 'custom',
                                        path=self.model_path, source='local', force_reload=True)
            self.model.to(self.device)
            print(f"YOLOv5模型加载成功，使用设备: {self.device}")
        except Exception as e:
            print(f"加载YOLOv5模型失败: {e}")
            raise e

    def detect(self, image_path, conf_threshold=0.25, iou_threshold=0.45):
        """
        对图片进行检测

        Args:
            image_path: 图片路径
            conf_threshold: 置信度阈值
            iou_threshold: IOU阈值

        Returns:
            检测结果列表，每个结果包含 [x1, y1, x2, y2, confidence, class]
        """
        try:
            # 设置模型参数
            self.model.conf = conf_threshold
            self.model.iou = iou_threshold

            # 进行检测
            results = self.model(image_path)

            # 解析检测结果
            detections = []
            if len(results.xyxy[0]) > 0:
                for detection in results.xyxy[0].cpu().numpy():
                    x1, y1, x2, y2, conf, cls = detection
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': int(cls),
                        'center_x': int((x1 + x2) / 2),
                        'center_y': int((y1 + y2) / 2)
                    })

            # 按置信度排序
            detections.sort(key=lambda x: x['confidence'], reverse=True)

            return detections

        except Exception as e:
            print(f"检测图片时出错: {e}")
            raise e

    def visualize_detections(self, image_path, detections, output_path):
        """
        可视化检测结果

        Args:
            image_path: 原图片路径
            detections: 检测结果
            output_path: 输出图片路径
        """
        try:
            # 读取图片
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")

            # 绘制检测框
            for i, detection in enumerate(detections):
                x1, y1, x2, y2 = detection['bbox']
                conf = detection['confidence']

                # 绘制矩形框
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 添加标签
                label = f"Hole_{i + 1}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(image, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(image, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # 保存图片
            cv2.imwrite(output_path, image)
            print(f"检测结果已保存到: {output_path}")

        except Exception as e:
            print(f"可视化检测结果时出错: {e}")
            raise e

    def filter_detections_by_grid(self, detections, grid_rows=8, grid_cols=12):
        """
        根据96孔板网格布局过滤和排序检测结果

        Args:
            detections: 原始检测结果
            grid_rows: 网格行数 (默认8)
            grid_cols: 网格列数 (默认12)

        Returns:
            按网格位置排序的检测结果
        """
        if not detections:
            return detections

        # 计算网格参数
        centers = [(det['center_x'], det['center_y']) for det in detections]

        if len(centers) < 2:
            return detections

        # 估算网格间距
        x_coords = sorted([c[0] for c in centers])
        y_coords = sorted([c[1] for c in centers])

        # 计算平均间距
        x_spacing = np.median(np.diff(x_coords)) if len(x_coords) > 1 else 50
        y_spacing = np.median(np.diff(y_coords)) if len(y_coords) > 1 else 50

        # 为每个检测分配网格位置
        for detection in detections:
            x, y = detection['center_x'], detection['center_y']

            # 计算网格位置
            grid_x = round((x - min(x_coords)) / x_spacing) if x_spacing > 0 else 0
            grid_y = round((y - min(y_coords)) / y_spacing) if y_spacing > 0 else 0

            detection['grid_row'] = min(grid_y, grid_rows - 1)
            detection['grid_col'] = min(grid_x, grid_cols - 1)
            detection['grid_id'] = f"{chr(65 + detection['grid_row'])}{detection['grid_col'] + 1}"

        # 按网格位置排序
        detections.sort(key=lambda x: (x['grid_row'], x['grid_col']))

        return detections