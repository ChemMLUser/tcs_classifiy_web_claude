import cv2
import numpy as np
import os
from datetime import datetime


def filter_valid_holes(detections):
    """
    过滤掉不符合要求的孔位（长宽比大于1.5的）

    Args:
        detections: 原始检测结果列表

    Returns:
        过滤后的检测结果列表
    """
    valid_detections = []

    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        width = x2 - x1
        height = y2 - y1

        # 计算长宽比
        aspect_ratio = max(width, height) / min(width, height)

        # 只保留长宽比小于等于1.5的孔位
        if aspect_ratio <= 1.5:
            valid_detections.append(detection)
        else:
            print(f"过滤掉长宽比为{aspect_ratio:.2f}的孔位: bbox={detection['bbox']}")

    return valid_detections


def extract_rgb_values(image_path, bbox):
    """
    从指定边界框区域提取平均RGB值

    Args:
        image_path: 图片路径
        bbox: 边界框 [x1, y1, x2, y2]

    Returns:
        RGB值字典 {'r': r_value, 'g': g_value, 'b': b_value}
    """
    try:
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # BGR转RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 提取ROI区域
        x1, y1, x2, y2 = bbox
        roi = image[y1:y2, x1:x2]

        if roi.size == 0:
            return {'r': 0, 'g': 0, 'b': 0}

        # 计算平均RGB值
        mean_rgb = np.mean(roi, axis=(0, 1))

        return {
            'r': int(mean_rgb[0]),
            'g': int(mean_rgb[1]),
            'b': int(mean_rgb[2])
        }

    except Exception as e:
        print(f"提取RGB值时出错: {e}")
        return {'r': 0, 'g': 0, 'b': 0}


def process_detection_results(image_path, detections, output_dir):
    """
    处理检测结果，生成带标注的图片和孔位数据

    Args:
        image_path: 原图片路径
        detections: 检测结果列表
        output_dir: 输出目录

    Returns:
        (processed_image_path, hole_data)
    """
    try:
        # 过滤掉不符合要求的孔位
        detections = filter_valid_holes(detections)

        if len(detections) == 0:
            raise ValueError("没有检测到符合要求的孔位")

        print(f"过滤后剩余{len(detections)}个有效孔位")

        # 读取原图
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 生成输出文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"processed_{timestamp}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 准备孔位数据
        hole_data = []

        # 在图片上绘制检测框和编号
        for i, detection in enumerate(detections):
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']

            # 提取RGB值
            rgb_values = extract_rgb_values(image_path, detection['bbox'])

            # 绘制检测框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # 绘制孔位编号
            hole_id = i + 1
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            # 绘制圆形背景
            cv2.circle(image, (center_x, center_y), 20, (255, 255, 255), -1)
            cv2.circle(image, (center_x, center_y), 20, (0, 0, 0), 2)

            # 绘制编号
            text = str(hole_id)
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = center_x - text_size[0] // 2
            text_y = center_y + text_size[1] // 2
            cv2.putText(image, text, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

            # 添加置信度标签
            conf_text = f"{conf:.2f}"
            cv2.putText(image, conf_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 保存孔位数据
            hole_info = {
                'id': hole_id,
                'bbox': detection['bbox'],
                'center_x': center_x,
                'center_y': center_y,
                'confidence': conf,
                'r': rgb_values['r'],
                'g': rgb_values['g'],
                'b': rgb_values['b'],
                'grid_id': detection.get('grid_id', f'H{hole_id}')
            }
            hole_data.append(hole_info)

        # 保存处理后的图片
        cv2.imwrite(output_path, image)

        # 返回相对路径
        relative_path = os.path.join(output_dir, output_filename).replace('\\', '/')

        return relative_path, hole_data

    except Exception as e:
        print(f"处理检测结果时出错: {e}")
        raise e


def enhance_image_contrast(image_path, output_path=None):
    """
    增强图像对比度，提高检测效果

    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径（可选）

    Returns:
        增强后的图片路径
    """
    try:
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")

        # 转换到LAB色彩空间
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # 分离L通道
        l, a, b = cv2.split(lab)

        # 应用CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

        # 合并通道
        enhanced_lab = cv2.merge([l, a, b])

        # 转换回BGR
        enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # 保存增强后的图片
        if output_path is None:
            name, ext = os.path.splitext(image_path)
            output_path = f"{name}_enhanced{ext}"

        cv2.imwrite(output_path, enhanced_image)

        return output_path

    except Exception as e:
        print(f"增强图片对比度时出错: {e}")
        return image_path  # 返回原图路径


def create_rgb_visualization(hole_data, output_path):
    """
    创建RGB值的可视化图表

    Args:
        hole_data: 孔位数据列表
        output_path: 输出图片路径
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # 左侧：RGB条形图
        hole_ids = [hole['id'] for hole in hole_data]
        r_values = [hole['r'] for hole in hole_data]
        g_values = [hole['g'] for hole in hole_data]
        b_values = [hole['b'] for hole in hole_data]

        x = np.arange(len(hole_ids))
        width = 0.25

        ax1.bar(x - width, r_values, width, label='Red', color='red', alpha=0.7)
        ax1.bar(x, g_values, width, label='Green', color='green', alpha=0.7)
        ax1.bar(x + width, b_values, width, label='Blue', color='blue', alpha=0.7)

        ax1.set_xlabel('孔位编号')
        ax1.set_ylabel('RGB值')
        ax1.set_title('各孔位RGB值分布')
        ax1.set_xticks(x)
        ax1.set_xticklabels(hole_ids)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右侧：颜色块显示
        grid_size = int(np.ceil(np.sqrt(len(hole_data))))
        for i, hole in enumerate(hole_data):
            row = i // grid_size
            col = i % grid_size

            # 创建颜色块
            color = (hole['r'] / 255, hole['g'] / 255, hole['b'] / 255)
            rect = patches.Rectangle((col, grid_size - row - 1), 1, 1,
                                     linewidth=1, edgecolor='black', facecolor=color)
            ax2.add_patch(rect)

            # 添加孔位编号
            ax2.text(col + 0.5, grid_size - row - 0.5, str(hole['id']),
                     ha='center', va='center', fontsize=10, fontweight='bold')

        ax2.set_xlim(0, grid_size)
        ax2.set_ylim(0, grid_size)
        ax2.set_aspect('equal')
        ax2.set_title('孔位颜色可视化')
        ax2.set_xticks([])
        ax2.set_yticks([])

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"RGB可视化图表已保存到: {output_path}")

    except ImportError:
        print("matplotlib未安装，跳过RGB可视化")
    except Exception as e:
        print(f"创建RGB可视化时出错: {e}")