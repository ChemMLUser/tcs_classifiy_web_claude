from flask import Flask, request, render_template, jsonify, send_file, redirect, url_for
import os
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import json
import joblib
from werkzeug.utils import secure_filename
import sys
import torch
from detector_utils.detection import YOLOv5Detector
from detector_utils.image_processing import extract_rgb_values, process_detection_results
from detector_utils.classification import ClassificationModel

# 添加yolov5路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5_master'))


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 确保必要的文件夹存在
for folder in ['uploads', 'outputs', 'static/processed_images']:
    os.makedirs(folder, exist_ok=True)

# 初始化模型
detector = YOLOv5Detector('yolov5_master/weights/best.pt')
classifier = ClassificationModel('model/best_classification_model.pkl')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有选择文件'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # 生成唯一文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            # 进行YOLO检测
            detection_results = detector.detect(filepath)

            # 处理检测结果
            processed_image_path, hole_data = process_detection_results(
                filepath, detection_results, 'static/processed_images'
            )

            return jsonify({
                'success': True,
                'processed_image': processed_image_path,
                'hole_data': hole_data,
                'original_filename': file.filename
            })

        except Exception as e:
            return jsonify({'error': f'处理图片时出错: {str(e)}'}), 500

    return jsonify({'error': '不支持的文件格式'}), 400


@app.route('/analyze', methods=['POST'])
def analyze_samples():
    try:
        data = request.get_json()

        # 从前端数据结构中正确提取信息
        samples_data_from_frontend = data.get('samples', [])
        sample_count = len(samples_data_from_frontend)
        original_filename = data.get('original_filename', 'unknown')

        # 添加调试信息
        print(f"调试信息:")
        print(f"  收到的样本数量: {sample_count}")
        print(f"  samples数据: {samples_data_from_frontend}")

        if sample_count == 0:
            return jsonify({'error': '没有收到样本数据'}), 400

        # 验证每个样本是否有4个孔位
        for i, sample in enumerate(samples_data_from_frontend):
            holes = sample.get('holes', [])
            if len(holes) != 4:
                return jsonify({'error': f'样品{i + 1}需要选择4个孔位，但实际选择了{len(holes)}个孔位'}), 400

        # 组织数据为分类模型需要的格式 (每个样品4个孔，每个孔3个RGB值 = 12维特征)
        samples_data = []
        for i, sample in enumerate(samples_data_from_frontend):
            sample_features = []
            holes = sample.get('holes', [])

            print(f"  样品{i + 1}的孔位数据:")
            for j, hole_data in enumerate(holes):
                # 检查孔位数据是否包含RGB值
                if not all(key in hole_data for key in ['r', 'g', 'b']):
                    print(f"    孔位{j + 1}数据缺少RGB值: {hole_data}")
                    return jsonify({'error': f'样品{i + 1}孔位{j + 1}数据格式错误，缺少RGB值'}), 400

                r, g, b = hole_data['r'], hole_data['g'], hole_data['b']
                sample_features.extend([r, g, b])
                print(f"    孔位{j + 1}: R={r}, G={g}, B={b}")

            samples_data.append(sample_features)
            print(f"  样品{i + 1}特征向量: {sample_features}")

        print(f"最终用于分类的数据: {samples_data}")

        # 进行分类预测
        predictions = classifier.predict(samples_data)
        probabilities = classifier.predict_proba(samples_data)

        # 准备结果数据
        results = []
        for i, (prediction, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                'sample_id': f'样品{i + 1}',
                'prediction': prediction,
                'confidence': float(prob.max()),
                'features': samples_data[i],
                'hole_details': samples_data_from_frontend[i]['holes']  # 保留原始孔位详细信息
            }
            results.append(result)
            print(f"样品{i + 1}预测结果: {prediction}, 置信度: {float(prob.max()):.4f}")

        # 保存结果到CSV
        csv_filename = save_results_to_csv(results, original_filename)

        return jsonify({
            'success': True,
            'results': results,
            'csv_filename': csv_filename
        })

    except Exception as e:
        print(f"分析时出现异常: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'分析时出错: {str(e)}'}), 500


def save_results_to_csv(results, original_filename):
    """保存结果到CSV文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"results_{timestamp}_{original_filename.split('.')[0]}.csv"
    csv_path = os.path.join(app.config['OUTPUT_FOLDER'], csv_filename)

    # 准备CSV数据
    csv_data = []
    for result in results:
        row = {
            '样品ID': result['sample_id'],
            '检测结果': result['prediction'],
            '置信度': result['confidence'],
            '检测时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 添加12维特征数据
        for i, feature in enumerate(result['features']):
            hole_num = (i // 3) + 1
            rgb_component = ['R', 'G', 'B'][i % 3]
            row[f'孔{hole_num}_{rgb_component}'] = feature

        csv_data.append(row)

    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    return csv_filename


@app.route('/files')
def list_files():
    """获取输出文件夹中的所有CSV文件"""
    try:
        files = []
        if os.path.exists(app.config['OUTPUT_FOLDER']):
            for filename in os.listdir(app.config['OUTPUT_FOLDER']):
                if filename.endswith('.csv'):
                    filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
                    file_info = {
                        'filename': filename,
                        'size': os.path.getsize(filepath),
                        'modified': datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d %H:%M:%S')
                    }
                    files.append(file_info)

        # 按修改时间倒序排列
        files.sort(key=lambda x: x['modified'], reverse=True)
        return jsonify({'files': files})

    except Exception as e:
        return jsonify({'error': f'获取文件列表时出错: {str(e)}'}), 500


@app.route('/download/<filename>')
def download_file(filename):
    """下载CSV文件"""
    try:
        return send_file(
            os.path.join(app.config['OUTPUT_FOLDER'], filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': f'下载文件时出错: {str(e)}'}), 500


@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    """删除CSV文件"""
    try:
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify({'success': True, 'message': '文件删除成功'})
        else:
            return jsonify({'error': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'error': f'删除文件时出错: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)