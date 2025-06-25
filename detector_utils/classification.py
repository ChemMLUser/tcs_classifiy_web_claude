import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os


class ClassificationModel:
    def __init__(self, model_path):
        """
        初始化分类模型

        Args:
            model_path: 训练好的分类模型路径，如果为None则使用默认路径
        """
        # 如果没有指定路径，使用默认路径
        if model_path is None:
            self.model_path = '../model_generator/models/knn_model.pkl'
        else:
            self.model_path = model_path

        self.model = None
        self.scaler = None
        self.classes = ['TC', 'OTC', 'CTC']  # 三种四环素类型
        self.load_model()

    def load_model(self):
        """加载训练好的分类模型"""
        try:
            if os.path.exists(self.model_path):
                # 加载模型文件
                model_data = joblib.load(self.model_path)

                # 检查模型数据格式
                if isinstance(model_data, dict):
                    self.model = model_data.get('model')
                    self.scaler = model_data.get('scaler')
                    self.classes = model_data.get('classes', self.classes)
                else:
                    # 如果只是模型对象
                    self.model = model_data
                    self.scaler = None

                print(f"分类模型加载成功: {self.model_path}")
                print(f"模型类型: {type(self.model)._name_}")
                print(f"支持的分类: {self.classes}")

            else:
                print(f"模型文件不存在: {self.model_path}")
                print("将使用模拟预测模式")
                self.model = None

        except Exception as e:
            print(f"加载分类模型失败: {e}")
            print("将使用模拟预测模式")
            self.model = None

    def preprocess_features(self, features):
        """
        预处理特征数据

        Args:
            features: 特征数据 (n_samples, 12) - 每个样品12维特征

        Returns:
            预处理后的特征数据
        """
        try:
            features = np.array(features)

            # 确保特征维度正确
            if features.ndim == 1:
                features = features.reshape(1, -1)

            if features.shape[1] != 12:
                raise ValueError(f"特征维度错误，期望12维，实际{features.shape[1]}维")

            # 标准化特征
            if self.scaler is not None:
                features = self.scaler.transform(features)
            else:
                # 简单的标准化
                features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)

            return features

        except Exception as e:
            print(f"特征预处理时出错: {e}")
            return features

    def predict(self, features):
        """
        进行分类预测

        Args:
            features: 特征数据 (n_samples, 12)

        Returns:
            预测结果列表
        """
        try:
            # 预处理特征
            processed_features = self.preprocess_features(features)

            if self.model is not None:
                # 使用真实模型进行预测
                predictions = self.model.predict(processed_features)

                # 如果预测结果是数字，转换为类别名称
                if isinstance(predictions[0], (int, np.integer)):
                    predictions = [self.classes[pred] for pred in predictions]

                return predictions
            else:
                # 模拟预测模式
                return self._simulate_prediction(processed_features)

        except Exception as e:
            print(f"预测时出错: {e}")
            return self._simulate_prediction(features)

    def predict_proba(self, features):
        """
        预测概率

        Args:
            features: 特征数据 (n_samples, 12)

        Returns:
            预测概率矩阵
        """
        try:
            processed_features = self.preprocess_features(features)

            if self.model is not None and hasattr(self.model, 'predict_proba'):
                return self.model.predict_proba(processed_features)
            elif self.model is not None:
                # 对于KNN等模型，如果没有predict_proba方法，可以使用distance来计算概率
                if hasattr(self.model, 'kneighbors'):
                    # KNN模型的概率计算
                    distances, indices = self.model.kneighbors(processed_features)
                    n_samples = len(processed_features)
                    n_classes = len(self.classes)
                    proba = np.zeros((n_samples, n_classes))

                    for i in range(n_samples):
                        # 获取最近邻的标签
                        neighbor_labels = self.model._y[indices[i]]
                        # 计算每个类别的概率
                        for j, class_name in enumerate(self.classes):
                            proba[i, j] = np.sum(neighbor_labels == class_name) / len(neighbor_labels)

                    return proba
                else:
                    # 其他没有概率预测的模型，返回one-hot编码
                    predictions = self.predict(features)
                    n_samples = len(predictions)
                    n_classes = len(self.classes)
                    proba = np.zeros((n_samples, n_classes))

                    for i, pred in enumerate(predictions):
                        class_idx = self.classes.index(pred)
                        proba[i, class_idx] = 1.0

                    return proba
            else:
                # 模拟概率预测
                n_samples = len(processed_features)
                n_classes = len(self.classes)

                # 生成随机概率，但让最大值比较明显
                proba = np.random.rand(n_samples, n_classes)

                # 增强最大值
                max_indices = np.argmax(proba, axis=1)
                for i, max_idx in enumerate(max_indices):
                    proba[i, max_idx] += 0.5

                # 归一化
                proba = proba / np.sum(proba, axis=1, keepdims=True)

                return proba

        except Exception as e:
            print(f"预测概率时出错: {e}")
            # 返回默认概率
            n_samples = len(features)
            return np.ones((n_samples, len(self.classes))) / len(self.classes)

    def _simulate_prediction(self, features):
        """
        模拟预测（当模型文件不存在时使用）

        Args:
            features: 特征数据

        Returns:
            模拟预测结果
        """
        n_samples = len(features)
        predictions = []

        for i in range(n_samples):
            # 基于特征值的简单规则进行模拟预测
            feature_row = features[i]

            # 计算RGB各通道的平均值
            r_avg = np.mean([feature_row[j] for j in range(0, 12, 3)])  # R通道
            g_avg = np.mean([feature_row[j] for j in range(1, 12, 3)])  # G通道
            b_avg = np.mean([feature_row[j] for j in range(2, 12, 3)])  # B通道

            # 基于颜色特征进行简单分类
            if r_avg > g_avg and r_avg > b_avg:
                pred = 'TC'  # 红色偏多
            elif g_avg > r_avg and g_avg > b_avg:
                pred = 'OTC'  # 绿色偏多
            else:
                pred = 'CTC'  # 蓝色偏多或平衡

            predictions.append(pred)

        return predictions

    def get_feature_importance(self):
        """
        获取特征重要性（如果模型支持）

        Returns:
            特征重要性数组或None
        """
        try:
            if self.model is not None and hasattr(self.model, 'feature_importances_'):
                return self.model.feature_importances_
            elif self.model is not None and hasattr(self.model, 'coef_'):
                return np.abs(self.model.coef_[0])
            else:
                return None
        except Exception as e:
            print(f"获取特征重要性时出错: {e}")
            return None

    def evaluate_model(self, X_test, y_test):
        """
        评估模型性能

        Args:
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            评估结果字典
        """
        try:
            from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

            if self.model is None:
                return {"error": "模型未加载"}

            # 预测
            y_pred = self.predict(X_test)
            y_proba = self.predict_proba(X_test)

            # 计算指标
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=self.classes)
            cm = confusion_matrix(y_test, y_pred)

            return {
                "accuracy": accuracy,
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
                "predictions": y_pred,
                "probabilities": y_proba.tolist()
            }

        except Exception as e:
            print(f"评估模型时出错: {e}")
            return {"error": str(e)}


def create_sample_model():
    """
    创建示例分类模型（用于测试）
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import numpy as np

        # 生成示例训练数据
        np.random.seed(42)
        n_samples = 300
        n_features = 12

        # 为每个类别生成特征
        X_tc = np.random.normal(100, 20, (n_samples // 3, n_features))  # TC类
        X_otc = np.random.normal(150, 25, (n_samples // 3, n_features))  # OTC类
        X_ctc = np.random.normal(200, 30, (n_samples // 3, n_features))  # CTC类

        X = np.vstack([X_tc, X_otc, X_ctc])
        y = ['TC'] * (n_samples // 3) + ['OTC'] * (n_samples // 3) + ['CTC'] * (n_samples // 3)

        # 训练模型
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        # 保存模型
        model_data = {
            'model': model,
            'scaler': scaler,
            'classes': ['TC', 'OTC', 'CTC']
        }

        os.makedirs('model', exist_ok=True)
        joblib.dump(model_data, 'model/best_classification_model.pkl')

        print("示例分类模型创建成功")
        return model_data

    except Exception as e:
        print(f"创建示例模型时出错: {e}")
        return None
