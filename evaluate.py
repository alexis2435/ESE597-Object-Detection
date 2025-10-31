from ultralytics import YOLO
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def evaluate_model_accuracy(model_path, data_yaml='coco128.yaml'):
    """评估模型精度（mAP等指标）"""
    print("\n" + "=" * 60)
    print("模型精度评估")
    print("=" * 60)

    model = YOLO(model_path)
    print(f"✓ 已加载模型: {model_path}")

    print("\n正在计算精度指标...")
    metrics = model.val(data=data_yaml, verbose=False)

    print("\n" + "=" * 60)
    print("精度指标 (Accuracy Metrics)")
    print("=" * 60)
    print(f"mAP50      : {metrics.box.map50:.4f}  (IoU=0.5时的平均精度)")
    print(f"mAP50-95   : {metrics.box.map:.4f}  (IoU=0.5-0.95的平均精度)")
    print(f"Precision  : {metrics.box.mp:.4f}  (精确率 - 预测为正的样本中真正为正的比例)")
    print(f"Recall     : {metrics.box.mr:.4f}  (召回率 - 所有正样本中被正确预测的比例)")
    print(
        f"F1 Score   : {2 * (metrics.box.mp * metrics.box.mr) / (metrics.box.mp + metrics.box.mr):.4f}  (精确率和召回率的调和平均)")
    print("=" * 60)

    return metrics


def evaluate_inference_speed(model_path, test_image='https://ultralytics.com/images/bus.jpg', num_runs=100):
    """评估推理速度（FPS）"""
    print("\n" + "=" * 60)
    print("推理速度评估")
    print("=" * 60)

    model = YOLO(model_path)
    print(f"✓ 已加载模型: {model_path}")

    # 加载测试图片
    if test_image.startswith('http'):
        import urllib.request
        req = urllib.request.urlopen(test_image)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, -1)
    else:
        img = cv2.imread(test_image)

    print(f"✓ 测试图片尺寸: {img.shape}")
    print(f"\n运行 {num_runs} 次推理测试...")

    # 预热
    for _ in range(10):
        _ = model(img, verbose=False)

    # 测试推理速度
    times = []
    for i in range(num_runs):
        start = time.time()
        results = model(img, verbose=False)
        elapsed = time.time() - start
        times.append(elapsed)

        if (i + 1) % 20 == 0:
            print(f"  进度: {i + 1}/{num_runs}")

    times = np.array(times)

    print("\n" + "=" * 60)
    print("速度指标 (Speed Metrics)")
    print("=" * 60)
    print(f"平均推理时间  : {np.mean(times) * 1000:.2f} ms")
    print(f"最快推理时间  : {np.min(times) * 1000:.2f} ms")
    print(f"最慢推理时间  : {np.max(times) * 1000:.2f} ms")
    print(f"标准差        : {np.std(times) * 1000:.2f} ms")
    print(f"平均 FPS      : {1 / np.mean(times):.2f} 帧/秒")
    print("=" * 60)

    return times


def compare_models(model_paths, labels, data_yaml='coco128.yaml'):
    """对比多个模型的性能"""
    print("\n" + "=" * 60)
    print("模型对比分析")
    print("=" * 60)

    results = {}

    for model_path, label in zip(model_paths, labels):
        print(f"\n评估模型: {label}")
        print("-" * 60)

        try:
            model = YOLO(model_path)

            # 评估精度
            metrics = model.val(data=data_yaml, verbose=False)

            # 评估速度
            img = cv2.imread('https://ultralytics.com/images/bus.jpg')
            times = []
            for _ in range(50):
                start = time.time()
                _ = model(img, verbose=False)
                times.append(time.time() - start)

            results[label] = {
                'mAP50': metrics.box.map50,
                'mAP50-95': metrics.box.map,
                'precision': metrics.box.mp,
                'recall': metrics.box.mr,
                'fps': 1 / np.mean(times)
            }

            print(f"✓ mAP50: {metrics.box.map50:.4f}, FPS: {1 / np.mean(times):.2f}")

        except Exception as e:
            print(f"✗ 评估失败: {e}")

    # 生成对比图表
    if len(results) > 1:
        plot_comparison(results)

    return results


def plot_comparison(results):
    """绘制模型对比图表"""
    labels = list(results.keys())

    metrics = {
        'mAP50': [results[l]['mAP50'] for l in labels],
        'mAP50-95': [results[l]['mAP50-95'] for l in labels],
        'Precision': [results[l]['precision'] for l in labels],
        'Recall': [results[l]['recall'] for l in labels],
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')

    for idx, (metric_name, values) in enumerate(metrics.items()):
        ax = axes[idx // 2, idx % 2]
        bars = ax.bar(labels, values, color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D'][:len(labels)])
        ax.set_ylabel(metric_name, fontsize=12)
        ax.set_ylim([0, 1])
        ax.grid(axis='y', alpha=0.3)

        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ 对比图表已保存: model_comparison.png")
    plt.show()


def test_different_conditions(model_path):
    """测试不同环境条件下的性能"""
    print("\n" + "=" * 60)
    print("不同条件下的性能测试")
    print("=" * 60)

    model = YOLO(model_path)
    test_image = cv2.imread('https://ultralytics.com/images/bus.jpg')

    conditions = {
        'Normal': test_image,
        'Dark (Low Light)': cv2.convertScaleAbs(test_image, alpha=0.5, beta=0),
        'Bright (High Light)': cv2.convertScaleAbs(test_image, alpha=1.5, beta=30),
        'Low Contrast': cv2.convertScaleAbs(test_image, alpha=0.7, beta=0),
        'Blurred': cv2.GaussianBlur(test_image, (15, 15), 0)
    }

    results_summary = {}

    for condition_name, img in conditions.items():
        print(f"\n测试条件: {condition_name}")

        # 运行检测
        results = model(img, verbose=False)

        # 统计检测结果
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        avg_confidence = np.mean([box[4] for box in results[0].boxes.data.cpu().numpy()]) if num_detections > 0 else 0

        results_summary[condition_name] = {
            'detections': num_detections,
            'avg_confidence': avg_confidence
        }

        print(f"  检测数量: {num_detections}")
        print(f"  平均置信度: {avg_confidence:.3f}")

    return results_summary


def generate_report(model_path, output_file='performance_report.txt'):
    """生成完整性能报告"""
    print("\n" + "=" * 60)
    print("生成性能报告")
    print("=" * 60)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("YOLOv8 模型性能评估报告\n")
        f.write("=" * 60 + "\n\n")

        # 1. 精度评估
        f.write("1. 精度指标\n")
        f.write("-" * 60 + "\n")
        metrics = evaluate_model_accuracy(model_path)
        f.write(f"mAP50      : {metrics.box.map50:.4f}\n")
        f.write(f"mAP50-95   : {metrics.box.map:.4f}\n")
        f.write(f"Precision  : {metrics.box.mp:.4f}\n")
        f.write(f"Recall     : {metrics.box.mr:.4f}\n\n")

        # 2. 速度评估
        f.write("2. 速度指标\n")
        f.write("-" * 60 + "\n")
        times = evaluate_inference_speed(model_path, num_runs=50)
        f.write(f"平均推理时间: {np.mean(times) * 1000:.2f} ms\n")
        f.write(f"平均 FPS    : {1 / np.mean(times):.2f} 帧/秒\n\n")

        # 3. 不同条件测试
        f.write("3. 不同环境条件性能\n")
        f.write("-" * 60 + "\n")
        condition_results = test_different_conditions(model_path)
        for condition, result in condition_results.items():
            f.write(f"{condition}:\n")
            f.write(f"  检测数量: {result['detections']}\n")
            f.write(f"  平均置信度: {result['avg_confidence']:.3f}\n")

        f.write("\n" + "=" * 60 + "\n")
        f.write("报告生成完成\n")
        f.write("=" * 60 + "\n")

    print(f"✓ 性能报告已保存: {output_file}")


def main():
    print("\n" + "=" * 60)
    print("YOLOv8 性能评估系统")
    print("=" * 60)
    print("\n请选择评估模式:")
    print("1. 完整性能评估（精度 + 速度）")
    print("2. 仅评估精度（mAP）")
    print("3. 仅评估速度（FPS）")
    print("4. 不同条件测试")
    print("5. 模型对比")
    print("6. 生成完整报告")

    choice = input("\n请输入选项 (1-6): ").strip()

    # 默认模型路径
    model_path = 'runs/detect/train_augmented/weights/best.pt'

    if choice == '1':
        # 完整评估
        evaluate_model_accuracy(model_path)
        evaluate_inference_speed(model_path)

    elif choice == '2':
        # 精度评估
        evaluate_model_accuracy(model_path)

    elif choice == '3':
        # 速度评估
        evaluate_inference_speed(model_path)

    elif choice == '4':
        # 不同条件测试
        test_different_conditions(model_path)

    elif choice == '5':
        # 模型对比
        print("\n输入要对比的模型路径（用逗号分隔）:")
        paths = input().strip().split(',')
        labels = [f"Model {i + 1}" for i in range(len(paths))]
        compare_models(paths, labels)

    elif choice == '6':
        # 生成完整报告
        generate_report(model_path)

    else:
        print("无效选项")


if __name__ == '__main__':
    main()