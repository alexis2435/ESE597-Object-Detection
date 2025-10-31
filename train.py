import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from ultralytics import YOLO


def main():
    print("=" * 60)
    print("YOLOv8 Training - Complete COCO Dataset")
    print("=" * 60)

    # 使用完整 COCO 数据集（自动下载）
    model = YOLO('yolov8n.pt')

    print("\nStarting training with full COCO dataset...")
    print("Note: First run will download ~20GB COCO dataset")
    print("-" * 60)

    results = model.train(
        data='coco.yaml',  # ✅ 使用完整 COCO 数据集
        epochs=50,  # ✅ 增加训练轮数
        imgsz=640,
        device='0',
        batch=16,

        # 数据增强参数（保持你的设置）
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,

        # 训练设置
        cache=False,  # ✅ 完整数据集太大，不缓存
        workers=8,
        project='runs/detect',
        name='train_full_coco',
        exist_ok=True,
        patience=10,  # ✅ 早停机制
        save=True,
        plots=True,
        val=True,
    )

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"Best weights: runs/detect/train_full_coco/weights/best.pt")

    # 验证模型
    metrics = model.val()

    print(f"\nPerformance Metrics:")
    print(f"  mAP50     : {metrics.box.map50:.4f}")
    print(f"  mAP50-95  : {metrics.box.map:.4f}")
    print(f"  Precision : {metrics.box.mp:.4f}")
    print(f"  Recall    : {metrics.box.mr:.4f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()