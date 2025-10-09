# train.py

from ultralytics import YOLO

def main():
    # Load a pre-trained YOLOv8n model.
    # 'yolov8n.pt' is the smallest and fastest model, ideal for testing on a MacBook Air.
    model = YOLO('yolov8n.pt')

    print("Starting model training...")
    # Train the model using the coco128 dataset.
    # Ultralytics will automatically download this dataset for you the first time.
    # data='coco128.yaml' specifies the dataset configuration file.
    # epochs=5 means the model will see the entire dataset 5 times. This is a short run for testing.
    # imgsz=640 resizes images to 640x640 pixels before feeding them to the model.
    # device='mps' tells PyTorch to use Apple's Metal Performance Shaders (MPS) for GPU acceleration.
    results = model.train(data='coco128.yaml', epochs=5, imgsz=640, device='mps')
    
    print("Training complete!")
    print("Results saved in:", results.save_dir)

if __name__ == '__main__':
    main()