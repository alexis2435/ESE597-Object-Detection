# detect.py

from ultralytics import YOLO
import cv2

def main():
    # IMPORTANT: Update this path to point to the weights of YOUR trained model.
    # After running train.py, the path will look something like 'runs/detect/train/weights/best.pt'
    # Check the output of train.py for the exact path.
    model_path = 'runs/detect/train/weights/best.pt' # <-- Make sure to update this path!

    # Load your custom-trained model
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the path to 'best.pt' is correct.")
        return

    # Define the source image for detection
    # Ultralytics provides a sample bus image for easy testing.
    source_image = 'https://ultralytics.com/images/bus.jpg'
    
    print(f"Running inference on {source_image}...")
    # Run inference on the source
    results = model(source_image)  # This can be an image path, URL, or video file

    # The results object contains the detections. We can visualize them.
    # This will display the image with bounding boxes drawn on it.
    for r in results:
        im_array = r.plot()  # plot a BGR numpy image of predictions
        cv2.imshow('YOLOv8 Inference', im_array)
        
        # Wait for a key press to close the image window
        print("Inference complete. Press any key to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()