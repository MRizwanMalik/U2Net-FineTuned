from ultralytics import YOLO
import cv2
import os

# Pretrained segmentation model
model = YOLO("yolov8x-seg.pt")  
input_folder = "original_img"
mask_folder = "masks_img"
os.makedirs(mask_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if file.endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(input_folder, file)
        results = model(path)

        # Check if segmentation available
        for r in results:
            if hasattr(r, 'masks') and r.masks is not None:
                # Combine all instance masks into one binary mask
                mask = r.masks.data.cpu().numpy()
                combined = (mask.sum(axis=0) > 0).astype("uint8") * 255

                cv2.imwrite(os.path.join(mask_folder, file), combined)
print(" YOLO masks generated successfully!")
