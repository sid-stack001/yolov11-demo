from ultralytics import YOLO

# Load a COCO-pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Run inference with the YOLO11n model on the 'bus.jpg' image
results = model("F:\Models\CycleGan\pytorch-CycleGAN-and-pix2pix\input2.mp4", save=True, show=True)
# Save the model to a new file