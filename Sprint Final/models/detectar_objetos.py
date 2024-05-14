from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image

def detectar_objetos_en_imagen(image_path):
    image = Image.open(image_path)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # Convertir los outputs (bounding boxes y class logits) a COCO API
    # Vamos a mantener solo las detecciones con un score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detected_objects.append(model.config.id2label[label.item()])
    
    return detected_objects
