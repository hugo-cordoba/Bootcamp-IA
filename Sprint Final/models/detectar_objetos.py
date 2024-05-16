from transformers import DetrImageProcessor, DetrForObjectDetection, AutoTokenizer, AutoModelForCausalLM
import torch
from PIL import Image

# Verifica si CUDA está disponible y limpia la memoria de la GPU
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Cargar modelos y tokenizadores
model_id = "nvidia/Llama3-ChatQA-1.5-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
language_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to('cuda' if torch.cuda.is_available() else 'cpu')

# Cargar procesador y modelo para detección de objetos
object_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
object_detection_model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

def detectar_objetos_en_imagen(image_path):
    image = Image.open(image_path)
    inputs = object_processor(images=image, return_tensors="pt")
    
    outputs = object_detection_model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = object_processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

    detected_objects = [object_detection_model.config.id2label[label.item()] for score, label, box in zip(results["scores"], results["labels"], results["boxes"])]
    return detected_objects

def get_formatted_input(messages, context):
    system = ("System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, "
                "detailed, and polite answers to the user's questions based on the context. The assistant should also indicate "
                "when the answer cannot be found in the context.")
    instruction = "Please give a full and complete answer for the question."

    for item in messages:
        if item['role'] == "user":
            item['content'] = instruction + " " + item['content']
            break

    conversation = '\n\n'.join([f"User: {item['content']}" if item["role"] == "user" else f"Assistant: {item['content']}" for item in messages]) + "\n\nAssistant:"
    formatted_input = f"{system}\n\n{context}\n\n{conversation}"
    
    return formatted_input

def get_message(objetos_detectados):
    # Eliminar duplicados manteniendo el orden original
    unique_items = list(dict.fromkeys(objetos_detectados))

    # Convertir la lista de elementos únicos a un string
    result_string = ' '.join(unique_items)

    messages = [
        {"role": "user", "content": "Generate a list of ten relevant and popular Instagram hashtags for the following words: " + result_string + ". Provide only the list of hashtags, with no additional text or explanation. Give only the best 10 hashtags."}
    ]

    document = ("You are an advanced AI model specialized in social media content creation. Your task is to generate relevant and popular hashtags for Instagram posts based on a given keyword. These hashtags should help increase the visibility and engagement of the post. Consider trending topics, popular culture, and commonly used hashtags within the Instagram community. Ensure the hashtags are appropriate and related to the keyword provided.")

    formatted_input = get_formatted_input(messages, document)
    tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to('cuda' if torch.cuda.is_available() else 'cpu')

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = language_model.generate(input_ids=tokenized_prompt.input_ids, attention_mask=tokenized_prompt.attention_mask, max_new_tokens=64, eos_token_id=terminators)

    response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
    hashtags = tokenizer.decode(response, skip_special_tokens=True)
    hashtags_list = hashtags.split()
    return hashtags_list[:10]
