import torch
from torchvision import models, transforms
import torch.nn as nn
import os
from PIL import Image

NUM_CLASSES = 7
MODEL_ROOTS = {
    "128_pretrained_random_cropping": "./logs_random_cropping/logs_128_pretrained",
    "128_scratch_random_cropping": "./logs_random_cropping/logs_128_scratch",
    "224_pretrained_random_cropping": "./logs_random_cropping/logs_224_pretrained",
    "224_scratch_random_cropping": "./logs_random_cropping/logs_224_scratch",
    "224_pretrained_center_cropping": "./logs_center_cropping/logs_224_pretrained",
    "128_pretrained_center_cropping": "./logs_center_cropping/logs_128_pretrained",
    "224_pretrained_center_decay": "./logs_best_models_weight_decay_center/logs_224_pretrained",
    "224_pretrained_random_decay": "./logs_best_models_weight_decay_random/logs_224_pretrained",
}

classes = ['Badminton', 'Cricket', 'Tennis', 'Swimming', 'Soccer', 'Wrestling', 'Karate']

def get_model_paths_by_input_size(selected_size):
    base = MODEL_ROOTS[selected_size]
    ckpt_dir = os.path.join(base, "checkpoints")
    model_paths = []
    print(f"Searching for models in {ckpt_dir}")
    for d in os.listdir(ckpt_dir):
        if os.path.isdir(os.path.join(ckpt_dir, d)):
            model_path = os.path.join(ckpt_dir, d)
            val_acc = get_val_accuracy(model_path)
            model_paths.append((d, model_path, val_acc))
    model_paths.sort(key=lambda x: x[2], reverse=True)
    return {name: path for name, path, _ in model_paths}

def build_model_from_name(model_name: str, input_size):
    model_name = model_name.lower()
    if "resnet18" in model_name:
        model = models.resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model = freeze_all_but_last_n(model, 2)

    elif  "resnet34" in model_name:
        model = models.resnet34(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model = freeze_all_but_last_n(model, 2)

    elif  "resnet50" in model_name:
        model = models.resnet50(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model = freeze_all_but_last_n(model, 2)

    elif  "resnet101" in model_name:
        model = models.resnet101(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model = freeze_all_but_last_n(model, 2)

    elif  "resnet152" in model_name:
        model = models.resnet152(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model = freeze_all_but_last_n(model, 2)

    elif  "vgg16" in model_name:
        model = models.vgg16(weights="DEFAULT")
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
        model = freeze_all_but_last_n(model, 1)

    elif  "alexnet" in model_name:
        model = models.alexnet(weights="DEFAULT")
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, NUM_CLASSES)
        model = freeze_all_but_last_n(model, 1)

    elif  "googlenet" in model_name:
        model = models.googlenet(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        model = freeze_all_but_last_n(model, 2)

    elif model_name.startswith("simplenet"):
        model = get_scratch_model(model_name, input_size)

    else:
        raise ValueError(f"Model {model_name} not recognized.")

    return model

# Placeholder for scratch models
def get_scratch_model(name, input_size):
    if input_size == 224:
        if name.startswith("simplenet1"):
            from scratch_models224 import SimpleNet1
            return SimpleNet1()
        elif name.startswith("simplenet2"):
            from scratch_models224 import SimpleNet2
            return SimpleNet2()
        elif name.startswith("simplenet3"):
            from scratch_models224 import SimpleNet3
            return SimpleNet3()
        elif name.startswith("simplenet4"):
            from scratch_models224 import SimpleNet4
            return SimpleNet4()
    elif input_size == 128:
        if name.startswith("simplenet1"):
            from scratch_models128 import SimpleNet1
            return SimpleNet1()
        elif name.startswith("simplenet2"):
            from scratch_models128 import SimpleNet2
            return SimpleNet2()
        elif name.startswith("simplenet3"):
            from scratch_models128 import SimpleNet3
            return SimpleNet3()
        elif name.startswith("simplenet4"):
            from scratch_models128 import SimpleNet4
            return SimpleNet4()
        
    raise ValueError("Unknown scratch model")

def load_model(model_dir, input_size):
    model_name = os.path.basename(model_dir)
    model = build_model_from_name(model_name, input_size)
    state_dict = torch.load(os.path.join(model_dir, "final_model.pt"), map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def preprocess_image(img, input_size=224):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)

def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
    index = predicted.item()
    return classes[index]
    # return predicted.item()

def load_metrics(model_dir):
    path = os.path.join(model_dir, "metrics.pt")
    if os.path.exists(path):
        return torch.load(path, map_location='cpu')
    return {}

def get_val_accuracy(model_dir):
    metrics = load_metrics(model_dir)
    val_acc = metrics.get("val_accs", [])
    return float(max(val_acc)) if val_acc else 0.0

def freeze_all_but_last_n(model, n=2):
    for param in model.parameters():
        param.requires_grad = False

    modules_with_params = [m for m in model.modules() if any(p.requires_grad is False for p in m.parameters())]
    for module in modules_with_params[-n:]:
        for param in module.parameters():
            param.requires_grad = True

    return model
