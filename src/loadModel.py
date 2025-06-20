import torch
from PIL import Image
from modelCNN import SkinCancerCNN
from transforms import get_transforms

def load_model(model_path='best.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SkinCancerCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

def predict_image(model, device, image_path):
    transform = get_transforms('val')
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    result = 'Melanoma' if predicted.item() == 1 else 'Non-Melanoma'
    confidence_score = confidence.item()
    
    return result, confidence_score

if __name__ == "__main__":
    model, device = load_model('best.pth')
    print("loaded succ")