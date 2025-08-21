import torch
from PIL import Image
from modelCNN import SkinCancerCNN
from transforms import get_transforms
import logging

logger = logging.getLogger(__name__)

def load_model(model_path='best_model_focal.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        model = SkinCancerCNN(pretrained=False).to(device)
        
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded model from checkpoint dict")
        else:
            model.load_state_dict(checkpoint)
            logger.info("Loaded model from state dict")
            
        model.eval()
        logger.info(f"Model loaded successfully on {device}")
        return model, device
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise e

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
    model, device = load_model('best_model_focal.pth')
    print("Model loaded successfully")