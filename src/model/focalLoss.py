import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, alpha=1.0, gamma=2.0, weight=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        
        pt = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class MelanomaFocusedLoss(nn.Module):

    def __init__(self, melanoma_weight=3.0, confidence_penalty=0.2, base_gamma=2.0):
        super(MelanomaFocusedLoss, self).__init__()
        self.melanoma_weight = melanoma_weight
        self.confidence_penalty = confidence_penalty
        self.base_gamma = base_gamma
        
    def forward(self, outputs, targets):
        probs = F.softmax(outputs, dim=1)
        
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.base_gamma * ce_loss
        
        max_probs = torch.max(probs, dim=1)[0]
        predicted_classes = torch.argmax(outputs, dim=1)
        
        melanoma_mask = (targets == 1)
        predicted_non_melanoma = (predicted_classes == 0)
        confident_wrong_melanoma = melanoma_mask & predicted_non_melanoma & (max_probs > 0.7)
        
        confidence_penalty = confident_wrong_melanoma.float() * self.confidence_penalty * max_probs
        
        weights = torch.ones_like(targets, dtype=torch.float)
        weights[melanoma_mask] = self.melanoma_weight
        
        total_loss = (focal_loss * weights + confidence_penalty).mean()
        
        return total_loss


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        y_pred = torch.sigmoid(x[:, 1])  
        y_true = y.float()
        
        xs_pos = y_pred
        
        xs_neg = 1 - y_pred
        
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        los_pos = y_true * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - y_true) * torch.log(xs_neg.clamp(min=1e-8))

        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(False)
        
        neg_weight = 1 - xs_neg
        pos_weight = 1 - xs_pos
        
        if self.disable_torch_grad_focal_loss:
            torch.set_grad_enabled(True)
            
        loss = los_pos * pos_weight ** self.gamma_pos + los_neg * neg_weight ** self.gamma_neg
        return -loss.mean()


def get_loss_function(loss_type='focal', **kwargs):

    if loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('alpha', 1.0),
            gamma=kwargs.get('gamma', 2.0),
            weight=kwargs.get('class_weights', None)
        )
    
    elif loss_type == 'melanoma_focused':
        return MelanomaFocusedLoss(
            melanoma_weight=kwargs.get('melanoma_weight', 3.0),
            confidence_penalty=kwargs.get('confidence_penalty', 0.2),
            base_gamma=kwargs.get('gamma', 2.0)
        )
    
    elif loss_type == 'asymmetric':
        return AsymmetricLoss(
            gamma_neg=kwargs.get('gamma_neg', 4),
            gamma_pos=kwargs.get('gamma_pos', 1),
            clip=kwargs.get('clip', 0.05)
        )
    
    elif loss_type == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=kwargs.get('class_weights', None))
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

