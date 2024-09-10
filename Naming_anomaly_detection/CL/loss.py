import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(args, tau):
    if args.loss_fn == 'CL':
        return CL(args.device, tau, args.lambd)
    elif args.loss_fn == 'CLCE':
        return CLCE(args.device, tau, args.lambd)
    elif args.loss_fn == 'FoCL':
        return FoCL(args.device, tau)
    else:
        raise ValueError(f"Invalid loss type: {args.loss_fn}")

# train CL and head separately
class CL(nn.Module):
    def __init__(self, device, tau, lambd):
        super().__init__()
        self.device = device
        self.tau = tau
        self.lambd = lambd
        
    def forward(self, layer_embeds, y_true):
        loss_temp = torch.zeros(len(layer_embeds), len(layer_embeds)*2-1, device=self.device, dtype=torch.float)
        for i in range(len(layer_embeds)):
            indice = 1
            pos = True
            for j in range(len(layer_embeds)):
                if i == j:  continue
                if y_true[i] == y_true[j] and pos:
                    loss_temp[i][0] = (F.cosine_similarity(layer_embeds[i].view(1,-1), layer_embeds[j].view(1,-1)) + 1) * 0.5 * self.tau
                    pos = False
                elif y_true[i] != y_true[j]: 
                    loss_temp[i][indice] = (F.cosine_similarity(layer_embeds[i].view(1,-1), layer_embeds[j].view(1,-1)) + 1) * 0.5 * self.tau
                    indice += 1
    
        CL_loss = -nn.LogSoftmax(dim=1)(loss_temp)
        CL_loss = torch.sum(CL_loss, dim=0)[0]
        CL_loss /= len(layer_embeds)
        
        loss = CL_loss
        return loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss
# one label at a time
class CLCE(nn.Module):
    def __init__(self, device, tau, lambd):
        super().__init__()
        self.device = device
        self.tau = tau
        self.lambd = lambd    #tweak
        self.gamma = 2
        print(f"lambda value for CLCE loss: {lambd}")
        
    def forward(self, layer_embeds, y_true, y_pred):
        loss_temp = torch.zeros(len(layer_embeds), len(layer_embeds)*2-1, device=self.device, dtype=torch.float)
        for i in range(len(layer_embeds)):
            indice = 1
            pos = True
            for j in range(len(layer_embeds)):
                if i == j:  continue
                if y_true[i] == y_true[j] and pos:
                    loss_temp[i][0] = (F.cosine_similarity(layer_embeds[i].view(1,-1), layer_embeds[j].view(1,-1)) + 1) * 0.5 * self.tau
                    pos = False
                elif y_true[i] != y_true[j]:
                    loss_temp[i][indice] = (F.cosine_similarity(layer_embeds[i].view(1,-1), layer_embeds[j].view(1,-1)) + 1) * 0.5 * self.tau
                    indice += 1         

        CL_loss = -nn.LogSoftmax(dim=1)(loss_temp)
        CL_loss = torch.sum(CL_loss, dim=0)[0]
        CL_loss /= len(layer_embeds)

        CE_loss = nn.CrossEntropyLoss()(y_pred, y_true)
        loss = self.lambd * CL_loss + (1 - self.lambd) * CE_loss

        return loss
    
class FoCL(nn.Module):
    def __init__(self, device, tau):
        super().__init__()
        self.device = device
        self.tau = tau
        self.lambd = 0.3    #tweak
        self.gamma = 2
        
    def forward(self, layer_embeds, y_true, y_pred):
        loss_temp = torch.zeros(len(layer_embeds), len(layer_embeds)*2-1, device=self.device, dtype=torch.float)
        for i in range(len(layer_embeds)):
            indice = 1
            pos = True
            for j in range(len(layer_embeds)):
                if i == j:  continue
                if y_true[i] == y_true[j] and pos:
                    loss_temp[i][0] = (F.cosine_similarity(layer_embeds[i].view(1,-1), layer_embeds[j].view(1,-1)) + 1) * 0.5 * self.tau
                    pos = False
                elif y_true[i] != y_true[j]:
                    loss_temp[i][indice] = (F.cosine_similarity(layer_embeds[i].view(1,-1), layer_embeds[j].view(1,-1)) + 1) * 0.5 * self.tau
                    indice += 1         

        CL_loss = -nn.LogSoftmax(dim=1)(loss_temp)
        CL_loss = torch.sum(CL_loss, dim=0)[0]
        CL_loss /= len(layer_embeds)
        
        CE_loss = nn.CrossEntropyLoss()(y_pred, y_true)
        pt = torch.exp(-CE_loss)
        focal_loss = (1-pt) ** self.gamma * CE_loss
        
        loss = self.lambd * CL_loss + (1 - self.lambd) * focal_loss
        return loss
    