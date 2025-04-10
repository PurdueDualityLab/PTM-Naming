import torch
import torch.nn as nn
import torch.nn.functional as F

def contrastive_loss(device, loss_fn, tau, lambd, label_type):
    if loss_fn == 'CL':
        return CL(device, tau, lambd)
    elif loss_fn == 'CLCE':
        return OptimizedCLCE(device, tau, lambd, label_type)
        # return CLCE(device, tau, lambd, label_type)
    elif loss_fn == 'FoCL':
        return FoCL(device, tau)
    else:
        raise ValueError(f"Invalid loss type: {loss_fn}")

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
    def __init__(self, device, tau, lambd, label_type):
        super().__init__()
        self.device = device
        self.tau = tau
        self.lambd = lambd    #tweak
        # self.gamma = 2
        self.label_type = label_type
        print(f"lambda value for CLCE loss: {lambd}")
        
    def forward(self, layer_embeds, logits, y_true):
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

        if self.label_type == 'task':
            CE_loss = nn.BCEWithLogitsLoss()(logits, y_true.float())
        else:
            CE_loss = nn.CrossEntropyLoss()(logits, y_true)
        loss = self.lambd * CL_loss + (1 - self.lambd) * CE_loss

        return loss
    
class OptimizedCLCE(nn.Module):
    def __init__(self, device, tau, lambd, label_type):
        super().__init__()
        self.device = device
        self.tau = tau
        self.lambd = lambd
        self.label_type = label_type
        print(f"lambda value for Optimized CLCE loss: {lambd}")
        
    def forward(self, layer_embeds, logits, y_true):
        # Normalize embeddings
        layer_embeds = F.normalize(layer_embeds, p=2, dim=1)  # L2 normalization
        
        # Compute cosine similarity matrix
        sim_matrix = torch.mm(layer_embeds, layer_embeds.T) / self.tau  # Temperature scaling

        # Mask self-similarity (diagonal elements)
        batch_size = layer_embeds.shape[0]
        mask = torch.eye(batch_size, device=self.device).bool()
        sim_matrix.masked_fill_(mask, float('-inf'))  # Ignore self-similarity

        # Handle different label types for contrastive loss
        if self.label_type == 'task':  # Multi-label case
            # Compute assimilation function (Jaccard index)
            # def jaccard_index(labels_i, labels_p):
            #     labels_i = labels_i.to(torch.int)
            #     labels_p = labels_p.to(torch.int) 
            #     intersection = (labels_i * labels_p).sum(dim=0)
            #     union = (labels_i + labels_p).clamp(0, 1).sum(dim=0)
            #     return intersection / (union + 1e-8)
            
            # # Compute assimilation matrix (size: batch_size x batch_size)
            # assimilation_matrix = torch.zeros((batch_size, batch_size), device=self.device)
            # for i in range(batch_size):
            #     for j in range(batch_size):
            #         assimilation_matrix[i, j] = jaccard_index(y_true[i], y_true[j])
            y_true_int = y_true.to(torch.int)
            intersection = torch.mm(y_true_int.float(), y_true_int.float().T)
            sum_y = y_true_int.sum(dim=1).float().unsqueeze(1)
            union = sum_y + sum_y.T - intersection
            assimilation_matrix = intersection / (union + 1e-8)
            
            c = 0.5  # Threshold for similarity
            pos_mask = (assimilation_matrix >= c).float()
        else:  # Single-label case
            pos_mask = (y_true.view(-1, 1) == y_true.view(1, -1)).float()
        pos_mask = pos_mask * (~mask).float()
        # Compute SupCon loss
        exp_sim = torch.exp(sim_matrix)
        denom = torch.sum(exp_sim, dim=1, keepdim=True)
        
        # Select positive pairs
        pos_exp_sim = exp_sim * pos_mask
        numerator = torch.sum(pos_exp_sim, dim=1, keepdim=True)
        
        # Compute loss only for valid samples (those with positive pairs)
        per_sample_loss = -torch.log(numerator / denom + 1e-8)
        
        row_sum = torch.sum(pos_mask, dim=1)
        valid_samples = (row_sum > 0)
        
        if valid_samples.sum() > 0:
            SupCon_loss = torch.sum(per_sample_loss * valid_samples.float().view(-1, 1)) / valid_samples.sum()
        else:
            # If no valid samples in the batch, set contrastive loss to 0
            SupCon_loss = torch.tensor(0.0, device=self.device)
        
        # Compute classification loss
        if self.label_type == 'task':
            # Multi-label classification - use BCE loss
            CE_loss = nn.BCEWithLogitsLoss()(logits, y_true.float())
        else:
            # Single-label classification - use CrossEntropy loss
            if len(y_true.shape) > 1 and y_true.shape[1] == 1:
                CE_loss = nn.CrossEntropyLoss()(logits, y_true.squeeze(1))
            else:
                CE_loss = nn.CrossEntropyLoss()(logits, y_true.squeeze())
        # Combine losses
        loss = self.lambd * SupCon_loss + (1 - self.lambd) * CE_loss
        print(f"CL loss: {SupCon_loss.item()}, CE loss: {CE_loss.item()}, combined loss: {loss.item()}")
        
    
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
    