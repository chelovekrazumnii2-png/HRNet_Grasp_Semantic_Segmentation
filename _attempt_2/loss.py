import torch
import torch.nn as nn
import torch.nn.functional as F

class GraspLoss(nn.Module):
    def __init__(self, lambda_regression=1.0):
        super(GraspLoss, self).__init__()
        self.lambda_regression = lambda_regression
        
        # 1. Лечим дисбаланс классов. 
        # pos_weight=20.0 означает, что ошибка на пикселе захвата 
        # штрафуется в 20 раз сильнее, чем ошибка на фоне.
        self.q_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([20.0]))
        
        # Важно: reduction='none', чтобы мы могли умножить на маску
        self.reg_loss_fn = nn.MSELoss(reduction='none') 

    def forward(self, preds, targets):
        # preds: [B, 4, H, W] -> Q, Cos, Sin, Width
        pred_q = preds[:, 0, ...]
        target_q = targets[:, 0, ...]
        
        # --- 1. Считаем Quality Loss ---
        # Обязательно проверяем, что pos_weight находится на том же устройстве (GPU)
        if self.q_loss_fn.pos_weight.device != pred_q.device:
            self.q_loss_fn.pos_weight = self.q_loss_fn.pos_weight.to(pred_q.device)
            
        loss_q = self.q_loss_fn(pred_q, target_q)

        # --- 2. Считаем Regression Loss ТОЛЬКО в зонах захвата ---
        # Создаем маску: 1 там, где есть захват, 0 - фон
        mask = (target_q > 0.5).float()
        
        # Если в батче вообще нет пикселей захвата (мало ли), избегаем деления на ноль
        mask_sum = mask.sum() + 1e-8 

        pred_cos = preds[:, 1, ...]
        pred_sin = preds[:, 2, ...]
        pred_w = preds[:, 3, ...]
        
        target_cos = targets[:, 1, ...]
        target_sin = targets[:, 2, ...]
        target_w = targets[:, 3, ...]

        # Умножаем поэлементную ошибку на маску и усредняем только по полезным пикселям
        loss_cos = (self.reg_loss_fn(pred_cos, target_cos) * mask).sum() / mask_sum
        loss_sin = (self.reg_loss_fn(pred_sin, target_sin) * mask).sum() / mask_sum
        loss_w = (self.reg_loss_fn(pred_w, target_w) * mask).sum() / mask_sum

        loss_reg = loss_cos + loss_sin + loss_w

        total_loss = loss_q + (self.lambda_regression * loss_reg)

        return total_loss, {
            'loss_q': loss_q.item(), 
            'loss_reg': loss_reg.item()
        }


# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class GraspLoss(nn.Module):
#     def __init__(self, lambda_regression=1.0):
#         super(GraspLoss, self).__init__()
#         self.lambda_reg = lambda_regression
        
#         # Binary Cross Entropy для карты качества (есть захват или нет)
#         self.bce = nn.BCEWithLogitsLoss()
        
#         # Smooth L1 Loss (более стабилен, чем MSE, к выбросам) для параметров
#         self.l1 = nn.SmoothL1Loss(reduction='none')

#     def forward(self, pred, target):
#         """
#         pred: [B, 4, H, W] - выход сети
#         target: [B, 4, H, W] - маски из нашего Dataset
#         """
#         # Разрезаем тензоры на отдельные карты
#         pred_q = pred[:, 0, :, :]
#         pred_cos = pred[:, 1, :, :]
#         pred_sin = pred[:, 2, :, :]
#         pred_w = pred[:, 3, :, :]
        
#         target_q = target[:, 0, :, :]
#         target_cos = target[:, 1, :, :]
#         target_sin = target[:, 2, :, :]
#         target_w = target[:, 3, :, :]

#         # 1. Loss для качества захвата (работает по всей картинке)
#         loss_q = self.bce(pred_q, target_q)

#         # 2. Regression Loss (только там, где реально ЕСТЬ захват в разметке)
#         # Мы создаем маску, чтобы не учить модель углу в пустом месте
#         mask = (target_q > 0.5).float()
        
#         num_positive = mask.sum()
#         if num_positive > 0:
#             loss_cos = (self.l1(pred_cos, target_cos) * mask).sum() / num_positive
#             loss_sin = (self.l1(pred_sin, target_sin) * mask).sum() / num_positive
#             loss_w = (self.l1(pred_w, target_w) * mask).sum() / num_positive
            
#             loss_reg = loss_cos + loss_sin + loss_w
#         else:
#             loss_reg = torch.tensor(0.0).to(pred.device)

#         # Итоговый лосс с весовым коэффициентом
#         total_loss = loss_q + self.lambda_reg * loss_reg
        
#         return total_loss, {"loss_q": loss_q.item(), "loss_reg": loss_reg.item()}