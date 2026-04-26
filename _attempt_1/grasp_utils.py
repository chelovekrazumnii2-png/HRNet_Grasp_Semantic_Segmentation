import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_sample(x, y):
    # 1. Переводим тензоры в Numpy и переносим на CPU
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    
    # 2. Подготовка RGB (берем первые 3 канала и меняем [C, H, W] -> [H, W, C])
    rgb = x_np[:3].transpose(1, 2, 0)
    # Клипаем значения, чтобы matplotlib не ругался на точность
    rgb = np.clip(rgb, 0, 1)
    
    # 3. Извлекаем маски
    pos = y_np[0]       # Качество
    sin = y_np[1]       # Синус
    width = y_np[3]     # Ширина
    
    plt.figure(figsize=(18, 6))
    
    # Слой 1: RGB + Маска качества
    plt.subplot(1, 3, 1)
    plt.imshow(rgb)
    # Накладываем маску: там где 0 — прозрачно, там где 1 — красный цвет
    masked_pos = np.ma.masked_where(pos == 0, pos)
    plt.imshow(masked_pos, cmap='autumn', alpha=0.8)
    plt.title("RGB + Grasp Zones")
    plt.axis('off')
    
    # Слой 2: Углы (Sin)
    plt.subplot(1, 3, 2)
    plt.imshow(sin, cmap='jet')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Angle Sin(2θ) Map")
    plt.axis('off')
    
    # Слой 3: Ширина
    plt.subplot(1, 3, 3)
    plt.imshow(width, cmap='magma')
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.title("Width Map")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def debug_predict(model, dataset, device, index=0, save_dir=None):
    """
    Сравнивает входное изображение, эталонную маску качества и предсказание сети.
    """
    model.eval()
    img, target = dataset[index]
    
    with torch.no_grad():
        input_tensor = img.unsqueeze(0).to(device)
        output = model(input_tensor)
        quality_map = torch.sigmoid(output[0, 0, :, :]).cpu().numpy()
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Вход (RGB)")
    plt.imshow(img[:3, :, :].permute(1, 2, 0).numpy())
    
    plt.subplot(1, 3, 2)
    plt.title("Эталон (Target Quality)")
    plt.imshow(target[0, :, :].numpy(), cmap='jet')
    
    plt.subplot(1, 3, 3)
    plt.title("Предсказание (Pred Quality)")
    plt.imshow(quality_map, cmap='jet')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"debug_pred_idx_{index}.png"), bbox_inches='tight')
    plt.show()

def visualize_grasp_pro(model, dataset, device, index=0, save_dir=None):
    """
    Ищет лучшую точку захвата и рисует вектор подхода и ширину раскрытия клешни.
    """
    model.eval()
    img, target = dataset[index]
    
    with torch.no_grad():
        input_tensor = img.unsqueeze(0).to(device)
        output = model(input_tensor)
        
        q = torch.sigmoid(output[0, 0, :, :]).cpu().numpy()
        sin = output[0, 1, :, :].cpu().numpy()
        cos = output[0, 2, :, :].cpu().numpy()
        width = output[0, 3, :, :].cpu().numpy() * 150
        
    y, x = np.unravel_index(np.argmax(q), q.shape)
    angle = np.arctan2(sin[y, x], cos[y, x]) / 2
    
    plt.figure(figsize=(6, 6))
    rgb = img[:3, :, :].permute(1, 2, 0).numpy()
    plt.imshow(rgb)
    
    dx, dy = np.cos(angle) * 40, np.sin(angle) * 40
    plt.arrow(x, y, dx, dy, color='red', head_width=10, label='Направление')
    
    wx, wy = np.cos(angle + np.pi/2) * (width[y, x] / 2), np.sin(angle + np.pi/2) * (width[y, x] / 2)
    plt.plot([x - wx, x + wx], [y - wy, y + wy], color='lime', linewidth=3, label='Ширина')
    
    plt.scatter(x, y, color='yellow', s=100)
    plt.title(f"Grasp Prediction\nAngle: {np.degrees(angle):.1f}° | Width: {width[y, x]:.1f} px")
    plt.legend()
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"grasp_pro_idx_{index}.png"), bbox_inches='tight')
    plt.show()

def visualize_dataset_hedgehog(dataset, index=15, step=4, save_dir=None):
    """
    Распаковывает тензор датасета обратно в линии захвата (Ground Truth).
    """
    img, target = dataset[index]
    
    q = target[0].numpy()
    sin = target[1].numpy()
    cos = target[2].numpy()
    width = target[3].numpy() * 150
    
    y_idx, x_idx = np.where(q > 0.9)
    
    plt.figure(figsize=(10, 10))
    rgb = img[:3, :, :].permute(1, 2, 0).numpy()
    plt.imshow(rgb)
    
    for i in range(0, len(y_idx), step):
        y, x = y_idx[i], x_idx[i]
        angle = np.arctan2(sin[y, x], cos[y, x]) / 2
        w = width[y, x]
        wx, wy = np.cos(angle + np.pi/2) * (w / 2), np.sin(angle + np.pi/2) * (w / 2)
        plt.plot([x - wx, x + wx], [y - wy, y + wy], color='blue', alpha=0.6, linewidth=2)
        plt.scatter(x, y, color='red', s=5, alpha=0.5)

    plt.title(f"Исходный 'Ёж' (Index: {index}) | Точек: {len(y_idx)}")
    plt.axis('off')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"gt_hedgehog_idx_{index}.png"), bbox_inches='tight')
    plt.show()

def visualize_all_predictions(model, dataset, device, index=15, threshold=0.7, step=5, save_dir=None):
    """
    Показывает ВСЕ варианты захвата, в которых сеть уверена выше заданного порога.
    """
    model.eval()
    img, _ = dataset[index]
    
    with torch.no_grad():
        input_tensor = img.unsqueeze(0).to(device)
        output = model(input_tensor)
        q = torch.sigmoid(output[0, 0, :, :]).cpu().numpy()
        sin = output[0, 1, :, :].cpu().numpy()
        cos = output[0, 2, :, :].cpu().numpy()
        width = output[0, 3, :, :].cpu().numpy() * 150
        
    plt.figure(figsize=(10, 10))
    rgb = img[:3, :, :].permute(1, 2, 0).numpy()
    plt.imshow(rgb)
    
    y_idx, x_idx = np.where(q > threshold)
    
    for i in range(0, len(y_idx), step):
        y, x = y_idx[i], x_idx[i]
        angle = np.arctan2(sin[y, x], cos[y, x]) / 2
        w = width[y, x]
        wx, wy = np.cos(angle + np.pi/2) * (w / 2), np.sin(angle + np.pi/2) * (w / 2)
        plt.plot([x - wx, x + wx], [y - wy, y + wy], color='lime', alpha=0.5, linewidth=2)
        
    plt.title(f"Все предсказания (Порог: {threshold}) | Index: {index}")
    plt.axis('off')
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"all_preds_idx_{index}.png"), bbox_inches='tight')
    plt.show()