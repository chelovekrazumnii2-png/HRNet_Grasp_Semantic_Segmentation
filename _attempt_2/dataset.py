
import glob
import cv2
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from PIL import Image

class JacquardDataset(Dataset):
    def __init__(self, config, is_train=True):
        self.config = config
        self.dataset_dir = config["DATASET_DIR"]
        self.is_train = is_train
        
        # --- ДОБАВЬТЕ ЭТИ СТРОКИ ---
        self.max_width = config.get("MAX_WIDTH", 100.0) # Берем из конфига или ставим 100 по умолчанию
        self.use_resize = config.get("USE_RESIZE", False)
        self.img_size = config.get("IMG_SIZE", (1024, 1024))
        # ---------------------------

        self.image_paths = []
        for root, dirs, files in os.walk(self.dataset_dir):
            for file in files:
                if file.endswith("RGB.png"):
                    self.image_paths.append(os.path.join(root, file))
        
        # Аугментация
        self.color_aug = T.ColorJitter(0.2, 0.2, 0.2, 0.05) if is_train else None
            
    # ... (остальные методы __len__ и __getitem__ остаются без изменений, 
    # только в __getitem__ добавьте применение self.color_aug(img) к тензору картинки перед нормализацией

    def __len__(self):
        return len(self.image_paths)

    def _create_grasp_masks(self, txt_path, h_img, w_img):
        """
        Парсит текстовый файл и создает 4 маски для плотной регрессии.
        """
        # Инициализируем пустые маски (заполнены нулями)
        mask_q = np.zeros((h_img, w_img), dtype=np.float32)
        mask_cos = np.zeros((h_img, w_img), dtype=np.float32)
        mask_sin = np.zeros((h_img, w_img), dtype=np.float32)
        mask_w = np.zeros((h_img, w_img), dtype=np.float32)
        
        if not os.path.exists(txt_path):
            return mask_q, mask_cos, mask_sin, mask_w

        with open(txt_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split(';')
            if len(parts) < 5:
                continue
            
            # Читаем параметры: центр (x, y), угол в градусах, ширина и высота (толщина губок)
            x, y, theta_deg, w, h = map(float, parts[:5])
            
            # Перевод в радианы
            theta_rad = np.radians(theta_deg)
            
            # Создаем структуру повернутого прямоугольника для OpenCV
            # Формат: ((center_x, center_y), (width, height), angle_in_degrees)
            rect = ((x, y), (w, h), theta_deg)
            box = cv2.boxPoints(rect)
            box = np.intp(box) # Конвертируем координаты в целые числа
            
            # Нормализация ширины
            norm_w = min(w / self.max_width, 1.0)
            
            # Отрисовка закрашенных полигонов на масках
            cv2.fillPoly(mask_q, [box], 1.0)
            cv2.fillPoly(mask_cos, [box], float(np.cos(2 * theta_rad)))
            cv2.fillPoly(mask_sin, [box], float(np.sin(2 * theta_rad)))
            cv2.fillPoly(mask_w, [box], float(norm_w))
            
        return mask_q, mask_cos, mask_sin, mask_w

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Путь к файлу с разметкой (меняем суффикс _RGB.png на _grasps.txt)
        txt_path = img_path.replace('_RGB.png', '_grasps.txt')
        
        # Загрузка изображения (OpenCV загружает в BGR, переводим в RGB)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        h_orig, w_orig = img.shape[:2]
        
        # Генерация масок на оригинальном разрешении
        mask_q, mask_cos, mask_sin, mask_w = self._create_grasp_masks(txt_path, h_orig, w_orig)
        
        # Изменение размера (если включено в конфиге)
        if self.use_resize:
            img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
            # Маски ресайзим методом ближайшего соседа, чтобы не размывать границы
            mask_q = cv2.resize(mask_q, self.img_size, interpolation=cv2.INTER_NEAREST)
            mask_cos = cv2.resize(mask_cos, self.img_size, interpolation=cv2.INTER_NEAREST)
            mask_sin = cv2.resize(mask_sin, self.img_size, interpolation=cv2.INTER_NEAREST)
            mask_w = cv2.resize(mask_w, self.img_size, interpolation=cv2.INTER_NEAREST)

        # img = self.color_aug(img)
        # Нормализация изображения (0-1) и перевод к формату PyTorch (C, H, W)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        
        # Собираем маски в один тензор [4, H, W]
        masks = np.stack([mask_q, mask_cos, mask_sin, mask_w], axis=0)
        
        # Возвращаем тензоры
        return torch.from_numpy(img), torch.from_numpy(masks)
    
def get_dataloaders(config):
    """
    Создает датасет, делит его на Train/Val и возвращает два DataLoader'а.
    """
    full_dataset = JacquardDataset(config, is_train=True)
    
    # 90% на обучение, 10% на валидацию
    train_size = int(config["TRAIN_SPLIT"] * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])
    
    # Чтобы валидация не использовала аугментацию, мы технически должны 
    # оборачивать val_subset, но пока для простоты ColorJitter на валидации не критичен.
    
    train_loader = DataLoader(
        train_subset, 
        batch_size=config["BATCH_SIZE"], 
        shuffle=True, 
        num_workers=config["NUM_WORKERS"]
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=config["BATCH_SIZE"], 
        shuffle=False, 
        num_workers=config["NUM_WORKERS"]
    )
    
    return train_loader, val_loader