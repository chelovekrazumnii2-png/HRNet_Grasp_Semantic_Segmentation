import torch
import torch.nn as nn

class GraspHead(nn.Module):
    def __init__(self, in_channels, out_channels=4):
        """
        Голова модели для задачи регрессии параметров захвата.
        :param in_channels: Количество входных каналов из HRNet (для W48 это обычно 720 на выходе после объединения всех веток)
        :param out_channels: 4 (Quality, Cos, Sin, Width)
        """
        super(GraspHead, self).__init__()
        
        # Финальная цепочка слоев: 
        # Сначала немного "сгущаем" признаки, затем выдаем финальные карты
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.conv_block(x)
        
        # Нам нужно ограничить значения некоторых карт:
        # 1-й канал (Quality) должен быть от 0 до 1 (вероятность)
        # Остальные каналы пока оставляем как есть, мы ограничим их через Loss или активации позже
        return x

def get_hrnet_model(config):
    """
    Функция-сборщик. Загружает backbone и приклеивает нашу голову.
    """
    # ВАЖНО: Здесь предполагается, что у вас скачан официальный репозиторий
    # и он находится в Python Path. 
    # Для примера используем упрощенную логику импорта:
    from HRNet_Semantic_Segmentation_HRNet_OCR.lib.models.seg_hrnet import get_seg_model
    import yaml
    
    # Нам нужен файл конфигурации .yaml от официального HRNet-W48
    # Обычно он лежит в репозитории в папке experiments/
    with open("hrnet_w48_config.yaml", 'r') as f:
        hrnet_config = yaml.safe_load(f)
        
    # Создаем базовую модель HRNet
    # Мы передаем hrnet_config как объект, к которому привыкла модель
    from types import SimpleNamespace
    hrnet_cfg_obj = SimpleNamespace(**hrnet_config)
    
    backbone = get_seg_model(hrnet_cfg_obj)
    
    # У HRNet-W48 финальное количество каналов после слияния всех веток (V2) 
    # обычно равно 720. Заменяем последний слой классификации на нашу голову.
    # В оригинальном коде это поле self.last_layer
    
    last_inp_channels = backbone.last_layer[0].in_channels
    backbone.last_layer = GraspHead(last_inp_channels, out_channels=4)
    
    return backbone