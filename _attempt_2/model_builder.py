import torch
import yaml
import os
from types import SimpleNamespace
from models.seg_hrnet import get_seg_model
from models_utils import GraspHead

# Специальный класс, который работает и как словарь, и как объект
class Map(dict):
    def __init__(self, *args, **kwargs):
        super(Map, self).__init__(*args, **kwargs)
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getattr__(self, attr):
        return self.get(attr)

    def __setattr__(self, key, value):
        self[key] = value

def build_grasp_hrnet(config, yaml_config_path, pretrained_path=None):
    if not os.path.exists(yaml_config_path):
        raise FileNotFoundError(f"Конфиг не найден: {yaml_config_path}")
        
    with open(yaml_config_path, 'r') as f:
        hrnet_cfg_dict = yaml.safe_load(f)
    
    # Создаем основной конфиг как Map (теперь всё будет доступно и через . и через [])
    hrnet_cfg_obj = Map(hrnet_cfg_dict)
    
    # Превращаем вложенные словари тоже в Map
    if 'MODEL' in hrnet_cfg_obj:
        hrnet_cfg_obj.MODEL = Map(hrnet_cfg_obj.MODEL)
        if 'EXTRA' in hrnet_cfg_obj.MODEL:
            hrnet_cfg_obj.MODEL.EXTRA = Map(hrnet_cfg_obj.MODEL.EXTRA)
    
    if 'DATASET' in hrnet_cfg_obj:
        hrnet_cfg_obj.DATASET = Map(hrnet_cfg_obj.DATASET)

    # --- ЗАПЛАТКИ ДЛЯ СОВМЕСТИМОСТИ (теперь через точки) ---
    if not hrnet_cfg_obj.MODEL.ALIGN_CORNERS:
        hrnet_cfg_obj.MODEL.ALIGN_CORNERS = True
        
    if not hrnet_cfg_obj.MODEL.PRETRAINED:
        hrnet_cfg_obj.MODEL.PRETRAINED = ""
        
    if not hrnet_cfg_obj.DATASET.NUM_CLASSES:
        hrnet_cfg_obj.DATASET.NUM_CLASSES = 1 
        
    if not hrnet_cfg_obj.MODEL.EXTRA.FINAL_CONV_KERNEL:
        hrnet_cfg_obj.MODEL.EXTRA.FINAL_CONV_KERNEL = 1
    # --------------------------

    print("Инициализация HRNet Backbone...")
    model = get_seg_model(hrnet_cfg_obj)
    
    # 3. Загружаем предобученные веса
    if pretrained_path and os.path.exists(pretrained_path):
        print(f"Загрузка предобученных весов: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        state_dict = {k: v for k, v in state_dict.items() if 'last_layer' not in k}
        model.load_state_dict(state_dict, strict=False)
        print("Загрузка весов завершена.")
    
    # 4. Заменяем голову
    print("Установка Grasp Head...")
    last_inp_channels = model.last_layer[0].in_channels
    model.last_layer = GraspHead(last_inp_channels, out_channels=4)
    
    return model