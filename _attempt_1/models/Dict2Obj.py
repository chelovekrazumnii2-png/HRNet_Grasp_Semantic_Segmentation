import torch
import torch.nn as nn
import torch.nn.functional as F

# Импортируем функцию сборки модели из скачанного файла
# Если файл лежит в папке, используй: from folder_name.seg_hrnet import get_seg_model
from models.seg_hrnet import get_seg_model

class Dict2Obj(dict):
    """
    Ультимативный конфиг: работает и как объект (cfg.MODEL), 
    и как словарь (cfg['MODEL']), поддерживая вложенность.
    """
    def __init__(self, d):
        super().__init__()
        for k, v in d.items():
            if isinstance(v, dict):
                v = Dict2Obj(v)
            self[k] = v

    def __getattr__(self, name):
        if name in self:
            return self[name]
        raise AttributeError(f"Key '{name}' not found in config")

    def __setattr__(self, name, value):
        self[name] = value