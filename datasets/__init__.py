"""Datasets package for fault diagnosis"""

from .BJTU_rao import dataset_save as BJTU_rao
from .XJTU_gearbox import dataset_save as XJTU_gearbox
from .XJTU_spurgear import dataset_save as XJTU_spurgear
from .OU_bearing import dataset_save as OU_bearing

__all__ = [
    'BJTU_rao',
    'XJTU_gearbox',
    'XJTU_spurgear',
    'OU_bearing',
]
