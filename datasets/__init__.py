from .dtu import MVSDatasetDTU
from .blender import MVSDatasetBlender
from .llff import MVSDatasetRealFF
from .colmap import MVSDatasetCOLMAP
from .ibrnet import MVSDatasetIBRNet


datas_dict = {
    'dtu': MVSDatasetDTU,
    'blender': MVSDatasetBlender,
    'llff': MVSDatasetRealFF,
    'colmap': MVSDatasetCOLMAP,
    'ibrnet': MVSDatasetIBRNet,
}
