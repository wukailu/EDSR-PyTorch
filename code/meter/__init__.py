from .classification_meter import ClassificationMeter, MultiClassificationMeter
from .relation_meter import MeanMeterWithKey
from .super_resolution_meter import SuperResolutionMeter
from .utils import Meter, all_sum

meter_dict = {
    "ClassificationMeter": ClassificationMeter,
    "MultiClassificationMeter": MultiClassificationMeter,
    "MeanMeterWithKey": MeanMeterWithKey,
    "SuperResolutionMeter": SuperResolutionMeter,
}


def get_meter(meter_type: str):
    return meter_dict[meter_type]
