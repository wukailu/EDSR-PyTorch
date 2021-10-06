from .classification_meter import ClassificationMeter, MultiClassificationMeter
from .relation_meter import MeanMeterWithKey
from .utils import Meter, all_sum

meter_dict = {
    "ClassificationMeter": ClassificationMeter,
    "MultiClassificationMeter": MultiClassificationMeter,
    "MeanMeterWithKey": MeanMeterWithKey,
}


def get_meter(meter_type: str):
    return meter_dict[meter_type]
