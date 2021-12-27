from enum import Enum
class Scalar(Enum):
    is_standard = 1
    is_minmax = 2
    
class EModel(Enum):
    isSVC = 1
    isMLP = 2
    
class Hyper:
    scalar_type = Scalar.is_standard
    model_type = EModel.isMLP
    is_correlation = True
    plot = True
    print_results=True
