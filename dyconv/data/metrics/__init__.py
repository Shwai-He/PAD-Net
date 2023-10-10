from .imagenet_evaluator import ImageNetEvaluator
from .multiclass_evaluator import MultiClsEvaluator


# def build_evaluator(cfg):
#     evaluator = {
#         'custom': CustomEvaluator,
#         'imagenet': ImageNetEvaluator,
#         'multiclass': MultiClsEvaluator,
#     }[cfg['type']]
#     return evaluator(**cfg['kwargs'])
