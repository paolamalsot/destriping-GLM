from src.destriping.GLUM.custom_regressors import (
    iterative_theta_after_cv_regressor,
    iterative_theta_regressor,
    warm_start_wrapper,
)
from src.destriping.GLUM import cv, fit
import logging

glum_logger_list = [
    iterative_theta_after_cv_regressor.logger,
    iterative_theta_regressor.logger,
    warm_start_wrapper.logger,
    cv.logger,
    fit.logger,
]


def set_level_all_glum_loggers(level=logging.INFO):
    for logger in glum_logger_list:
        logger.setLevel(level)
