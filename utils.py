import logging
import os.path as osp


def use_logging(level='info'):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if level == 'warn':
                logging.warning("%s is running" % func.__name__)
            elif level == "info":
                logging.info("%s is running" % func.__name__)
            return func(*args)

        return wrapper

    return decorator


def get_model_log_dir(comment, model_name):
    import socket
    from datetime import datetime
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = osp.join(
        current_time + '_' + socket.gethostname() + '_' + comment + '_' + model_name)
    return log_dir


def to_cuda(data_list, device):
    for i, data in enumerate(data_list):
        for k, v in data:
            data[k] = v.to(device)
        data_list[i] = data
    return data_list
