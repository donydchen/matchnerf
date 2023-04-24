import os
import sys
import time
import shutil
import datetime
import torch
import ipdb
import termcolor
import socket
import contextlib
from easydict import EasyDict as edict
from collections import OrderedDict
import numpy as np
import re
import skvideo.io
import cv2
from PIL import Image
import torchvision.transforms as T


# convert to colored strings
def red(message, **kwargs): return termcolor.colored(str(message), color="red", attrs=[k for k, v in kwargs.items() if v is True])
def green(message, **kwargs): return termcolor.colored(str(message), color="green", attrs=[k for k, v in kwargs.items() if v is True])
def blue(message, **kwargs): return termcolor.colored(str(message), color="blue", attrs=[k for k, v in kwargs.items() if v is True])
def cyan(message, **kwargs): return termcolor.colored(str(message), color="cyan", attrs=[k for k, v in kwargs.items() if v is True])
def yellow(message, **kwargs): return termcolor.colored(str(message), color="yellow", attrs=[k for k, v in kwargs.items() if v is True])
def magenta(message, **kwargs): return termcolor.colored(str(message), color="magenta", attrs=[k for k, v in kwargs.items() if v is True])
def grey(message, **kwargs): return termcolor.colored(str(message), color="grey", attrs=[k for k, v in kwargs.items() if v is True])


def get_time(sec):
    d = int(sec//(24*60*60))
    h = int(sec//(60*60) % 24)
    m = int((sec//60) % 60)
    s = int(sec % 60)
    return d, h, m, s


def add_datetime(func):
    def wrapper(*args, **kwargs):
        datetime_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(grey("[{}] ".format(datetime_str), bold=True), end="")
        return func(*args, **kwargs)
    return wrapper


def add_functionname(func):
    def wrapper(*args, **kwargs):
        print(grey("[{}] ".format(func.__name__), bold=True))
        return func(*args, **kwargs)
    return wrapper


def pre_post_actions(pre=None, post=None):
    def func_decorator(func):
        def wrapper(*args, **kwargs):
            if pre:
                pre()
            retval = func(*args, **kwargs)
            if post:
                post()
            return retval
        return wrapper
    return func_decorator


debug = ipdb.set_trace


class Log:
    def __init__(self): pass

    def process(self, pid):
        print(grey("Process ID: {}".format(pid), bold=True))

    def title(self, message):
        print(yellow(message, bold=True, underline=True))

    def info(self, message):
        print(magenta(message, bold=True))

    def warn(self, message):
        print(yellow(message, bold=False, underline=False))

    def options(self, opt, level=0):
        for key, value in sorted(opt.items()):
            if isinstance(value, (dict, edict)):
                print("   "*level+cyan("* ")+green(key)+":")
                self.options(value, level+1)
            else:
                print("   "*level+cyan("* ")+green(key)+":", yellow(value))

    def loss_train(self, opt, ep, lr_dict, loss, timer):
        if not opt.max_epoch:
            return
        message = grey("[train] ", bold=True)
        message += "epoch {}/{}".format(cyan(ep, bold=True), opt.max_epoch)
        for k, v in lr_dict.items():
            message += ", lr_{}:{}".format(k, yellow("{:.2e}".format(v), bold=True))
        message += ", loss:{}".format(red("{:.3e}".format(loss), bold=True))
        message += ", time:{}".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.elapsed)), bold=True))
        message += " (ETA:{})".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.arrival))))
        print(message)

    def loss_val(self, loss):
        message = grey("[val] ", bold=True)
        message += "loss:{}".format(red("{:.3e}".format(loss), bold=True))
        print(message)

    def metric_test(self, metric):
        message = grey("[test] ", bold=True)
        message += f"{blue(metric, bold=True)}"
        print(message)


log = Log()


def update_timer(opt, timer, ep, it_per_ep):
    if not opt.max_epoch:
        return
    momentum = 0.99
    timer.elapsed = time.time()-timer.start
    timer.it = timer.it_end-timer.it_start
    # compute speed with moving average
    timer.it_mean = timer.it_mean*momentum+timer.it*(1-momentum) if timer.it_mean is not None else timer.it
    timer.arrival = timer.it_mean*it_per_ep*(opt.max_epoch-ep)

# move tensors to device in-place


def move_to_device(X, device):
    if isinstance(X, dict):
        for k, v in X.items():
            X[k] = move_to_device(v, device)
    elif isinstance(X, list):
        for i, e in enumerate(X):
            X[i] = move_to_device(e, device)
    elif isinstance(X, tuple) and hasattr(X, "_fields"):  # collections.namedtuple
        dd = X._asdict()
        dd = move_to_device(dd, device)
        return type(X)(**dd)
    elif isinstance(X, torch.Tensor):
        return X.to(device=device)
    return X


def to_dict(D, dict_type=dict):
    D = dict_type(D)
    for k, v in D.items():
        if isinstance(v, dict):
            D[k] = to_dict(v, dict_type)
    return D


def get_child_state_dict(state_dict, key):
    return {".".join(k.split(".")[1:]): v for k, v in state_dict.items() if k.startswith("{}.".format(key))}


def load_gmflow_checkpoint(model_enc, ckpt_path, device, gmflow_n_blocks=6):
    gmflow_ckpt = torch.load(ckpt_path, map_location=device)
    gmflow_weights = gmflow_ckpt['model'] if 'model' in gmflow_ckpt else gmflow_ckpt
    gmflow_weights_updated = OrderedDict()
    for k, v in gmflow_weights.items():
        keep_key = True
        for idx in range(gmflow_n_blocks, 6):
            if k.startswith("transformer.layers.%d" % idx):
                keep_key = False
                break
        if k.startswith('upsampler'):  # remove the gmflow upsampler
            keep_key = False
        if k.startswith('feature_flow_attn'):  # do not need the refine self-attention anymore
            keep_key = False
        if keep_key:
            gmflow_weights_updated[k] = v

    for name, child in model_enc.named_children():
        if name != "featup_net":  # our upsample is different from gmflow's
            child_state_dict = get_child_state_dict(gmflow_weights_updated, name)
            child.load_state_dict(child_state_dict, strict=True)


def restore_checkpoint(model, ckpt_path, device, resume=False, log=None, optims_scheds=None):
    checkpoint = torch.load(ckpt_path, map_location=device)

    for name, child in model.named_children():
        child_state_dict = get_child_state_dict(checkpoint["model"], name)
        if child_state_dict:
            child.load_state_dict(child_state_dict, strict=True)
            log.info(f"  * restored {name} from {ckpt_path}")

    if resume:
        ep, it = checkpoint["epoch"], checkpoint["iter"]
        assert optims_scheds is not None, "Must provide full optims (and / or scheds) for resume training."
        for name, method in optims_scheds.items():
            if name not in checkpoint:
                log.warn(f"  * can NOT find {name} in checkpoint, skip.")
                continue
            method.load_state_dict(checkpoint[name])
            log.info(f"  * restored {name} from {ckpt_path}")
        log.info("resuming from epoch {0} (iteration {1})".format(ep, it))
    else:
        ep, it = None, None

    return ep, it


def save_checkpoint(saved_dir, checkpoint, ep, it, backup_ckpt=True, children=None):
    ckpt_dir = os.path.join(saved_dir, 'models')
    os.makedirs(ckpt_dir, exist_ok=True)
    if children is not None:
        model_state_dict = {k: v for k, v in checkpoint['model'].items() if k.startswith(children)}
    else:
        model_state_dict = checkpoint['model']
    checkpoint.update(dict(epoch=ep, iter=it, model=model_state_dict))
    torch.save(checkpoint, os.path.join(ckpt_dir, 'latest.pth'))
    if backup_ckpt:
        # backup only the model info, not sched nor optim
        for rm_key in ['optim', 'sched']:
            if rm_key in checkpoint:
                checkpoint.pop(rm_key)
        torch.save(checkpoint, os.path.join(ckpt_dir, f'ep{ep}_it{it}.pth'))


def check_socket_open(hostname, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_open = False
    try:
        s.bind((hostname, port))
    except socket.error:
        is_open = True
    finally:
        s.close()
    return is_open


def get_layer_dims(layers):
    # return a list of tuples (k_in,k_out)
    return list(zip(layers[:-1], layers[1:]))


@contextlib.contextmanager
def suppress(stdout=False, stderr=False):
    with open(os.devnull, "w") as devnull:
        if stdout:
            old_stdout, sys.stdout = sys.stdout, devnull
        if stderr:
            old_stderr, sys.stderr = sys.stderr, devnull
        try:
            yield
        finally:
            if stdout:
                sys.stdout = old_stdout
            if stderr:
                sys.stderr = old_stderr


def colorcode_to_number(code):
    ords = [ord(c) for c in code[1:]]
    ords = [n-48 if n < 58 else n-87 for n in ords]
    rgb = (ords[0]*16+ords[1], ords[2]*16+ords[3], ords[4]*16+ords[5])
    return rgb


def list_all_images(root_dir):
    image_extensions = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
        '.tif', '.TIF', '.tiff', '.TIFF',
    ]
    all_images = []
    for filename in os.listdir(root_dir):
        if any(filename.endswith(ext) for ext in image_extensions):
            all_images.append(filename)
    return sorted(all_images)


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def write_video(out_path, frames, pts_rates=2.0):
    writer = skvideo.io.FFmpegWriter(out_path, outputdict={'-pix_fmt': 'yuv420p', '-crf': '21', '-vf': f'setpts={pts_rates}*PTS'})
    for frame in frames:
        writer.writeFrame(frame)
    writer.close()


def visualize_depth(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    if type(depth) is not np.ndarray:
        depth = depth.cpu().numpy()

    x = np.nan_to_num(depth)  # change nan to 0
    if minmax is None:
        mi = np.min(x[x > 0])  # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi, ma = minmax

    x = (x-mi)/(ma-mi+1e-8)  # normalize to 0~1
    x = (255*x).astype(np.uint8)
    im_cv2 = cv2.applyColorMap(x, cmap)
    x_ = Image.fromarray(cv2.cvtColor(im_cv2, cv2.COLOR_BGR2RGB))
    x_ = T.ToTensor()(x_)  # (3, H, W)
    return x_, [mi, ma]
