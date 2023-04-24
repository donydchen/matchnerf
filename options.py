import numpy as np
import os
import torch
import random
import string
import yaml
from easydict import EasyDict as edict
import time
import sys

from misc import utils
from misc.utils import log

# torch.backends.cudnn.enabled = False
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


def parse_arguments(args):
    """
    Parse arguments from command line.
    Syntax: --key1.key2.key3=value --> value
            --key1.key2.key3=      --> None
            --key1.key2.key3       --> True
            --key1.key2.key3!      --> False
            --key1.key2.key3=value, --> list
    """
    opt_cmd = {}
    for arg in args:
        assert(arg.startswith("--"))
        if "=" not in arg[2:]:
            key_str, value = (arg[2:-1], "false") if arg[-1] == "!" else (arg[2:], "true")
        else:
            key_str, value = arg[2:].split("=")
        keys_sub = key_str.split(".")
        opt_sub = opt_cmd
        for k in keys_sub[:-1]:
            if k not in opt_sub:
                opt_sub[k] = {}
            opt_sub = opt_sub[k]
        assert keys_sub[-1] not in opt_sub, keys_sub[-1]
        loaded_value = yaml.safe_load(value)
        if isinstance(loaded_value, str) and ',' in loaded_value:
            loaded_value = [int(x) if x.isdigit() else x for x in loaded_value.split(',') if x.strip()]
        opt_sub[keys_sub[-1]] = loaded_value
    opt_cmd = edict(opt_cmd)
    return opt_cmd


def set(opt_cmd={}):
    log.info("setting configurations...")
    # load config from yaml file
    assert("yaml" in opt_cmd)
    fname = "configs/{}.yaml".format(opt_cmd.yaml)
    opt_base = load_options(fname)
    # override with command line arguments
    opt = override_options(opt_base, opt_cmd, key_stack=[], safe_check=True)
    process_options(opt)
    log.options(opt)
    return opt


def load_options(fname):
    with open(fname) as file:
        opt = edict(yaml.safe_load(file))
    if "_parent_" in opt:
        # load parent yaml file(s) as base options
        parent_fnames = opt.pop("_parent_")
        if type(parent_fnames) is str:
            parent_fnames = [parent_fnames]
        for parent_fname in parent_fnames:
            opt_parent = load_options(parent_fname)
            opt_parent = override_options(opt_parent, opt, key_stack=[])
            opt = opt_parent
    print("loading {}...".format(fname))
    return opt


def override_options(opt, opt_over, key_stack=None, safe_check=False):
    for key, value in opt_over.items():
        if isinstance(value, dict):
            # parse child options (until leaf nodes are reached)
            opt[key] = override_options(opt.get(key, dict()), value, key_stack=key_stack+[key], safe_check=safe_check)
        else:
            # ensure command line argument to override is also in yaml file
            if safe_check and key not in opt:
                add_new = None
                while add_new not in ["y", "n"]:
                    key_str = ".".join(key_stack+[key])
                    add_new = input("\"{}\" not found in original opt, add? (y/n) ".format(key_str))
                if add_new == "n":
                    print("safe exiting...")
                    exit()
            opt[key] = value
    return opt


def process_options(opt):
    if opt.name is None:
        opt.name = time.strftime("%b%d_%H%M%S").lower()
    if isinstance(getattr(opt, "gpu_ids"), int):
        opt.gpu_ids = [opt.gpu_ids]

    if '_debug' in opt.name:
        if hasattr(opt, 'data_train'):
            opt.data_train.max_len = 20
        if hasattr(opt, 'data_val'):
            opt.data_val.max_len = 1
        if hasattr(opt, 'data_test'):
            for x in opt.data_test:
                opt.data_test[x].max_len = 1
        opt.max_epoch = 2

    # set seed
    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        if opt.seed != 0:
            opt.name = str(opt.name)+"_seed{}".format(opt.seed)
    else:
        # create random string as run ID
        randkey = "".join(random.choice(string.ascii_uppercase) for _ in range(4))
        opt.name = str(opt.name)+"_{}".format(randkey)
    # other default options
    opt.output_path = os.path.join(opt.output_root, opt.name)
    os.makedirs(opt.output_path, exist_ok=True)
    # save run commands
    with open(os.path.join(opt.output_path, 'run.bash'), 'a+') as f:
        command = 'python %s\n' % (' '.join(sys.argv))
        f.write(command)
    opt.device = "cpu" if opt.cpu or not torch.cuda.is_available() else "cuda:{}".format(opt.gpu_ids[0])


def save_options_file(opt):
    opt_fname = "{}/options.yaml".format(opt.output_path)
    if os.path.isfile(opt_fname):
        with open(opt_fname) as file:
            opt_old = yaml.safe_load(file)
        if opt != opt_old:
            # prompt if options are not identical
            opt_new_fname = "{}/options_temp.yaml".format(opt.output_path)
            with open(opt_new_fname, "w") as file:
                yaml.safe_dump(utils.to_dict(opt), file, default_flow_style=False, indent=4)
            print("existing options file found (different from current one)...")
            os.system("diff {} {}".format(opt_fname, opt_new_fname))
            os.system("rm {}".format(opt_new_fname))
            override = None
            while override not in ["y", "n"]:
                override = input("override? (y/n) ")
            if override == "n":
                print("safe exiting...")
                exit()
        else:
            print("existing options file found (identical)")
    else:
        print("(creating new options file...)")
    with open(opt_fname, "w") as file:
        yaml.safe_dump(utils.to_dict(opt), file, default_flow_style=False, indent=4)
