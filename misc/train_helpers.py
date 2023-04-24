import torch
import os
import numpy as np
from collections import OrderedDict


@torch.no_grad()
def summarize_metrics(metrics, out_dir, it=None, ep=None):
    head_info = ""
    if it is not None:
        head_info = f" at Iteration [{it}]"
    if ep is not None:
        head_info = f" at Epoch [{ep}]"

    dataset_metrics = {}
    for dataname, raw_metrics in metrics.items():
        dataset_metrics[dataname] = {}
        header = f"------------ {dataname.upper()} Nearest 3{head_info} ------------"
        all_msgs = [header]
        cur_scene = ""
        for view_id, view_metrics in raw_metrics.items():
            if view_id.split('_')[0] != cur_scene:
                if cur_scene != "":  # summarise scene buffer and log
                    scene_info = f"====> scene: {cur_scene},"
                    for k, v in scene_metrics.items():
                        scene_info = scene_info + f" {k}: {float(np.array(v).mean())},"
                    all_msgs.append(scene_info)
                else:  # init dataset
                    dataset_metrics[dataname] = OrderedDict({k: [] for k in view_metrics.keys()})
                # reset scene buffer
                cur_scene = view_id.split('_')[0]
                scene_metrics = {k: [] for k in view_metrics.keys()}
            # log view
            view_info = f"==> view: {view_id},"
            for k, v in view_metrics.items():
                view_info = view_info + f" {k}: {float(v)},"
                scene_metrics[k].append(v)
                dataset_metrics[dataname][k].append(v)
            all_msgs.append(view_info)
        # summarise dataset
        data_info = f"======> {dataname.upper()}{head_info},"
        for k, v in dataset_metrics[dataname].items():
            data_info = data_info + f" {k}: {float(np.array(v).mean())},"
        all_msgs.append(data_info)
        with open(os.path.join(out_dir, f"0results_{dataname}.txt"), "a+") as f:
            f.write('\n'.join(all_msgs))
            f.write('\n')
    return dataset_metrics


def summarize_loss(loss, loss_weight):
    loss_all = 0.
    assert("all" not in loss)
    # weigh losses
    for key in loss:
        assert(key in loss_weight)
        assert(loss[key].shape == ())
        if loss_weight[key] is not None:
            assert not torch.isinf(loss[key]), "loss {} is Inf".format(key)
            assert not torch.isnan(loss[key]), "loss {} is NaN".format(key)
            loss_all = loss_all + float(loss_weight[key]) * loss[key]
    loss.update(all=loss_all)
    return loss


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
