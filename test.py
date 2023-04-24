import os
import options
import sys
import torch

from coach import Coach
from misc.utils import log


def main():
    log.process(os.getpid())
    log.title("[{}] (PyTorch code for testing MatchNeRF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)

    with torch.cuda.device(opt.device):
        m = Coach(opt)

        m.build_networks()
        m.restore_checkpoint()

        m.load_dataset(splits=['test'])

        if opt.nerf.render_video:
            m.test_model_video(leave_tqdm=True)
        else:
            m.test_model(leave_tqdm=True)


if __name__ == "__main__":
    main()
