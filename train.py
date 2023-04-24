import torch
import sys
import os

from coach import Coach
from misc.utils import log
import options


def main():
    log.process(os.getpid())
    log.title("[{}] (PyTorch code for training MatchNeRF)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)

    with torch.cuda.device(opt.device):
        m = Coach(opt)

        # setup model
        m.build_networks()

        # setup dataset
        m.load_dataset(splits=['train', 'val', 'test'])

        # setup trianing utils
        m.setup_visualizer()
        m.setup_optimizer()

        if opt.resume or opt.load is not None:
            m.restore_checkpoint()

        m.train_model()


if __name__ == "__main__":
    main()
