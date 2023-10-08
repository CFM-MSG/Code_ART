import json
import argparse
from time import time

from src.experiment import common_functions as cmf
from src.utils import io_utils


""" Get parameters """
def _get_argument_params():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", type=str, required=True,
                        help="Experiment or configuration name")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="filename of checkpoint.")
    parser.add_argument("--method", default="tgn_lgi",
                        help="Method type")
    parser.add_argument("--dataset", default="charades",
                        help="dataset to train models [charades|anet].")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="The number of workers for data loader.")
    parser.add_argument("--debug_mode" , action="store_true", default=False,
                        help="Run the script in debug mode")
    parser.add_argument('--seed', default=1, help='seed name', type=int)

    params = vars(parser.parse_args())
    print (json.dumps(params, indent=4))
    return params

def main(params):
    # s_time = time()
    config = io_utils.load_yaml(params["config"])
    cmf.set_seed(params)

    # prepare dataset
    D = cmf.get_dataset(params["dataset"])
    dsets, L = cmf.get_loader(D, split=["test"],
                              loader_configs=[config["test_loader"]],
                              num_workers=params["num_workers"])

    # Build network
    num_case = dsets['test'].num_instances
    print("************num_case")
    print(num_case)
    M = cmf.get_method(params["method"])
    net = M(config, logger=None)
    print("load from: ", params["checkpoint"])
    net.load_checkpoint(params["checkpoint"], True)
    if config["model"]["use_gpu"]: net.gpu_mode()

    # Evaluating networks
    # print(1)
    # print(L["test"].__len__())
    cmf.test(config, L["test"], net, -1, num_case, None, mode="Test")
    # e_time = time()
    # print('Time:' + str((e_time - s_time) / num_case))

if __name__ == "__main__":
    params = _get_argument_params()
    main(params)
