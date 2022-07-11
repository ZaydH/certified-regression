__all__ = [
    "parse_args",
]

from argparse import Namespace, ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path

from bound import config
import bound.dirs
from bound import logger
import bound.utils


def parse_args() -> Namespace:
    r""" Parse, checks, and refactors the input arguments"""
    args = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # noinspection PyTypeChecker
    args.add_argument("config_file", help="Path to the configuration file", type=Path)

    args.add_argument("-d", help="Debug mode -- Disable non-determinism", action="store_true")
    args.add_argument("-q", help="Enable quiet mode", action="store_true")
    args.add_argument("--deg", help="Number of spread degree", type=int, default=None)
    args.add_argument("--no_multi", help="Disables multicover mode", action="store_true")
    args.add_argument("--dist", help="Overrides the bound distance", type=str, default=None)
    args.add_argument("--greedy", help="Use greedy mode to calculate the bound",
                      action="store_true")
    args = args.parse_args()

    if not args.config_file.exists() or not args.config_file.is_file():
        raise ValueError(f"Unknown configuration file \"{args.config_file}\"")

    # Need to update directories first as other commands rely on these directories
    logger.setup()

    config.parse(args.config_file)
    if args.d:
        config.enable_debug_mode()
        bound.utils.set_debug_mode(seed=1)
    if args.deg:
        config.override_spread_degree(spread_degree=args.deg)
    if args.no_multi:
        config.disable_multicover_mode()
    if args.greedy:
        config.enable_greedy_mode()
    if args.dist:
        for type_func in (int, float):
            try:
                args.dist = type_func(args.dist)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Unable to decode bound distance \"{args.dist}\"")
        config.override_bound_dist(dist=args.dist)
    if args.q:
        config.set_quiet()

    # Configure the random seeds based on torch's seed
    bound.utils.set_random_seeds()
    # Generates the data for learning
    config.print_configuration()
    args.tg = bound.utils.configure_dataset_args()
    return args
