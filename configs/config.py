# this file is modified from tandem https://github.com/tum-vision/tandem
import argparse
from ast import literal_eval
from os.path import dirname, join
from typing import Tuple

import yaml

DEFAULT_CONFIG_FILE = join(dirname(__file__), "default.yaml")


def _parse_dict(d, d_out=None, prefix=""):
    if d is None:
        return {}
    d_out = d_out if d_out is not None else {}
    for k, v in d.items():
        if isinstance(v, dict):
            _parse_dict(v, d_out, prefix=prefix + k + ".")
        else:
            if isinstance(v, str):
                try:
                    v = literal_eval(v)  # try to parse
                except (ValueError, SyntaxError):
                    pass  # v is really a string

            if isinstance(v, list):
                v = tuple(v)
            d_out[prefix + k] = v
    if prefix == "":
        return d_out


def load(fname):
    with open(fname, "r") as fp:
        return _parse_dict(yaml.safe_load(fp))


def merge_from_config(config, config_merge):
    for k, v in config_merge.items():
        # assert k in config, f"The key {k} is not in the base config for the merge."
        # if not k in config:
        # print("Add new args {} to config".format(k))
        config[k] = v


def merge_from_file(config, fname):
    merge_from_config(config, load(fname))


def merge_from_list(config, list_merge):
    assert len(list_merge) % 2 == 0, "The list must have key value pairs."
    config_merge = _parse_dict(dict(zip(list_merge[0::2], list_merge[1::2])))
    merge_from_config(config, config_merge)


def default():
    return load(DEFAULT_CONFIG_FILE)


def parse_args(
    parser: argparse.ArgumentParser,
) -> Tuple[str, dict, str, argparse.Namespace]:
    args = parser.parse_args()
    config = default()
    config_path = args.config
    if config_path is not None:
        merge_from_file(config, config_path)
    if args.opts is not None:
        merge_from_list(config, args.opts)
    args_dict = args.__dict__
    for k, v in args_dict.items():
        if not k in config:
            config[k] = v
    return config


def get_from_path(config_path):
    config = default()
    if config_path is not None:
        merge_from_file(config, config_path)

    return config


def save_yaml(config, file_name):
    hierarchical_dict = dict()
    for k, v in config.items():
        splited_keys = k.split(".")
        current_dict = hierarchical_dict
        last_key = splited_keys.pop()
        for key in splited_keys:
            if key in current_dict.keys():
                current_dict = current_dict[key]
            else:
                current_dict[key] = {}
                current_dict = current_dict[key]
        current_dict[last_key] = v
    with open(file_name, "w") as fp:
        yaml.safe_dump(hierarchical_dict, fp)
