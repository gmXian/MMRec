# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import os
import argparse
import yaml
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--config_file', type=str, default=None, help='extra config yaml file')
    parser.add_argument('--eval_attack', action='store_true', help='enable attack evaluation')

    config_dict = {
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()

    if args.config_file and os.path.isfile(args.config_file):
        with open(args.config_file, 'r', encoding='utf-8') as f:
            file_cfg = yaml.safe_load(f) or {}
        config_dict.update(file_cfg)

    if args.eval_attack:
        attack_cfg = config_dict.get('attack', {})
        attack_cfg['enable'] = True
        config_dict['attack'] = attack_cfg

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)

