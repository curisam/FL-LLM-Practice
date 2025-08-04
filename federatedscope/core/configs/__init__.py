import copy
from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py")) #federatedscope/core/configs/ 디렉터리 안의 모든 .py 파일 경로를 찾음. join(dirname(__file__)='/home/seongyoon/jupyter/FedBiscuit/federatedscope/core/configs'
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]#파일 이름에서 디렉터리와 .py 확장자를 떼어낸 리스트를 만듬. basename은 맨 마지막 부분만 남김.
#__all__=['cfg_differential_privacy', 'yacs_config', 'cfg_training', 'cfg_data', 'constants', 'config', 'cfg_asyn', 'cfg_aggregator', 'cfg_compression', 'cfg_llm', 'cfg_model', 'cfg_hpo', 'cfg_attack', 'cfg_fl_setting', 'cfg_evaluation', 'cfg_fl_algo']


# to ensure the sub-configs registered before set up the global config
all_sub_configs = copy.copy(__all__)
if "config" in all_sub_configs:
    all_sub_configs.remove('config')

#all_sub_configs=['cfg_differential_privacy', 'yacs_config', 'cfg_training', 'cfg_data', 'constants', 'config', 'cfg_asyn', 'cfg_aggregator', 'cfg_compression', 'cfg_llm', 'cfg_model', 'cfg_hpo', 'cfg_attack', 'cfg_fl_setting', 'cfg_evaluation', 'cfg_fl_algo']    

from federatedscope.core.configs.config import CN, init_global_cfg
__all__ = __all__ + \
          [
              'CN',
              'init_global_cfg'
          ] #2개 더 추가.


# reorder the config to ensure the base config will be registered first
base_configs = [
    'cfg_data', 'cfg_fl_setting', 'cfg_model', 'cfg_training', 'cfg_evaluation'
]
for base_config in base_configs:
    all_sub_configs.pop(all_sub_configs.index(base_config))
    all_sub_configs.insert(0, base_config)


#all_sub_configs=['cfg_evaluation', 'cfg_training', 'cfg_model', 'cfg_fl_setting', 'cfg_data', 'cfg_differential_privacy', 'yacs_config', 'constants', 'cfg_asyn', 'cfg_aggregator', 'cfg_compression', 'cfg_llm', 'cfg_hpo', 'cfg_attack', 'cfg_fl_algo']


