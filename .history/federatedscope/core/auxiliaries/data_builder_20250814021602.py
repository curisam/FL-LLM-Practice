import logging

from importlib import import_module

from federatedscope.core.data.utils import RegexInverseMap, load_dataset, \
    convert_data_mode  #import 하는 과정에서 상위의 federatedscope.core.data의 init 실행 중 from federatedscope.core.auxiliaries.splitter_builder import get_splitter 통해 resiter.data_dict generate 됨. 
from federatedscope.core.auxiliaries.utils import setup_seed

import federatedscope.register as register


logger = logging.getLogger(__name__)


try:
    from federatedscope.contrib.data import * #resiter.data_dict generate 됨!! (federatedscope.core.data.base_translator->federatedscope.core.auxiliaries.splitter_builder 실행)
except ImportError as error:
    logger.warning(
        f'{error} in `federatedscope.contrib.data`, some modules are not '
        f'available.')


# TODO: Add PyGNodeDataTranslator and PyGLinkDataTranslator
# TODO: move splitter to PyGNodeDataTranslator and PyGLinkDataTranslator
TRANS_DATA_MAP = {
    'BaseDataTranslator': [
        '.*?@.*?', 'hiv', 'proteins', 'imdb-binary', 'bbbp', 'tox21', 'bace',
        'sider', 'clintox', 'esol', 'freesolv', 'lipo', 'cifar4cl', 'cifar4lp'
    ],
    'DummyDataTranslator': [
        'toy', 'quadratic', 'femnist', 'celeba', 'shakespeare', 'twitter',
        'subreddit', 'synthetic', 'ciao', 'epinions', '.*?vertical_fl_data.*?',
        '.*?movielens.*?', '.*?netflix.*?', '.*?cikmcup.*?',
        'graph_multi_domain.*?', 'cora', 'citeseer', 'pubmed', 'dblp_conf',
        'dblp_org', 'csbm.*?', 'fb15k-237', 'wn18', 'adult', 'abalone',
        'credit', 'blog'
    ],  # Dummy for FL dataset
    'RawDataTranslator': ['hetero_nlp_tasks'],
}
DATA_TRANS_MAP = RegexInverseMap(TRANS_DATA_MAP, None)
"""DATA_TRANS_MAP._items ={
  '.*?@.*?': 'BaseDataTranslator',
  'hiv':       'BaseDataTranslator',
  'proteins':  'BaseDataTranslator',
  …,
  'toy':       'DummyDataTranslator',
  'femnist':   'DummyDataTranslator',
  'cora':      'DummyDataTranslator',
  …,
  'hetero_nlp_tasks': 'RawDataTranslator'
}"""


def get_data(config, client_cfgs=None):
    """Instantiate the data and update the configuration accordingly if
    necessary.

    Arguments:
        config: a cfg node object
        client_cfgs: dict of client-specific cfg node object
    Returns:
        The dataset object and the updated configuration.

    Note:
      The available ``data.type`` is shown below:
        ==================================  ===========================
        Data type                           Domain
        ==================================  ===========================
        FEMNIST	                            CV
        Celeba	                            CV
        ``${DNAME}@torchvision``	        CV
        Shakespeare	                        NLP
        SubReddit	                        NLP
        Twitter (Sentiment140)	            NLP
        ``${DNAME}@torchtext``	            NLP
        ``${DNAME}@huggingface_datasets``  	NLP
        Cora	                            Graph (node-level)
        CiteSeer	                        Graph (node-level)
        PubMed	                            Graph (node-level)
        DBLP_conf	                        Graph (node-level)
        DBLP_org	                        Graph (node-level)
        csbm	                            Graph (node-level)
        Epinions	                        Graph (link-level)
        Ciao	                            Graph (link-level)
        FB15k	                            Graph (link-level)
        FB15k-237	                        Graph (link-level)
        WN18	                            Graph (link-level)
        MUTAG	                            Graph (graph-level)
        BZR	                                Graph (graph-level)
        COX2	                            Graph (graph-level)
        DHFR	                            Graph (graph-level)
        PTC_MR	                            Graph (graph-level)
        AIDS	                            Graph (graph-level)
        NCI1	                            Graph (graph-level)
        ENZYMES	                            Graph (graph-level)
        DD	                                Graph (graph-level)
        PROTEINS	                        Graph (graph-level)
        COLLAB	                            Graph (graph-level)
        IMDB-BINARY	                        Graph (graph-level)
        IMDB-MULTI	                        Graph (graph-level)
        REDDIT-BINARY	                    Graph (graph-level)
        HIV	                                Graph (graph-level)
        ESOL	                            Graph (graph-level)
        FREESOLV	                        Graph (graph-level)
        LIPO	                            Graph (graph-level)
        PCBA	                            Graph (graph-level)
        MUV	                                Graph (graph-level)
        BACE	                            Graph (graph-level)
        BBBP	                            Graph (graph-level)
        TOX21	                            Graph (graph-level)
        TOXCAST	                            Graph (graph-level)
        SIDER	                            Graph (graph-level)
        CLINTOX	                            Graph (graph-level)
        graph_multi_domain_mol	            Graph (graph-level)
        graph_multi_domain_small	        Graph (graph-level)
        graph_multi_domain_biochem	        Graph (graph-level)
        cikmcup	                            Graph (graph-level)
        toy	                                Tabular
        synthetic	                        Tabular
        quadratic	                        Tabular
        ``${DNAME}openml``	                Tabular
        vertical_fl_data	                Tabular(vertical)
        VFLMovieLens1M	                    Recommendation
        VFLMovieLens10M	                    Recommendation
        HFLMovieLens1M	                    Recommendation
        HFLMovieLens10M	                    Recommendation
        VFLNetflix	                        Recommendation
        HFLNetflix	                        Recommendation
        ==================================  ===========================
    """
    # Fix the seed for data generation
    setup_seed(12345)



    for func in register.data_dict.values(): #func:"file", "mini-graph-dc"  둘다 해당하지 않음. data_and_config는 None이 된다.
        data_and_config = func(config, client_cfgs)
        if data_and_config is not None:
            return data_and_config

    # Load dataset from source files
    dataset, modified_config = load_dataset(config, client_cfgs) ##(train_dataset, val_dataset, test_dataset)의 형태. 각각은 LLMDataset의 형태.




    # Apply translator to non-FL dataset to transform it into its federated
    # counterpart
    if dataset is not None: #BaseDataTranslator로 됨.
        translator = getattr(import_module('federatedscope.core.data'),
                             DATA_TRANS_MAP[config.data.type.lower()])(
                                 modified_config, client_cfgs) ##BaseDataTranslator로 됨. init 실행
        data = translator(dataset) #BaseDataTranslator의 call 실행. 

        # Convert `StandaloneDataDict` to `ClientData` when in distribute mode


        data = convert_data_mode(data, modified_config) #pass
    else:
        data = None

    # Restore the user-specified seed after the data generation
    setup_seed(config.seed)

    return data, modified_config


    # get_data()가 돌려주는 data_dict
    """{
    0: ClientData(server_cfg, train=…, val=…, test=…),
    1: ClientData(client1_cfg, train=…, val=…, test=…),
    2: ClientData(client2_cfg, …),
    …,
    N: ClientData(clientN_cfg, …)
    }
    Key: 0은 서버, 1~N은 클라이언트

    Value: ClientData 인스턴스 (각자의 train/val/test 데이터 보관)

    이걸 다시 StandaloneDataDict로 감싸서,
    runner 쪽에 넘기면 “각 참가자 ID별 데이터”를 바로 꺼내 쓸 수 있게 되는 구조입니다.""" 