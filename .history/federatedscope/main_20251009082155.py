import os
import sys

import argparse
import copy
import time
import yaml
import torch


import logging  # ← 추가

from federatedscope.core.auxiliaries.logging import update_logger


sys.setrecursionlimit(100000)
from federatedscope.core.cmd_args import parse_args, parse_client_cfg
from federatedscope.core.auxiliaries.data_builder import get_data

from federatedscope.core.auxiliaries.utils import setup_seed
from federatedscope.core.auxiliaries.logging import update_logger
from federatedscope.core.auxiliaries.worker_builder import get_client_cls, \
    get_server_cls


from federatedscope.core.configs.config import global_cfg, CfgNode
from federatedscope.core.auxiliaries.runner_builder import get_runner 


import torch, torchvision, torchaudio, importlib.metadata as m

from federatedscope.llm.misc.cluster_schedule import compute_round_schedule

if __name__ == '__main__':

    init_cfg = global_cfg.clone()
    args = parse_args() #argparser 정의
    if args.cfg_file: #argparser 덮어씌우기.
        init_cfg.merge_from_file(args.cfg_file)
    cfg_opt, client_cfg_opt = parse_client_cfg(args.opts)#arg.opts를 partition한다.
    init_cfg.merge_from_list(cfg_opt)

    # ✅ outdir은 YAML에서 최종 경로로 직접 주는 것을 권장:
    #    예) outdir: exp/tldr/choice_qwen/fedbis   ,  expname: ""
    #    (expname로 합치는 로직은 일단 제거하는게 안전)
    os.makedirs(init_cfg.outdir, exist_ok=True)

    # ✅ 로거 초기화는 최우선
    # cfg 로딩/머지 끝난 직후, outdir 확정한 다음에
    logger = update_logger(init_cfg, clear_before_add=True)  # ← 기존 호출 유지 가능
    logger.info(f"[main] outdir={init_cfg.outdir}")         # ← 이렇게 찍기


    setup_seed(init_cfg.seed) 


    # load clients' cfg file
    if args.client_cfg_file:
        client_cfgs = CfgNode.load_cfg(open(args.client_cfg_file, 'r'))
        # client_cfgs.set_new_allowed(True)
        client_cfgs.merge_from_list(client_cfg_opt)
    else:
        client_cfgs = None

    # federated dataset might change the number of clients
    # thus, we allow the creation procedure of dataset to modify the global
    # cfg object

    #data는 (train, val, test) dataset. modified_cfg는 cfg로부터 바뀌지 않음.


    data, modified_cfg = get_data(config=init_cfg.clone(),
                                  client_cfgs=client_cfgs)
    
    """data={
    0: ClientData(server_cfg, train=…, val=…, test=…),
    1: ClientData(client1_cfg, train=…, val=…, test=…),
    2: ClientData(client2_cfg, …),
    …,
    N: ClientData(clientN_cfg, …)
    }
    Key: 0은 서버, 1~N은 클라이언트

    Value: ClientData 인스턴스 (각자의 train/val/test 데이터 보관)

    runner 쪽에 넘기면 “각 참가자 ID별 데이터”를 바로 꺼내 쓸 수 있게 되는 구조입니다."""

    init_cfg.merge_from_other_cfg(modified_cfg)

    if init_cfg.federate.client_idx_for_local_train != 0: #FALSE
        init_cfg.federate.client_num = 1
        new_data = {0: data[0]} if 0 in data.keys() else dict()
        new_data[1] = data[init_cfg.federate.client_idx_for_local_train]
        data = new_data


    if getattr(init_cfg.federate, 'sampler', 'uniform') == 'cluster' and int(init_cfg.llm.adapter.count) > 1:
        init_cfg = compute_round_schedule(init_cfg)
        print("[cluster] clusters len:", len(init_cfg.llm.adapter.clusters))
        print("[cluster] s_per       :", init_cfg.llm.adapter.sample_num_per_adapter)


    runner = get_runner(data=data,
                        server_class=get_server_cls(init_cfg),
                        client_class=get_client_cls(init_cfg),
                        config=init_cfg.clone(),
                        client_configs=client_cfgs)
    _ = runner.run()