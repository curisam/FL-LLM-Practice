import argparse
import sys
from federatedscope.core.configs.config import global_cfg


def parse_args(args=None):
    parser = argparse.ArgumentParser(description='FederatedScope',
                                     add_help=False) #argparse.ArgumentParser의 형태
    parser.add_argument('--cfg',
                        dest='cfg_file',
                        help='Config file path',
                        required=False,
                        type=str)
    parser.add_argument('--client_cfg',
                        dest='client_cfg_file',
                        help='Config file path for clients',
                        required=False,
                        default=None,
                        type=str)#--client_cfg 의 값은 args.client_cfg_file 로 저장.
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser.add_argument(
        '--help',
        nargs="?",
        const="all",
        default="",
    )
    parser.add_argument('opts',
                        help='See federatedscope/core/configs for all options',
                        default=None,
                        nargs=argparse.REMAINDER)
    
   
    parse_res = parser.parse_args(args) #args 가 None 이면 실제 커맨드라인(sys.argv[1:]) 이 자동 사용됩니다. #ArgumentParser가 아닌  argparse.Namespace 객체
   
    
    init_cfg = global_cfg.clone()
    # when users type only "main.py" or "main.py help"
    if len(sys.argv) == 1 or parse_res.help == "all": #python main.py or python main.py --help로 실행-> # 전체 도움말 표시 후 종료
        parser.print_help()
        init_cfg.print_help()
        sys.exit(1)
    elif hasattr(parse_res, "help") and isinstance(
            parse_res.help, str) and parse_res.help != "": #python main.py --help key (단일 키) -> # 특정 키 도움말만 출력 후 종료
        init_cfg.print_help(parse_res.help)
        sys.exit(1)
    elif hasattr(parse_res, "help") and isinstance(
            parse_res.help, list) and len(parse_res.help) != 0: #python main.py --help key1 key2 … (다중 키) -> # 리스트를 순회하며 각 키에 대한 도움말 출력 후 종료
        for query in parse_res.help:
            init_cfg.print_help(query)
        sys.exit(1)

    return parse_res #global_cfg 반영안됨!!


def parse_client_cfg(arg_opts):
    """
    Arguments:
        arg_opts: list pairs of arg.opts
    """
    client_cfg_opts = []
    i = 0
    while i < len(arg_opts):
        if arg_opts[i].startswith('client'):#현재 토큰이 'client' 로 시작하면 예) 'client.lr', 'client_1.lr'
            #입력  arg_opts: ['--lr', '0.01', 'client_0.batch', '16', 'client_1.lr', '0.001']                                   
            #조건  'client'로 시작 O
            #pop(i) → 'client_0.batch'  -> client_cfg_opts
            #pop(i) → '16'              -> client_cfg_opts
            #리스트 길이 2 줄어듦

            client_cfg_opts.append(arg_opts.pop(i))
            client_cfg_opts.append(arg_opts.pop(i))
        else:
            i += 1

        #최종 반환:
        #arg_opts        = ['--lr', '0.01']
        #client_cfg_opts = ['client_0.batch', '16', 'client_1.lr', '0.001']
    return arg_opts, client_cfg_opts
