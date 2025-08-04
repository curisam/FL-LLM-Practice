# import copy
# import json
# import logging
# import os
# import re
# import time
# import yaml

# import numpy as np
# from datetime import datetime

# logger = logging.getLogger(__name__)


# class CustomFormatter(logging.Formatter):
#     """Logging colored formatter, adapted from
#     https://stackoverflow.com/a/56944256/3638629"""
#     def __init__(self, fmt):
#         super().__init__()
#         grey = '\x1b[38;21m'
#         blue = '\x1b[38;5;39m'
#         yellow = "\x1b[33;20m"
#         red = '\x1b[38;5;196m'
#         bold_red = '\x1b[31;1m'
#         reset = '\x1b[0m'

#         self.FORMATS = {
#             logging.DEBUG: grey + fmt + reset,
#             logging.INFO: blue + fmt + reset,
#             logging.WARNING: yellow + fmt + reset,
#             logging.ERROR: red + fmt + reset,
#             logging.CRITICAL: bold_red + fmt + reset
#         }

#     def format(self, record):
#         log_fmt = self.FORMATS.get(record.levelno)
#         formatter = logging.Formatter(log_fmt)
#         return formatter.format(record)


# class LoggerPrecisionFilter(logging.Filter):
#     def __init__(self, precision):
#         super().__init__()
#         self.print_precision = precision

#     def str_round(self, match_res):
#         return str(round(eval(match_res.group()), self.print_precision))

#     def filter(self, record):
#         # use regex to find float numbers and round them to specified precision
#         if not isinstance(record.msg, str):
#             record.msg = str(record.msg)
#         if record.msg != "":
#             if re.search(r"([-+]?\d+\.\d+)", record.msg):
#                 record.msg = re.sub(r"([-+]?\d+\.\d+)", self.str_round,
#                                     record.msg)
#         return True


# def update_logger(cfg, clear_before_add=False, rank=0):
#     root_logger = logging.getLogger("federatedscope")

#     # clear all existing handlers and add the default stream
#     if clear_before_add:
#         root_logger.handlers = []
#         handler = logging.StreamHandler()
#         fmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
#         handler.setFormatter(CustomFormatter(fmt))

#         root_logger.addHandler(handler)

#     # update level
#     if rank == 0:
#         if cfg.verbose > 0:
#             logging_level = logging.INFO
#         else:
#             logging_level = logging.WARN
#             root_logger.warning("Skip DEBUG/INFO messages")
#     else:
#         root_logger.warning(f"Using deepspeed, and we will disable "
#                             f"subprocesses {rank} logger.")
#         logging_level = logging.CRITICAL
#     root_logger.setLevel(logging_level)

#     # ================ create outdir to save log, exp_config, models, etc,.
#     if cfg.outdir == "":
#         cfg.outdir = os.path.join(os.getcwd(), "exp")
#     if cfg.expname == "":
#         cfg.expname = f"{cfg.federate.method}_{cfg.model.type}_on" \
#                       f"_{cfg.data.type}_lr{cfg.train.optimizer.lr}_lste" \
#                       f"p{cfg.train.local_update_steps}"
#     if cfg.expname_tag:
#         cfg.expname = f"{cfg.expname}_{cfg.expname_tag}"
#     cfg.outdir = os.path.join(cfg.outdir, cfg.expname)

#     if rank != 0:
#         return

#     # if exist, make directory with given name and time
#     if os.path.isdir(cfg.outdir) and os.path.exists(cfg.outdir):
#         outdir = os.path.join(cfg.outdir, "sub_exp" +
#                               datetime.now().strftime('_%Y%m%d%H%M%S')
#                               )  # e.g., sub_exp_20220411030524
#         while os.path.exists(outdir):
#             time.sleep(1)
#             outdir = os.path.join(
#                 cfg.outdir,
#                 "sub_exp" + datetime.now().strftime('_%Y%m%d%H%M%S'))
#         cfg.outdir = outdir
#     # if not, make directory with given name
#     # os.makedirs(cfg.outdir)
#     os.makedirs(cfg.outdir, exist_ok=True)

#     # create file handler which logs even debug messages
#     fh = logging.FileHandler(os.path.join(cfg.outdir, 'exp_print.log'))
#     fh.setLevel(logging.DEBUG)
#     logger_formatter = logging.Formatter(
#         "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
#     fh.setFormatter(logger_formatter)
#     root_logger.addHandler(fh)

#     # set print precision for terse logging
#     np.set_printoptions(precision=cfg.print_decimal_digits)
#     precision_filter = LoggerPrecisionFilter(cfg.print_decimal_digits)
#     # attach the filter to the fh handler to propagate the filter, since
#     # "Filters, unlike levels and handlers, do not propagate",
#     # ref https://stackoverflow.com/questions/6850798/why-doesnt-filter-
#     # attached-to-the-root-logger-propagate-to-descendant-loggers
#     for handler in root_logger.handlers:
#         handler.addFilter(precision_filter)

#     import socket
#     root_logger.info(f"the current machine is at"
#                      f" {socket.gethostbyname(socket.gethostname())}")
#     root_logger.info(f"the current dir is {os.getcwd()}")
#     root_logger.info(f"the output dir is {cfg.outdir}")

#     if cfg.wandb.use:
#         import sys
#         sys.stderr = sys.stdout  # make both stderr and stdout sent to wandb
#         # server
#         init_wandb(cfg)


# def init_wandb(cfg):
#     try:
#         import wandb
#         # on some linux machines, we may need "thread" init to avoid memory
#         # leakage
#         os.environ["WANDB_START_METHOD"] = "thread"
#     except ImportError:
#         logger.error("cfg.wandb.use=True but not install the wandb package")
#         exit()
#     dataset_name = cfg.data.type
#     method_name = cfg.federate.method
#     exp_name = cfg.expname

#     tmp_cfg = copy.deepcopy(cfg)
#     if tmp_cfg.is_frozen():
#         tmp_cfg.defrost()
#     tmp_cfg.clear_aux_info(
#     )  # in most cases, no need to save the cfg_check_funcs via wandb
#     tmp_cfg.de_arguments()
#     cfg_yaml = yaml.safe_load(tmp_cfg.dump())

#     wandb.init(project=cfg.wandb.name_project,
#                entity=cfg.wandb.name_user,
#                config=cfg_yaml,
#                group=dataset_name,
#                job_type=method_name,
#                name=exp_name,
#                notes=f"{method_name}, {exp_name}")


# def logfile_2_wandb_dict(exp_log_f, raw_out=True):
#     """
#         parse the logfiles [exp_print.log, eval_results.log] into
#         wandb_dict that contains non-nested dicts

#     :param exp_log_f: opened exp_log file
#     :param raw_out: True indicates "exp_print.log", otherwise indicates
#     "eval_results.log",
#         the difference is whether contains the logger header such as
#         "2022-05-02 16:55:02,843 (client:197) INFO:"

#     :return: tuple including (all_log_res, exp_stop_normal, last_line,
#     log_res_best)
#     """
#     log_res_best = {}
#     exp_stop_normal = False
#     all_log_res = []
#     last_line = None
#     for line in exp_log_f:
#         last_line = line
#         exp_stop_normal, log_res = logline_2_wandb_dict(
#             exp_stop_normal, line, log_res_best, raw_out)
#         if "'Role': 'Server #'" in line:
#             all_log_res.append(log_res)
#     return all_log_res, exp_stop_normal, last_line, log_res_best


# def logline_2_wandb_dict(exp_stop_normal, line, log_res_best, raw_out):
#     log_res = {}
#     if "INFO:" in line and "Find new best result for" in line:
#         # Logger type 1, each line for each metric, e.g.,
#         # 2022-03-22 10:48:42,562 (server:459) INFO: Find new best result
#         # for client_best_individual.test_acc with value 0.5911787974683544
#         line = line.split("INFO: ")[1]
#         parse_res = line.split("with value")
#         best_key, best_val = parse_res[-2], parse_res[-1]
#         # client_best_individual.test_acc -> client_best_individual/test_acc
#         best_key = best_key.replace("Find new best result for",
#                                     "").replace(".", "/")
#         log_res_best[best_key.strip()] = float(best_val.strip())

#     if "Find new best result:" in line:
#         # each line for all metric of a role, e.g.,
#         # Find new best result: {'Client #1': {'val_loss':
#         # 132.9812364578247, 'test_total': 36, 'test_avg_loss':
#         # 3.709533585442437, 'test_correct': 2.0, 'test_loss':
#         # 133.54320907592773, 'test_acc': 0.05555555555555555, 'val_total':
#         # 36, 'val_avg_loss': 3.693923234939575, 'val_correct': 4.0,
#         # 'val_acc': 0.1111111111111111}}
#         line = line.replace("Find new best result: ", "").replace("\'", "\"")
#         res = json.loads(s=line)
#         for best_type_key, val in res.items():
#             for inner_key, inner_val in val.items():
#                 log_res_best[f"best_{best_type_key}/{inner_key}"] = inner_val

#     if "'Role'" in line:
#         if raw_out:
#             line = line.split("INFO: ")[1]
#         res = line.replace("\'", "\"")
#         res = json.loads(s=res)
#         # pre-process the roles
#         cur_round = res['Round']
#         if "Server" in res['Role']:
#             if cur_round != "Final" and 'Results_raw' in res:
#                 res.pop('Results_raw')
#         role = res.pop('Role')
#         # parse the k-v pairs
#         for key, val in res.items():
#             if not isinstance(val, dict):
#                 log_res[f"{role}, {key}"] = val
#             else:
#                 if cur_round != "Final":
#                     if key == "Results_raw":
#                         for key_inner, val_inner in res["Results_raw"].items():
#                             log_res[f"{role}, {key_inner}"] = val_inner
#                     else:
#                         for key_inner, val_inner in val.items():
#                             assert not isinstance(val_inner, dict), \
#                                 "Un-expected log format"
#                             log_res[f"{role}, {key}/{key_inner}"] = val_inner
#                 else:
#                     exp_stop_normal = True
#                     if key == "Results_raw":
#                         for final_type, final_type_dict in res[
#                                 "Results_raw"].items():
#                             for inner_key, inner_val in final_type_dict.items(
#                             ):
#                                 log_res_best[
#                                     f"{final_type}/{inner_key}"] = inner_val
#     return exp_stop_normal, log_res



# -*- coding: utf-8 -*-
"""
Single-exp logging:
- 절대 sub_exp_* 디렉토리를 만들지 않음
- 콘솔 로그는 모든 rank에서 출력 (원하시면 rank>0 숨길 수도 있음)
- 파일(exp_print.log)은 rank0에서만 append
- wandb 헬퍼(logline_2_wandb_dict, logfile_2_wandb_dict)도 최소구현 제공
"""

import logging
import os
import sys
import ast
from typing import Tuple, List, Dict, Any


# -------------------------
# rank/main 판별 유틸
# -------------------------
def _is_main_process() -> bool:
    try:
        r = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
        return int(r) == 0
    except Exception:
        return True


# -------------------------
# 포맷터
# -------------------------
def _make_formatter(print_decimal_digits: int = 6) -> logging.Formatter:
    # 예: 2025-08-03 21:00:00,123 (client:322) INFO: message
    fmt = f"%(asctime)s (%(name)s:%(lineno)d) %(levelname)s: %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    return formatter


# -------------------------
# 메인 엔트리: main.py가 import 하는 함수
# -------------------------
def update_logger(cfg) -> logging.Logger:
    """
    main.py에서 호출하는 로거 초기화 진입점.
    - cfg.outdir 하위에 exp_print.log를 rank0만 기록
    - 콘솔 핸들러는 모든 rank에 장착
    - 기존 핸들러 제거 후 재설정(중복 방지)
    """
    # 루트 로거
    root = logging.getLogger()
    # 중복 핸들러 제거
    for h in list(root.handlers):
        try:
            root.removeHandler(h)
            h.close()
        except Exception:
            pass

    # 레벨
    # cfg.verbose가 있으면 INFO/DEBUG/ WARNING 등으로 선택
    level = logging.INFO
    try:
        v = int(getattr(cfg, "verbose", 1))
        if v >= 2:
            level = logging.DEBUG
        elif v <= 0:
            level = logging.WARNING
    except Exception:
        pass
    root.setLevel(level)

    # 포맷터
    formatter = _make_formatter(getattr(cfg, "print_decimal_digits", 6))

    # 콘솔 핸들러: 모든 rank에서 출력(원하면 _is_main_process() 로 가드 하세요)
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # 파일 핸들러: rank0만
    try:
        outdir = getattr(cfg, "outdir", "exp") or "exp"
        os.makedirs(outdir, exist_ok=True)
        if _is_main_process():
            fp = os.path.join(outdir, "exp_print.log")
            fh = logging.FileHandler(fp, mode="a", encoding="utf-8")
            fh.setLevel(level)
            fh.setFormatter(formatter)
            root.addHandler(fh)
    except Exception as e:
        # 파일 핸들러 실패해도 학습은 계속되도록
        logging.getLogger(__name__).warning(f"[logging] file handler init failed: {e}")

    # 모듈별 기본 레벨 조정이 필요하면 여기서 setLevel 추가

    return root


# -------------------------
# wandb 헬퍼 (모니터에서 import)
# -------------------------
def _safe_parse_obj(s: str) -> Any:
    """
    str(dict) 형태를 dict로 복원. 실패 시 None
    """
    try:
        return ast.literal_eval(s)
    except Exception:
        return None


def logline_2_wandb_dict(
    exp_stop_normal: bool,
    line: str,
    log_res_best: Dict[str, Any],
    raw_out: bool = False
) -> Tuple[bool, Dict[str, Any]]:
    """
    한 줄(str(dict))을 받아 wandb로 보낼 수 있는 dict로 파싱.
    실제 wandb 사용 안하면 최소 동작만 보장.
    """
    obj = _safe_parse_obj(line.strip())
    if isinstance(obj, dict):
        # eval_results.log의 한 줄(dict)을 wandb.log(...)에 바로 넘길 수 있게 반환
        return exp_stop_normal, obj
    return exp_stop_normal, {}


def logfile_2_wandb_dict(
    f,
    raw_out: bool = False
) -> Tuple[List[Dict[str, Any]], bool, str, Dict[str, Any]]:
    """
    파일 핸들을 받아 모든 줄을 dict로 파싱해서 리스트로 반환.
    (all_log_res, exp_stop_normal, last_line, log_res_best)
    """
    all_log_res: List[Dict[str, Any]] = []
    last_line = ""
    log_res_best: Dict[str, Any] = {}
    exp_stop_normal = False

    for line in f:
        line = line.strip()
        if not line:
            continue
        last_line = line
        obj = _safe_parse_obj(line)
        if isinstance(obj, dict):
            all_log_res.append(obj)
            # 가장 마지막 줄을 best로 가정(원 구현과 달라도 wandb.use=False면 영향 없음)
            log_res_best = obj

    return all_log_res, exp_stop_normal, last_line, log_res_best
