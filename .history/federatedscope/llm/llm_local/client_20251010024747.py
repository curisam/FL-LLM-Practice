import copy
import logging

import torch
from torch.utils.data import random_split, ConcatDataset, Subset

import random

from federatedscope.core.message import Message
from federatedscope.core.workers.client import Client
from federatedscope.core.data import ClientData
from federatedscope.llm.utils_dist import barrier_all


import os
import pickle
import json

logger = logging.getLogger(__name__)



class LLMMultiLoRAClient(Client):
    """
    Client implementation of
    "Offsite-Tuning: Transfer Learning without Full Model" paper
    """
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None,
                 model=None,
                 device='cpu',
                 strategy=None,
                 *args,
                 **kwargs):

        super(LLMMultiLoRAClient,
              self).__init__(ID, server_id, state, config, data, model, device,
                             strategy, *args, **kwargs)
        

    def _sync_after_adapter_swap(self):
        """어댑터 스왑 직후 랭크 동기화 + CUDA 동기화로 레이스 창 제거."""
        try:
            # 트레이너가 Accelerator를 들고 있음
            acc = getattr(self.trainer, "accelerator", None)
            if acc is not None:
                acc.wait_for_everyone()
        except Exception:
            pass
        # 프로세스 배리어 + 커널 동기화
        barrier_all()
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


    def _register_default_handlers(self):
        super()._register_default_handlers()
        self.register_handlers('adapter_eval',
                               self.callback_funcs_for_adapter_eval,
                               ['grouping']) #서버로부터 'adapter_eval' 이름의 Message를 받음. 해당 메시지의 content는 서버 모델..
        
        self.register_handlers('set_active_adapter_idx',
                               self.callback_funcs_for_setting_adapter_idx,
                               [None])#서버로부터 'set_active_adapter_idx' 이름의 Message를 받음. 해당 메시지의 content는 해당 클라이언트가 어떤 adpater를 사용해야하는지에 대한 index 정보.
        

    def _append_raw_line(self, role_str: str, round_idx, results_dict: dict, filename: str, adapter_idx: int = None):
        """
        adapter_idx가 인자로 오면 그대로 top-level에 기록.
        없으면 results_dict 또는 ctx에서 추출해 기록 시도(역호환).
        """
        try:
            outdir = getattr(self._monitor, "outdir", None)
            os.makedirs(outdir, exist_ok=True)
            outpath = os.path.join(outdir, filename)

            # adapter_idx 자동 보강(인자 > results_dict > ctx 순)
            if adapter_idx is None:
                adapter_idx = results_dict.get("adapter_idx", None)
                if adapter_idx is None:
                    try:
                        adapter_idx = int(getattr(self.trainer.ctx, "current_adapter_idx", -1))
                    except Exception:
                        adapter_idx = None

            line = {
                "Role": role_str,
                "Round": round_idx,
                "Results_raw": results_dict
            }
            # top-level에 명시적으로 남김(있을 때만)
            if adapter_idx is not None:
                line["Adapter"] = int(adapter_idx)

            with open(outpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
        except Exception as e:
            self.logger.warning(f"[append_raw_line] failed to write {filename}: {e}")



    def callback_funcs_for_model_para(self, message: Message):
        round = message.state  # 서버가 보낸 라운드 번호
        sender = message.sender  # 메시지를 보낸 주체(서버) ID
        timestamp = message.timestamp  # 서버 시각
        content = message.content  # 실제 모델 파라미터


        if self._cfg.federate.process_num > 1:
            for k, v in content.items():
                content[k] = v.to(self.device) 

        is_main = self._is_main_process()


        self.trainer.update(content,
                            strict=self._cfg.federate.share_local_model) # 서버에서 보낸 파라미터로 모델을 덮어씌웁니다. (일단 모든 adapter 다 update 됨.)
        self.state = round


        if getattr(self._cfg.federate, 'sampler', 'uniform') == 'cluster' or getattr(self._cfg.llm.adaptercluster_runtime, 'scheduled_file', '') != '' :
            # This is client-wise
            self.model.set_active_adapter(f'Adapter_{self.adapter_idx}')
            adapter_idx = self.adapter_idx
            self.trainer.ctx.current_adapter_idx = adapter_idx
            self.model.train()
            self._sync_after_adapter_swap()


        elif self._cfg.llm.adapter.count > 1: #3. “군집/선택” 모드:
            # This is clustering
            warmup_round = self._cfg.llm.adapter.warmup.round #25
            total_warmup_round = warmup_round * self._cfg.llm.adapter.count #75

            if self._cfg.llm.adapter.warmup.use and \
                    self.state < total_warmup_round:
                # Initialization for all adapters
                adapter_idx = self.state // warmup_round #워밍업 라운드: adapter_idx = state // warmup_round로 0..(count-1) 순환 활성화.
            elif self._cfg.llm.adapter.grouping.use: #True. 서버/외부에서 정해준 self.adapter_idx를 그대로 사용. GFL 과정 중에는  Warm-up 이후에는 일반적으로 이것.
                adapter_idx = self.adapter_idx
            else: #서버에서 정해주지 않아서 val셋으로 각 어댑터를 평가해서 val_avg_loss가 가장 작은 어댑터 후보를 뽑고(동률이면 리스트에 모음), 그 중 랜덤으로 하나 선택.
                # select the adapter with min val loss

                with torch.no_grad():
                    min_loss, adapter_indices = 10.0, []
                    for i in range(self._cfg.llm.adapter.count):
                        if len(self.data.val_data) == 0:
                            adapter_indices.append(i)
                            continue

                        #self.trainer.ctx.model에서도 동시에 적용됨.
                        self.model.set_active_adapter(f'Adapter_{i}') 
                        self.model.eval()
                        self._sync_after_adapter_swap()

                        # ✅ (중요) 이번 라운드 val 전에 split 캐시 리셋 — 누적 방지
                        self._reset_ctx_split_metrics('val')

                        metrics = self.trainer.evaluate(
                            target_data_split_name='val')
                        
                        logger.info(
                            f'Adapter {i} with the results: {metrics}')
                        
                        if i == 0 or min_loss > metrics['val_avg_loss']:
                            min_loss, adapter_indices = metrics[
                                'val_avg_loss'], [i]
                            
                        elif min_loss == metrics['val_avg_loss']:
                            adapter_indices.append(i)

                    logger.info(adapter_indices)
                    adapter_idx = random.choice(adapter_indices)
            # activate the selected adapter for further training
            logger.info(
                f'Activate the adapter {adapter_idx} for training...')
            self.model.set_active_adapter(f'Adapter_{adapter_idx}')
            self.trainer.ctx.current_adapter_idx = adapter_idx
            self.model.train()
            self._sync_after_adapter_swap()
            
        else:
            raise ValueError(
                'You should set llm.adapter.local_only to True '
                'or llm.adapter.count > 1')

        
        # ✅ (중요) 이번 라운드 train 전에 split 캐시 리셋 — 누적 방지
        self._reset_ctx_split_metrics('train')


        #할당된 adapter로부터 학습.

        """
        # sample_size: 전체 train data 갯수
        # model_para_all: requires_grad=True인 파라미터 포함인 것 혹은 self.adapter_names에 있는 어댑터 이름이 파라미터 이름 문자열에 포함되면, requires_grad=False라도 포함. 즉 활성/비활성 모든 adapter들만 반환.
        # results: split routine 돌며 집계된 results (num_total, loss, acc 등) 리턴
        """

        sample_size, model_para_all, results = self.trainer.train(round_num=round)  #여기서 model_para_all은 active 뿐만 아니라 모든 adpater 다 받아온 것. 그리고 DDP여도 .module이 다  제거된 상태.
        if self._cfg.federate.share_local_model and not \
                self._cfg.federate.online_aggr:
            model_para_all = copy.deepcopy(model_para_all) # 안전하게 복사
            model_para_all = {
                key: value
                for key, value in model_para_all.items()
                if f'Adapter_{adapter_idx}.' in key
            } #Active adapter 정보인 Adapter_{adapter_idx}에 한한 것만 필터링.

        # ✅ 1)랭크별 train result 로그 띄움 2)모들 랭크 종합한 train result 로그 띄움.
        self._log_split_metrics(
            role_str=f'Client #{self.ID}',
            round_idx=self.state,
            split='train',
            trainer_ctx=self.trainer.ctx
        )

        # train 끝나고 _log_split_metrics(...) 뒤:     

        ctx = getattr(self.trainer, "ctx", None)
        agg = getattr(ctx, "eval_metrics", {}) if ctx is not None else {}
        train_agg = {k: v for k, v in (agg or {}).items() if k.startswith("train_")}#한 클라이언트 기준 집계된 train split 결과.
        
        if is_main: # ✅ rank0(=main process)에서만 집계본을 파일로 기록
            self._append_raw_line(
                role_str=f"Client #{self.ID}",
                round_idx=self.state,
                results_dict=train_agg,
                filename="train_results.raw",
                adapter_idx=self.trainer.ctx.current_adapter_idx

            )

    

        train_agg = {
            k: v for k, v in getattr(self.trainer.ctx, "eval_metrics", {}).items()
            if k.startswith("train_")
        }
        if is_main: # ✅ exp_print용: train 집계본만 한 줄 (rank0에서만)
            self.logger.info({
                'Role': f'Client #{self.ID}',
                'Round': self.state,
                'Results_raw': train_agg
            }) #INFO: {'Role': 'Client #44', 'Round': 0, 'Results_raw': {'train_total': 480, 'train_loss': 348.3512268066406, 'train_avg_loss': 0.7257317225138347, 'train_seen': 480, 'train_correct': 234, 'train_acc': 0.4875}}

        # Return the feedbacks to the server after local update

        shared_model_para = model_para_all

        # ✅ 모든 rank에 대해서 동일하게 메시지를 보냄. 서버는 각 process마다 동일하게 self.comm_manager.comm_queue를 관리해야함.
        self.comm_manager.send(
            Message(msg_type='model_para', # ↔ 서버가 “train” 단계로 인식
                    sender=self.ID, # 이 클라이언트 ID
                    receiver=[sender], # 앞서 저장한 서버 ID
                    state=self.state, # (같은) 라운드 번호
                    timestamp=self._gen_timestamp(init_timestamp=timestamp,
                                                instance_number=sample_size), # → 서버의 time-based staleness 제어용
                    content=(sample_size, shared_model_para))) # 데이터 갯수 및 로컬 active adapter 모델만을 content로 담아서 보낸다.
            

    def callback_funcs_for_evaluate(self, message: Message):
        """
        The handling function for receiving the request of evaluating
        """

        # 항상 먼저 초기화해서 NameError 방지
 

        metrics = {}
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state


        is_main = self._is_main_process()

        # 1) 서버 파라미터로 동기화
        if message.content is not None:
            self.trainer.update(
                message.content,
                strict=self._cfg.federate.share_local_model
            )



        # ★ 1-1) 이번 평가에 사용할 adapter_idx를 확정
        if self._cfg.llm.adapter.local_only:
            eval_idx = self.ID
        elif self._cfg.llm.adapter.grouping.use:
            # 서버가 이미 골라준 adapter (ex. self.adapter_idx)
            eval_idx = self.adapter_idx
        elif self._cfg.llm.adapter.warmup.use and self.state < self._cfg.llm.adapter.warmup.round * self._cfg.llm.adapter.count:
            eval_idx = self.state // self._cfg.llm.adapter.warmup.round
        else:
            # 필요하면 fallback: 현재 활성 어댑터 유지 or 최근 선택 값
            eval_idx = getattr(self, 'adapter_idx', 0)

        # ★ 1-2) 모델/트레이너에 동일 어댑터 반영
        self.model.set_active_adapter(f'Adapter_{eval_idx}')
        self.model.eval()
        self.trainer.ctx.current_adapter_idx = eval_idx
        self._sync_after_adapter_swap()

        # 2) 평가 실행
        try:
            # ✅ (중요) 이번 라운드 평가 시작 전, 요청된 split 캐시를 선제 리셋
            #    (val → test 순서로 돌더라도 라운드 간/스플릿 간 누적 방지)
            for sp in set(self._cfg.eval.split): #['val', 'test']
                self._reset_ctx_split_metrics(sp)

            if self._cfg.finetune.before_eval: #False. PFL에서는 True로 해도 될듯.
                self.trainer.finetune()

            for split in self._cfg.eval.split:  #['val', 'test']
                # logger.info(f"[DEBUG] after evaluate(split={split}): eval_metrics={eval_metrics}")
                # eval_metrics = self.trainer.evaluate(target_data_split_name=split)
                eval_metrics = self.trainer.evaluate(target_data_split_name=split)
                """
                    eval_metrics = {
                    f'{split}_total':    int(total_all),
                    f'{split}_loss':     float(loss_all),
                    f'{split}_avg_loss': float(loss_all / total_all),
                    f'{split}_seen':     int(seen_all),
                    f'{split}_correct':  int(correct_all),
                    f'{split}_acc':      float(correct_all / max(1, seen_all))
                    } 

                    의 형태.    
                """

                if is_main:
                    logger.info(f"[DEBUG][after {split}] eval_metrics            = {eval_metrics}")
                    logger.info(f"[DEBUG][after {split}] metrics (merged so far) = {metrics}")
                    logger.info(f"[DEBUG][after {split}] ctx.eval_metrics        = {self.trainer.ctx.eval_metrics}")



                metrics.update(**eval_metrics)


                # ✅ 원하는 포맷의 로그 3종(로컬 / per-proc / 집계) 출력
                self._log_split_metrics(
                    role_str=f'Client #{self.ID}',
                    round_idx=self.state,
                    split=split,
                    trainer_ctx=self.trainer.ctx
                )

        except Exception as e:
            # 평가 중 문제가 나도 서버로는 빈 dict라도 보내도록
            self.logger.warning(f"[evaluate] exception during evaluation: {e}", exc_info=True)



          

        # 우선순위: trainer.ctx.eval_metrics(집계본) -> metrics(병합본)
        ctx = getattr(self.trainer, "ctx", None)
        agg_all = getattr(ctx, "eval_metrics", {}) if ctx is not None else {}

        logger.info(f"[DEBUG][before write] agg_all={agg_all}, metrics={metrics}") 

        base = {**agg_all, **metrics}

        # test_/val_만 추출
        combined = {k: v for k, v in (base or {}).items()
                    if k.startswith("test_") or k.startswith("val_")}
        


        has_test = any(k.startswith("test_") for k in combined) #Boolean
        has_val  = any(k.startswith("val_") for k in combined) #Boolean

        logger.info(f"[DEBUG] combined keys={list(combined.keys())}, has_val={has_val}, has_test={has_test}")


        write_key = (self.ID, int(self.state))

        # 3) rank0만 파일 기록/서버 전송
        if is_main:

            self._ensure_outdir()


            # ✅ 파일 기록: test & val 둘 다 있고, 아직 안 쓴 경우 1회만
            if has_test and has_val:
                if write_key not in self._eval_written_marker:
                    self._append_raw_line(
                        role_str=f"Client #{self.ID}",
                        round_idx=self.state,
                        results_dict=combined,
                        filename="eval_results.raw",
                        adapter_idx=self.trainer.ctx.current_adapter_idx
                    )
                    self._eval_written_marker.add(write_key)
                else:
                    self.logger.debug(f"[skip duplicate eval write] {write_key}")
            else:
                self.logger.debug(
                    f"[skip write eval_results.raw] only one split present. "
                    f"available={list(combined.keys())}"
                )


            # 3-2) 모니터에 기록. round_formatted_results_raw 반환하는 것.
            try:
                self._monitor.format_eval_res(
                    metrics, rnd=self.state, role=f'Client #{self.ID}', forms=['log']
                )
            except Exception as e:
                # metrics가 비어도 로깅 실패하지 않도록
                self.logger.debug(f"[format_eval_res] skip log due to: {e}")

        #4) 모든 RANK에서 서버로  메시지 전송
        if metrics:    # ← metrics 가 비어있지 않을 때만 전송

            # _seen, _correct 접미사 키 제거한 사본 만들기
            pruned_metrics = {
                k: v for k, v in metrics.items()
                if not (k.endswith('_seen') or k.endswith('_correct'))
            }

            # 비워졌으면 굳이 보내지 않음
            if pruned_metrics:
                self.comm_manager.send(
                        Message(msg_type='metrics',
                                sender=self.ID,
                                receiver=[sender],
                                state=self.state,
                                timestamp=timestamp,
                                content=pruned_metrics)
                    )
        else:
            logger.debug(f"[skip send metrics] empty metrics for Client #{self.ID}, round={self.state}")



    def callback_funcs_for_adapter_eval(self, message: Message):
        """
        모든 어댑터에 대해:
        - 서버 전송: val_avg_loss (기존 유지)
        - 로컬 파일(adapter_eval_result.raw) 기록: val_* 및 test_* 메트릭만 기록 (train 제거)
        """
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state

        # 0) 서버 파라미터 동기화 (기존 유지)
        if message.content is not None:
            self.trainer.update(
                message.content,
                strict=self._cfg.federate.share_local_model
            )

        # 1) 서버로 보낼 요약 (각 어댑터의 val_avg_loss만)
        metrics = {}

        # 2) 파일 기록은 rank0만 수행 (중복 방지용 마커)
        is_main = getattr(self, "_is_main_process", lambda: True)()
        if is_main and not hasattr(self, "_adapter_eval_written_marker"):
            self._adapter_eval_written_marker = set()  # (client_id, round, adapter_idx)

        results_to_write = []  # 어댑터별 val/test 결과를 모아 한 번에 기록

        with torch.no_grad():
            for i in range(self._cfg.llm.adapter.count):
                # 어댑터 전환
                self.model.set_active_adapter(f'Adapter_{i}')
                self.model.eval()
                self._sync_after_adapter_swap()

                # -------- (A) 서버 전송용: val --------
                try:
                    self._reset_ctx_split_metrics('val')
                except Exception:
                    pass
                val_metrics = self.trainer.evaluate(target_data_split_name='val')
                logger.info(f'Client {self.ID} Adapter {i} with val results: {val_metrics}')
                # ★ 서버로는 val_avg_loss만 유지
                metrics[f'adapter_{i}_avg_loss'] = val_metrics['val_avg_loss']

                # -------- (B) 파일 기록용: val/test --------
                rec = {
                    "client_id": int(self.ID),
                    "round": int(self.state),
                    "adapter_idx": int(i),
                }
                # val_* 메트릭 추가
                rec.update({k: v for k, v in val_metrics.items() if k.startswith("val_")})

                # test_* 메트릭 추가
                try:
                    self._reset_ctx_split_metrics('test')
                except Exception:
                    pass
                try:
                    test_metrics = self.trainer.evaluate(target_data_split_name='test')
                    rec.update({k: v for k, v in test_metrics.items() if k.startswith("test_")})
                    logger.info(f'Client {self.ID} Adapter {i} with test results: {test_metrics}')
                except Exception as e:
                    logger.debug(f"[adapter_eval] skip test eval for adapter {i}: {e}")

                results_to_write.append(rec)

        # 3) 서버로 메시지 전송(기존 포맷 유지: 각 어댑터의 val_avg_loss만 포함)
        self.comm_manager.send(
            Message(
                msg_type='grouping',
                sender=self.ID,
                receiver=[sender],
                state=self.state,
                timestamp=timestamp,
                content=metrics
            )
        )

        # 4) 로컬 파일 기록: adapter_eval_result.raw (rank0만)
        if is_main:
            try:
                self._ensure_outdir()
            except Exception:
                pass

            for rec in results_to_write:
                key = (rec["client_id"], rec["round"], rec["adapter_idx"])
                if key in self._adapter_eval_written_marker:
                    continue
                try:
                    self._append_raw_line(
                        role_str=f"Client #{self.ID}",
                        round_idx=self.state,
                        results_dict=rec,
                        filename="adapter_eval_result.raw",
                        adapter_idx=rec["adapter_idx"]
                    )
                    self._adapter_eval_written_marker.add(key)
                except Exception as e:
                    logger.debug(f"[adapter_eval] write failed for {key}: {e}")






    def callback_funcs_for_setting_adapter_idx(self, message: Message): #서버에서 지정해 준 adapter idx로 지정하여 학습.
        self.adapter_idx = message.content
