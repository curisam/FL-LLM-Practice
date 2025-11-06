import logging
import torch                # ← NEW (GPU in-place copy)
import gc                   # ← NEW (garbage-collection)
import torch
import random
import math
from federatedscope.core.message import Message

from federatedscope.core.workers.server import Server
from federatedscope.core.auxiliaries.utils import merge_param_dict

logger = logging.getLogger(__name__)


class LLMMultiLoRAServer(Server):
    """
    Server implementation
    We broadcast the model to each client and ask them to train locally
    Afterward, we collect the model back and save it as checkpoints
    """

    #멀티 LoRA 시나리오에서 어그리게이터가 전체 학습 크기/클라 수를 알고 있어야 가중 평균 등 올바른 집계를 수행 가능.
    def __init__(self,
                 ID=-1,
                 state=0,
                 config=None,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 **kwargs):
        super(LLMMultiLoRAServer,
              self).__init__(ID, state, config, data, model, client_num,
                             total_round_num, device, strategy, **kwargs)
        


        self._grouping_is_fixed = False
        # (A) cluster 샘플러 사용하는 경우 (GFL)
        if getattr(self._cfg.federate, 'sampler', 'uniform') == 'cluster' and int(self._cfg.llm.adapter.count) > 1:
            self._init_fixed_cluster_grouping()

        # (B) PFL처럼 sampler가 'uniform'이어도, 스케줄 파일/클러스터가 있으면 고정 그룹핑 사용
        rt = getattr(self._cfg.llm.adapter, 'cluster_runtime', None)
        has_schedule = False
        if rt is not None:
            if isinstance(rt, dict):
                has_schedule = bool(rt.get('schedule_file', ''))
            else:  # CfgNode 등
                has_schedule = hasattr(rt, 'schedule_file') and bool(rt.schedule_file)

        if int(self._cfg.llm.adapter.count) > 1 and has_schedule:
            self._init_fixed_cluster_grouping()



        if self._cfg.llm.adapter.count > 1: #True
            # self.aggregator.total_train_size = len(data.train_data) #모든 client들의 train data 갯수 총합. 
            self.aggregator.total_train_size = 157155 #모든 client들의 train data 갯수 총합. 
            self.aggregator.num_clients = client_num #전체 클라이언트 수


        #Local-only 모드면 샘플링 없이 전 클라로 1라운드만 돌리는 특수 모드. (지금은 사용 안 함)
        if self._cfg.llm.adapter.local_only: #False
            logger.warning("In local training mode, we will use all clients. "
                           "And we set the total round to 0 for one training "
                           "round only. ")

            self.sampler = None
            self.sample_client_num = client_num 

        # grouping 활성화 시, 각 라운드마다 클라이언트별 “어댑터 평가 결과”를 임시 저장할 버퍼 채널을 준비.
        if self._cfg.llm.adapter.grouping.use: #True (FedBiscuit의 경우만)
            self.msg_buffer['adapter_eval'] = dict()


    #핸들러 등록: cmsg_type='grouping'으로 들어오는 메시지를 callback_funcs_for_grouping으로 처리. 서버가 보낼 수 있는 후속 메시지 타입으로 set_active_adapter_idx를 선언
    def _register_default_handlers(self): 
        super()._register_default_handlers()
        self.register_handlers('grouping', self.callback_funcs_for_grouping,
                               ['set_active_adapter_idx']) 




    def _init_fixed_cluster_grouping(self):
        """
        cluster_schedule.json 또는 cfg.llm.adapter.clusters(1-based)에서
        adapter_idx(0-based) -> [client_ids(1-based)] 고정 그룹핑 및
        client_id -> adapter_idx 매핑 구성.
        """
        import os, json

        clusters_1b = None
        rt = getattr(self._cfg.llm.adapter, 'cluster_runtime', None)
        sched_path = None
        if rt is not None:
            if isinstance(rt, dict):
                sched_path = rt.get('schedule_file', None)
            else:
                sched_path = getattr(rt, 'schedule_file', None)

        if sched_path and os.path.exists(sched_path):
            with open(sched_path, 'r') as f:
                info = json.load(f)
            clusters_1b = info.get('clusters_1_based') or info.get('clusters')
        if clusters_1b is None and hasattr(self._cfg.llm.adapter, 'clusters'):
            clusters_1b = getattr(self._cfg.llm.adapter, 'clusters')

        if clusters_1b is None:
            raise ValueError("고정 그룹핑 초기화 실패: 스케줄/클러스터 정의가 없습니다.")

        # dict: adapter_idx -> [client_ids]
        self._fixed_groups = {int(aidx): sorted(list(map(int, group)))
                            for aidx, group in enumerate(clusters_1b)}

        # dict: client_id -> adapter_idx
        self._client2adp = {int(cid): int(aidx)
                            for aidx, group in self._fixed_groups.items()
                            for cid in group}
        


        self._grouping_is_fixed = True



        

    #라운드 시작 훅


    """
    Grouping 주기(예: r 라운드마다) + warmup 끝난 시점에 도달하면

    Client에게 adapter_eval을 브로드캐스트해 각 LoRA 어댑터의 평가(avg_loss 등) 를 모든 Client가 수행하도록 시킴

    그 결과(Client→Server)가 다 모이면(아래 check_and_grouping) 그때 그루핑을 실행하고,

    이어서 다음 라운드를 시작(skip_grouping=True로 재호출)

    포인트: 그루핑 라운드에서는 곧장 학습으로 안 가고, 먼저 adapter_eval 요청을 쏘고 클라 회신을 기다리도록 return으로 탈출한다.
    
    
    """
    def _start_new_training_round(self, aggregated_num=0, skip_grouping=False): #grouping trigger에 걸리면 모든 클라이언트에게 adapter_eval' 메시지를 보냄. 그렇지 않으면 'model_para' 메시지를 sampling한 클라이언트들에게 전달.
        if self._grouping_is_fixed:
            return super()._start_new_training_round(aggregated_num) 
 
 
        if self._cfg.llm.adapter.grouping.use and not skip_grouping:
            total_warmup_round = 0
            if self._cfg.llm.adapter.warmup.use: 
                warmup_round = self._cfg.llm.adapter.warmup.round
                total_warmup_round = \
                    warmup_round * self._cfg.llm.adapter.count #warm up round 갯수를 총 lora adapter 갯수배 만큼 증강.

            r = self._cfg.llm.adapter.grouping.round #re-grouping할 주기
            if self.state >= total_warmup_round and \
                    (self.state - total_warmup_round) % r == 0: #Grouping 트리거 조건.
                logger.info('Server: Performing a grouping step...')
                #모든 클라이언트들에게 ''adapter_eval'' 메시지와 함께 서버의 파라미터를 content로 넣어서 보냄.
                self.broadcast_model_para(msg_type='adapter_eval',
                                          filter_unseen_clients=False)
                return

        super()._start_new_training_round(aggregated_num) #다음 FL 라운드에 참여할 클라이언트를 정한 후 'model_para' 메시지로 Server model para로 컨텐츠 넣어서 보냄.


    def trigger_for_start(self):

        # start feature engineering (This part is for hard code)
        if self.check_client_join_in(): ##전체 클라이언트 수만큼 join_in이 반영됐는지 여부
            logger.info('Waited all clients join, start now...')
            if self._grouping_is_fixed:
                for adap_idx, receiver in getattr(self, "_fixed_groups", {}).items():
                    if receiver:
                        self.comm_manager.send(Message(
                            msg_type='set_active_adapter_idx', sender=self.ID,
                            receiver=receiver, state=self.state,
                            timestamp=self.cur_timestamp, content=adap_idx))
                # 2) PFL일 경우 전체 참여를 원하면 sample_client_num 보정
                s_cur = self._current_required_sample_num()  # sampler='uniform'이면 federate.sample_client_num을 그대로 반환
                if s_cur is None or int(s_cur) <= 0:         # -1 이나 0이면 전체
                    s_cur = self.client_num
                self.trigger_for_feat_engr(self.broadcast_model_para, {
                    'msg_type': 'model_para',
                    'sample_client_num': s_cur,
                })
                logger.info('----------- Starting training (Round #{:d}) -------------'.format(self.state))
            else:
                #모든 클라이언트들에게 ''adapter_eval'' 메시지와 함께 서버의 파라미터를 content로 넣어서 보냄.
                self.trigger_for_feat_engr(self.broadcast_model_para, {
                    'msg_type': 'adapter_eval',
                    'filter_unseen_clients': False,
                })

                logger.info(
                    '----------- Starting training (Round #{:d}) -------------'.
                    format(self.state))
                logger.info('Server: Performing a grouping step...')


    def _perform_federated_aggregation(self):
        """
        Perform federated aggregation and update the global model
        """

        #현재 라운드의 학습 결과 버퍼 가져오기
        train_msg_buffer = self.msg_buffer['train'][self.state]
        for model_idx in range(self.model_num): #self.model_num=1인 상황.
            model = self.models[model_idx]
            aggregator = self.aggregators[model_idx]
            msg_list = list()

            # ① 클라이언트 피드백 수집
            msg_list = []
            for _, content in train_msg_buffer.items():
                if self.model_num == 1: #True. Client에서 보낸 content=(데이터크기, 파라미터dict)을 뽑아 aggregation input인 msg_list에 모음.
                    msg_list.append(content) #content= (sample_size, model)
                else:
                    n, multi_states = content
                    msg_list.append((n, multi_states[model_idx]))
            
            aggregated_num = len(msg_list)


            # import ipdb; ipdb.set_trace(context=15)


            # ② Aggregator 호출
            agg_info = {
                'client_feedback': msg_list,
                'recover_fun': self.recover_fun,
            }

            warmup_round = self._cfg.llm.adapter.warmup.round #warm up 라운드

            total_warmup_round = warmup_round * self._cfg.llm.adapter.count #모든 adapter에 적용되는 총 warm up 라운드 수

            if getattr(self._cfg.federate, 'sampler', 'uniform') == 'cluster':
                result = aggregator.aggregate(agg_info)

            else: 
                server_type = str(getattr(self._cfg.federate, "server_type", "")).lower()
                is_moe = "moeserver" in server_type

                if (getattr(self._cfg.llm.adapter.warmup, "use", False) and self.state < total_warmup_round) or is_moe: #warm up 라운드일 떄 적용되는 aggregation
                    result = aggregator.aggregate(agg_info)
                else: #warm up 라운드가 아닐 때 적용되는 aggregation
                    result = aggregator.aggregate_on_model(agg_info)


            # # Due to lazy load, we merge two state dict
            merged_param = merge_param_dict(model.state_dict().copy(), result)
            model.load_state_dict(merged_param, strict=False)
 
            # ④ 반환된 텐서를 CPU로 이동시켜 GPU 메모리 해제
            for tensor in result.values():
                _ = tensor.cpu()
            result.clear()

            # ⑤ 강제 캐시 비우기 + 가비지 수집
            del msg_list
            torch.cuda.empty_cache()
            gc.collect()



        return aggregated_num



    def callback_funcs_for_grouping(self, message: Message): #클라가 어댑터별 평가 결과를 들고 서버에 msg_type='grouping' 으로 응답한 상황에서 발동.


        if self._grouping_is_fixed:
            return False


        rnd = message.state
        sender = message.sender
        content = message.content

        if rnd not in self.msg_buffer['adapter_eval'].keys():
            self.msg_buffer['adapter_eval'][rnd] = dict()

        #rnd에서, 해당 sender(클라)가 보낸 (어댑터 idx, 평균손실) 리스트를 손실 오름차순으로 정렬해 저장.

        """
        {'adapter_0_avg_loss': 0.75,  'adapter_1_avg_loss': 0.62, 'adapter_2_avg_loss': 0.81} -> [(0, 0.75), (1, 0.62), (2, 0.81)]
        """
        self.msg_buffer['adapter_eval'][rnd][sender] = [(i, content[f'adapter_{i}_avg_loss']) for i in range(self._cfg.llm.adapter.count)]

        #key=는 “정렬 기준 값을 뽑는 함수. lambda x: x[1]는 **각 튜플의 두 번째 원소(= 평균손실)**를 리턴하므로, “평균손실이 작은 것부터 큰 것 순서”로 오름차순 정렬.
        #[(0, 0.75), (1, 0.62), (2, 0.81)] -> [(1, 0.62), (0, 0.75), (2, 0.81)]
        self.msg_buffer['adapter_eval'][rnd][sender] = sorted(self.msg_buffer['adapter_eval'][rnd][sender], key=lambda x: x[1]) #x는 self.msg_buffer['adapter_eval'][rnd][sender]의 각 원소.

        return self.check_and_grouping() #모든 클라의 평가가 모였으면 그룹핑을 실제로 실행. 이 값자체는 Boolean. Grouping이 끝나여 다음라운드를 진행해도 되는지 관한 것.


    def check_and_grouping(self):

        if self._grouping_is_fixed:
            self.adapter_grouping = dict(getattr(self, "_fixed_groups", {}))
            self.client2adapter   = dict(getattr(self, "_client2adp", {}))
            return False



        if 'adapter_eval' not in self.msg_buffer.keys() or \
                len(self.msg_buffer['adapter_eval'].keys()) == 0:
            return False #Grouping 보류

        buffer = self.msg_buffer['adapter_eval']
        cur_round = max(buffer.keys())


        #0) 준비

        cur_buffer = buffer[cur_round] #“평균손실이 작은 것부터 큰 것 순서”로 오름차순 정렬이 된 상태. [(1, 0.62), (0, 0.75), (2, 0.81)]

        if len(cur_buffer) < self.client_num:
            return False #Grouping 보류

        num_adap = self._cfg.llm.adapter.count

        balance = bool(getattr(self._cfg.llm.adapter, "balance", True))  # ← 새 플래그(기본 True)
        
        # 최종 결과 컨테이너 초기화
        self.adapter_grouping = dict()

        if balance:

            #각 클라의 선호 리스트를 이터레이터로 변환. 이후 next(cur_buffer[sender])로 “다음 선호”를 꺼냄. 
            for sender in cur_buffer.keys():
                cur_buffer[sender] = iter(cur_buffer[sender])

            #라운드-로빈/용량 제한 기반 할당
            # 각 클라에게 “좋아하는 어댑터”를 우선 주되, 어댑터별 수용 인원 상한을 max_size로 엄격히 관리해 균형 잡힌 그룹 크기를 만든다.
            # 과밀 어댑터는 잘라서 확정하고, 남은 클라는 그 다음 반복에서 자기 선호 목록의 다음 어댑터로 시도한다.
            # 이 과정을 반복하면, 대략 공평한 크기의 그룹이 얻어진다(= 각 어댑터당 클라 비슷한 수).      
            


            
            adapter_grouping = {i: [] for i in range(num_adap)} # (후보) 임시 버킷


            senders = [sender for sender in cur_buffer.keys()] #아직 미배정 클라들. 처음엔 전체 클라, 반복하면서 변경.
            random.shuffle(senders) #셔플을 해 동률/경합 시 랜덤성을 주어 공평성(편향 최소화)을 확보.

            unassigned_client_num = len(senders)

            #1) 반복 루프
            while unassigned_client_num > 0:
                num_finished = len(self.adapter_grouping) # 이미 확정된 어댑터 수

                max_size = math.ceil(unassigned_client_num /
                                    (num_adap - num_finished)) #남은 어댑터 수에 비례해 정원(cap) 을 동적으로 계산. ⇒ “남은 인원 ÷ 남은 어댑터 수”의 올림이므로, 최대 균형에 맞는 상한.

                # step 1: 현재 대기 중인 클라를 “남은 어댑터 중 best인 것"에 1차 배치
                """
                각 클라가 아직 확정되지 않은 어댑터 중에서 가장 선호하는 곳에 후보 등록.
                특정 어댑터가 이미 확정되어 버킷에서 빠졌다면, 다음 선호로 자동 이동.
                """
                for sender in senders:
                    adap_idx, loss = next(cur_buffer[sender])
                    while adap_idx not in adapter_grouping: # 이미 확정되어 빠진 어댑터면
                        adap_idx, loss = next(cur_buffer[sender]) # 다음 선호로 이동
                    adapter_grouping[adap_idx].append(sender)

                # step 2: 가장 지원자가 많은 어댑터 선택. 
                """
                수요가 가장 큰 어댑터를 하나 고름.
                """
                max_adap_idx_size = [0, 0]
                for adap_idx, candidates in adapter_grouping.items():
                    if len(candidates) > max_adap_idx_size[1]:
                        max_adap_idx_size = [adap_idx, len(candidates)]

                # step 3: 해당 어댑터에서 max_size만 확정
                """
                인기 어댑터에 몰린 지원자 중 앞에서부터 정원만큼 확정.
                (앞쪽 순서는 senders가 섞여 있으므로 randomness도 fair)
                """
                adap_idx = max_adap_idx_size[0]
                candidates = adapter_grouping[adap_idx][:max_size]

                # step 4: 남은 후보를 다음 반복으로 넘기고, 어댑터 하나 확정 종료
                senders = adapter_grouping[adap_idx][max_size:] # 초과분만 다음 라운드로
                self.adapter_grouping[adap_idx] = candidates #선택분은 확정지음
                adapter_grouping.pop(adap_idx) # 확정된 어댑터 제거
                unassigned_client_num -= len(self.adapter_grouping[adap_idx])
                logger.info(f'Adapter {adap_idx} is done with the clients '
                            f'{self.adapter_grouping[adap_idx]}')
                
        else:
            # ===== 단순 최소-loss 모드 (균형 고려 X, 동률이면 랜덤) =====
            # cur_buffer: {sender: [(adap_idx, avg_loss), ... (asc)]}
            plain = {i: [] for i in range(num_adap)}
            for sender, ranked in cur_buffer.items():
                if not ranked:
                    # 비정상 케이스 방어: 아무 후보가 없으면 랜덤 배정
                    chosen = random.randrange(num_adap)
                    plain[chosen].append(sender)
                    continue

                min_loss = ranked[0][1]
                # 동률 후보 모으기 (float 안전 위해 isclose 사용)
                ties = [ad for (ad, l) in ranked
                        if math.isclose(l, min_loss, rel_tol=1e-12, abs_tol=1e-12)]
                chosen = random.choice(ties) if len(ties) > 1 else ties[0]
                plain[chosen].append(sender)

            # 빈 어댑터는 제외한 dict로 축소
            self.adapter_grouping = {k: v for k, v in plain.items() if len(v) > 0}            
            
        # 2) Grouping 결과 모든 클라이언트들에게 브로드캐스트 & 학습 재개
        for adap_idx, receiver in self.adapter_grouping.items():
            self.comm_manager.send(
                Message(msg_type='set_active_adapter_idx',
                        sender=self.ID,
                        receiver=receiver,
                        state=self.state,
                        timestamp=self.cur_timestamp,
                        content=adap_idx))

        # resume the training based on the new group...
        self._start_new_training_round(skip_grouping=True)

        return True  # move_on_flag
