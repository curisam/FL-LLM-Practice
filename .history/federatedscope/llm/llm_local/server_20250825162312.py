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
        if self._cfg.llm.adapter.count > 1: #True
            self.aggregator.total_train_size = len(data.train_data) #모든 client들의 train data 갯수 총합. 
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
        

    #라운드 시작 훅


    """
    Grouping 주기(예: r 라운드마다) + 워름업 끝난 시점에 도달하면

    Client에게 adapter_eval을 브로드캐스트해 각 LoRA 어댑터의 평가(avg_loss 등) 를 모든 Client가 수행하도록 시킴

    그 결과(Client→Server)가 다 모이면(아래 check_and_grouping) 그때 그루핑을 실행하고,

    이어서 다음 라운드를 시작(skip_grouping=True로 재호출)

    포인트: 그루핑 라운드에서는 곧장 학습으로 안 가고, 먼저 adapter_eval 요청을 쏘고 클라 회신을 기다리도록 return으로 탈출한다.
    
    
    """
    def _start_new_training_round(self, aggregated_num=0, skip_grouping=False):
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

            #모든 클라이언트들에거 ''adapter_eval'' 메시지와 함께 서버의 파라미터를 content로 넣어서 보냄.
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
        (메모리 절약 – in-place parameter copy + GPU 메모리 즉시 해제)
        """
        logger.info(f"[in-place aggregation 시작] round={self.state}")

        #현재 라운드의 train buffer 수집.
        train_msg_buffer = self.msg_buffer['train'][self.state]
        aggregated_num   = 0

        for model_idx in range(self.model_num):
            model      = self.models[model_idx]
            aggregator = self.aggregators[model_idx]

            # ① 클라이언트 피드백 수집
            msg_list = []
            for _, content in train_msg_buffer.items():
                if self.model_num == 1: #True. Client에서 보낸 (데이터크기, 파라미터dict)을 뽑아 aggregation input인 msg_list에 모음.
                    msg_list.append(content)
                else:
                    n, multi_states = content
                    msg_list.append((n, multi_states[model_idx]))
            aggregated_num = len(msg_list)

            # ② Aggregator 호출
            agg_info     = {
                "client_feedback": msg_list,
                "recover_fun"    : self.recover_fun,
            }
            result_state = aggregator.aggregate(agg_info)  # GPU 텐서 반환

            # ③ in-place로 모델 파라미터 덮어쓰기
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in result_state:
                        param.data.copy_(result_state[name].to(param.device))

            # ④ 반환된 텐서를 CPU로 이동시켜 GPU 메모리 해제
            for tensor in result_state.values():
                _ = tensor.cpu()
            result_state.clear()

            # ⑤ 강제 캐시 비우기 + 가비지 수집
            del msg_list
            torch.cuda.empty_cache()
            gc.collect()

        return aggregated_num





    def callback_funcs_for_grouping(self, message: Message):
        rnd = message.state
        sender = message.sender
        content = message.content

        if rnd not in self.msg_buffer['adapter_eval'].keys():
            self.msg_buffer['adapter_eval'][rnd] = dict()

        self.msg_buffer['adapter_eval'][rnd][sender] = \
            [(i, content[f'adapter_{i}_avg_loss'])
             for i in range(self._cfg.llm.adapter.count)]
        self.msg_buffer['adapter_eval'][rnd][sender] = \
            sorted(self.msg_buffer['adapter_eval'][rnd][sender],
                   key=lambda x: x[1])

        return self.check_and_grouping()

    def check_and_grouping(self):
        if 'adapter_eval' not in self.msg_buffer.keys() or \
                len(self.msg_buffer['adapter_eval'].keys()) == 0:
            return False

        buffer = self.msg_buffer['adapter_eval']
        cur_round = max(buffer.keys())
        cur_buffer = buffer[cur_round]

        if len(cur_buffer) < self.client_num:
            return False


        #각 클라에 ‘선호 어댑터 순회자’(iterator) 부여->이제 이터레이터에서 next() 할 때마다 그 클라의 다음 선호 어댑터를 받는다.
 
        for sender in cur_buffer.keys():
            cur_buffer[sender] = iter(cur_buffer[sender])

        num_adap = self._cfg.llm.adapter.count
        self.adapter_grouping = dict()

        #라운드-로빈/용량 제한 기반 할당
        # 각 클라에게 “좋아하는 어댑터”를 우선 주되, 어댑터별 수용 인원 상한을 max_size로 엄격히 관리해 균형 잡힌 그룹 크기를 만든다.
        # 과밀 어댑터는 잘라서 확정하고, 남은 클라는 그 다음 반복에서 자기 선호 목록의 다음 어댑터로 시도한다.
        # 이 과정을 반복하면, 대략 공평한 크기의 그룹이 얻어진다(= 각 어댑터당 클라 비슷한 수).        
        adapter_grouping = {i: [] for i in range(num_adap)}
        senders = [sender for sender in cur_buffer.keys()]
        random.shuffle(senders)
        unassigned_client_num = len(senders)
        while unassigned_client_num > 0:
            num_finished = len(self.adapter_grouping)
            max_size = math.ceil(unassigned_client_num /
                                 (num_adap - num_finished))

            # step 1: Assign to the adapter where the clients
            # well performs
            for sender in senders:
                adap_idx, loss = next(cur_buffer[sender])
                while adap_idx not in adapter_grouping:
                    adap_idx, loss = next(cur_buffer[sender])
                adapter_grouping[adap_idx].append(sender)

            # step 2: Find the adapter with the most clients
            max_adap_idx_size = [0, 0]
            for adap_idx, candidates in adapter_grouping.items():
                if len(candidates) > max_adap_idx_size[1]:
                    max_adap_idx_size = [adap_idx, len(candidates)]

            # step 3: If the number of candidates is greater than
            # max_size, preserve the first max_size
            adap_idx = max_adap_idx_size[0]
            candidates = adapter_grouping[adap_idx][:max_size]

            # step 4: update the senders list, remove the selected
            # adapter from adapter_grouping
            senders = adapter_grouping[adap_idx][max_size:]
            self.adapter_grouping[adap_idx] = candidates
            adapter_grouping.pop(adap_idx)
            unassigned_client_num -= len(self.adapter_grouping[adap_idx])
            logger.info(f'Adapter {adap_idx} is done with the clients '
                        f'{self.adapter_grouping[adap_idx]}')
            




        # broadcast the new grouping info to all clients
        # Grouping 결과 브로드캐스트 & 학습 재개
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
