import logging
import copy
import os
import sys

import numpy as np
import pickle

from federatedscope.core.monitors.early_stopper import EarlyStopper
from federatedscope.core.message import Message
from federatedscope.core.communication import StandaloneCommManager, \
    StandaloneDDPCommManager, gRPCCommManager
from federatedscope.core.auxiliaries.aggregator_builder import get_aggregator
from federatedscope.core.auxiliaries.sampler_builder import get_sampler
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    Timeout, merge_param_dict, add_prefix_to_path, get_ds_rank
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.secret_sharing import AdditiveSecretSharing
from federatedscope.core.workers.base_server import BaseServer


import gc  
import torch 


logger = logging.getLogger(__name__)
if get_ds_rank() == 0:
    logger.setLevel(logging.INFO)


class Server(BaseServer): #클라이언트 등록, 메시지 수신 및 응답 처리, 모델 평가용 사본 생성 등 실제 FL 서버 기능 구현
    """
    The Server class, which describes the behaviors of server in an FL \
    course. The behaviors are described by the handled functions (named as \
    ``callback_funcs_for_xxx``).

    Arguments:
        ID: The unique ID of the server, which is set to 0 by default
        state: The training round
        config: the configuration
        data: The data owned by the server (for global evaluation)
        model: The model used for aggregation
        client_num: The (expected) client num to start the FL course
        total_round_num: The total number of the training round
        device: The device to run local training and evaluation

    Attributes:
        ID: ID of worker
        state: the training round index
        model: the model maintained locally
        cfg: the configuration of FL course, \
            see ``federatedscope.core.configs``
        mode: the run mode for FL, ``distributed`` or ``standalone``
        monitor: monite FL course and record metrics, \
            see ``federatedscope.core.monitors.monitor.Monitor``
        trainer: instantiated trainer, see ``federatedscope.core.trainers``
        best_results: best results ever seen
        history_results: all evaluation results
        early_stopper: determine when to early stop, \
            see ``federatedscope.core.monitors.early_stopper.EarlyStopper``
        aggregators: a protocol for aggregate all clients' model(s), see \
            ``federatedscope.core.aggregators``
        sample_client_num: number of client aggregated in each round
        msg_buffer: dict buffer for storing message
        staled_msg_buffer: list buffer for storing staled message
        comm_manager: manager for communication, \
            see ``federatedscope.core.communication``
    """
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
                 unseen_clients_id=None,
                 **kwargs):
        super(Server, self).__init__(ID, state, config, model, strategy)
        # Register message handlers, 'train', 'eval' 등 메시지 타입에 대응되는 핸들러 등록
        self._register_default_handlers()

        # Un-configured worker
        if config is None:
            return

        self.data = data
        self.device = device
        self.best_results = dict() #"client_best_individual", "client_summarized_avg", "client_summarized_weighted_avg", "client_summarized_fairness" key로 구성될 예정. 통계량들만 있음.
        self.history_results = dict()  #아래의 formatted_logs_all_set form을 eval 라운드 마다 누적해서 쌓을 것임. key: 'Role', 'Round', Results_weighted_avg', Results_avg', Results_fairness', Results_raw'
        #{'Role': 'Server #', 'Round': 24, 'Results_weighted_avg': {'val_total': 124.9622641509434, 'val_loss': 99.96387704124663, 'val_avg_loss': 0.6407083052442673, 'val_acc': 0.6385323871357391, 'test_total': 40.0, 'test_loss': 25.894947483854473, 'test_avg_loss': 0.6473736870963619, 'test_acc': 0.6183962264150943}, 'Results_avg': {'val_total': 124.9622641509434, 'val_loss': 80.0643604836374, 'val_avg_loss': 0.6450719820537335, 'val_acc': 0.6339984738853919, 'test_total': 40.0, 'test_loss': 25.894947483854473, 'test_avg_loss': 0.647373687096362, 'test_acc': 0.6183962264150943}, 'Results_fairness': {'val_total': 124.9622641509434, 'test_total': 40.0, 'val_loss_std': 40.26031761787951, 'val_loss_bottom_decile': 17.608741760253906, 'val_loss_top_decile': 130.40792846679688, 'val_loss_min': 6.489111423492432, 'val_loss_max': 143.4759521484375, 'val_loss_bottom10%': 9.588900661468506, 'val_loss_top10%': 135.79025268554688, 'val_loss_cos1': 0.8934065944700241, 'val_loss_entropy': 3.8212773654406504, 'val_avg_loss_std': 0.060552038285135466, 'val_avg_loss_bottom_decile': 0.5847968344362626, 'val_avg_loss_top_decile': 0.7173797607421875, 'val_avg_loss_min': 0.5036511739095052, 'val_avg_loss_max': 0.8567123413085938, 'val_avg_loss_bottom10%': 0.5518826093219575, 'val_avg_loss_top10%': 0.7636649671131263, 'val_avg_loss_cos1': 0.9956232405959634, 'val_avg_loss_entropy': 3.965981491239061, 'val_acc_std': 0.07289379505770262, 'val_acc_bottom_decile': 0.5540540540540541, 'val_acc_top_decile': 0.7192982456140351, 'val_acc_min': 0.36363636363636365, 'val_acc_max': 0.7666666666666667, 'val_acc_bottom10%': 0.4747722778585586, 'val_acc_top10%': 0.7364474950001266, 'val_acc_cos1': 0.9934552236834064, 'val_acc_entropy': 3.96327855153204, 'test_loss_std': 2.737384274028946, 'test_loss_bottom_decile': 23.15328025817871, 'test_loss_top_decile': 29.409709930419922, 'test_loss_min': 19.09391975402832, 'test_loss_max': 31.324756622314453, 'test_loss_bottom10%': 21.632117462158202, 'test_loss_top10%': 30.28991985321045, 'test_loss_cos1': 0.9944589750905024, 'test_loss_entropy': 3.9646973764426416, 'test_avg_loss_std': 0.06843460685072364, 'test_avg_loss_bottom_decile': 0.5788320064544678, 'test_avg_loss_top_decile': 0.7352427482604981, 'test_avg_loss_min': 0.47734799385070803, 'test_avg_loss_max': 0.7831189155578613, 'test_avg_loss_bottom10%': 0.5408029365539551, 'test_avg_loss_top10%': 0.7572479963302613, 'test_avg_loss_cos1': 0.9944589750905025, 'test_avg_loss_entropy': 3.964697376459541, 'test_acc_std': 0.0828659612442278, 'test_acc_bottom_decile': 0.525, 'test_acc_top_decile': 0.725, 'test_acc_min': 0.4, 'test_acc_max': 0.825, 'test_acc_bottom10%': 0.46499999999999997, 'test_acc_top10%': 0.7541666666666668, 'test_acc_cos1': 0.9911409426294011, 'test_acc_entropy': 3.961152079464096}}

        """
        ipdb> self._cfg.early_stop.patience
        0
        ipdb> self._cfg.early_stop.improve_indicator_mode
        'best'
        ipdb> self._monitor.the_larger_the_better
        False
        ipdb> self._cfg.early_stop.delta
        0.0
        ipdb> self._cfg.early_stop.patience
        0        
        
        """
        self.early_stopper = EarlyStopper(
            self._cfg.early_stop.patience, self._cfg.early_stop.delta,
            self._cfg.early_stop.improve_indicator_mode,
            self._monitor.the_larger_the_better)
        

        if self._cfg.federate.share_local_model \
                and not self._cfg.federate.process_num > 1 \
                and not self._cfg.llm.deepspeed.use \
                and not self._cfg.llm.accelerator.use:
            # if self._cfg.train.is_enable_half:
            #     model.to(torch.bfloat16)
            # put the model to the specified device
            model.to(device)

        # Build aggregator
        self.aggregator = get_aggregator(self._cfg.federate.method,
                                         model=model,
                                         device=device,
                                         online=self._cfg.federate.online_aggr,
                                         config=self._cfg)
        #self._cfg.federate.method=fedvag: FedBiscuit->MultiLoRAAvgAggregator, ClientsAvgAggregator
        #self._cfg.federate.method=local only-> NoCommunicationAggregator
        
        self._deferred_after_eval = None   # 다음 라운드 시작을 평가 완료까지 연기할 때 저장할 값
        
        self.model_num = config.model.model_num_per_trainer #1
        
        self.models = [self.model]
        self.aggregators = [self.aggregator]


        # Initialize the number of joined-in clients
        self._client_num = client_num
        self._total_round_num = total_round_num
        self.sample_client_num = int(self._cfg.federate.sample_client_num)


        self.join_in_client_num = 0 #join in하는 client 갯수
        self.join_in_info = dict() #클라이언트가 보낸 join_in_info를 저장한 딕셔너리


        # the unseen clients indicate the ones that do not contribute to FL
        # process by training on their local data and uploading their local
        # model update. The splitting is useful to check participation
        # generalization gap in
        # [ICLR'22, What Do We Mean by Generalization in Federated Learning?]
        self.unseen_clients_id = [] if unseen_clients_id is None \
            else unseen_clients_id

        # Server state
        self.is_finish = False

        # Sampler
        # if self._cfg.federate.sampler in ['uniform']: #True
        #     self.sampler = get_sampler(
        #         sample_strategy=self._cfg.federate.sampler,
        #         client_num=self.client_num,
        #         client_info=None) #UniformSampler
        #      #in #indi# #
        #     # Initialize the sampler when loading from last checkpoint
        #     if self.sample_client_num > 0:
        #         for _ in range(self.state): #self.state=0
        #             # idle 상태인 클라이언트 중 size명을 뽑아, 그들의 상태를 working(0) 으로 전환
        #             temp = self.sampler.sample(size=self.sample_client_num)
        #             # 방금 뽑힌 size명을 즉시 다시 idle(1) 상태로 복구
        #             self.sampler.change_state(temp, 'idle')
        # else:
        #     # Some type of sampler would be instantiated in trigger_for_start,
        #     # since they need more information
        #     self.sampler = None

        if self._cfg.federate.sampler in ['uniform', 'cluster']:
            self.sampler = get_sampler(
                sample_strategy=self._cfg.federate.sampler,
                client_num=self.client_num,
                client_info=None,
                config=self._cfg
            )

        else:
            # Some type of sampler would be instantiated later if needed
            self.sampler = None



        # Current Timestamp
        self.cur_timestamp = 0  #asyncronous에서 쓰이는 거 같음
        self.deadline_for_cur_round = 1 # 현 라운드가 끝나야 하는 시점을 cur_timestamp 기준으로 지정한 값입니다. 기본값 1 이지만, 비동기(cfg.asyn.use=True)일 때는 매 라운드마다 time_budget을 더해 갱신됩니다.

        # Staleness toleration
        self.staleness_toleration = self._cfg.asyn.staleness_toleration if \
            self._cfg.asyn.use else 0  #0으로 적용됨. 몇 라운드까지 “지연된(stale)” 메시지를 허용할지 설정합니다. 동기 모드(asyn.use=False)이면 0으로, 즉 절대 지연 메시지를 받지 않습니다.
        self.dropout_num = 0 #너무 오래된 메시지( round < state - staleness_toleration )를 서버가 버릴 때마다 1씩 올려주는 카운터입니다.

        # Device information
        self.resource_info = kwargs['resource_info'] \
            if 'resource_info' in kwargs else None
        self.client_resource_info = kwargs['client_resource_info'] \
            if 'client_resource_info' in kwargs else None

        # Initialize communication manager and message buffer
        self.msg_buffer = {'train': dict(), 'eval': dict()}  #클라이언트로부터 수신한 메시지들을 라운드별로 저장. Aggregation 시점까지 메시지를 누적 관리하며, 수가 충분해지면 check_and_move_on() 수행

        self.staled_msg_buffer = list() #지연 허용치” 이내에 왔지만 현재 라운드보다 이전 에 속한 메시지를 임시로 모아 두는 리스트. 나중에 집계에 포함하거나 드롭합니다.
        if self.mode == 'standalone': # 프로세스 수 >1 이면 DDP, 1이면 단일 프로세스 큐 기반
            comm_queue = kwargs.get('shared_comm_queue', None)
            if self._cfg.federate.process_num > 1:
                id2comm = kwargs.get('id2comm', None)
                self.comm_manager = StandaloneDDPCommManager(
                    comm_queue=comm_queue,
                    monitor=self._monitor,
                    id2comm=id2comm)
            else:
                self.comm_manager = StandaloneCommManager(
                    comm_queue=comm_queue, monitor=self._monitor)
                

        elif self.mode == 'distributed': # gRPC 서버로 동작
            host = kwargs['host']
            port = kwargs['port']
            self.comm_manager = gRPCCommManager(host=host,
                                                port=port,
                                                client_num=client_num,
                                                cfg=self._cfg.distribute)
            logger.info('Server: Listen to {}:{}...'.format(host, port))

        # inject noise before broadcast
        self._noise_injector = None #이 필드를 나중에 register_noise_injector() 로 설정해 두면, broadcast_model_para() 할 때 각 파라미터에 추가적인 노이즈(프라이버시/보안 목적)를 뿌려 줄 수 있습니다.

    def _current_required_sample_num(self, round_idx: int = None) -> int:
        """
        cluster 샘플러일 때 현 라운드의 필요 응답 수(=s[i]) 반환.
        cluster가 아니면 fallback으로 sample_client_num 반환.
        """
        from bisect import bisect_right
        if round_idx is None:
            round_idx = self.state
        if getattr(self._cfg.federate, 'sampler', 'uniform') != 'cluster':
            base = int(self._cfg.federate.sample_client_num)
            return base if base > 0 else len(self.comm_manager.get_neighbors().keys())

        round_ends = getattr(self._cfg.llm.adapter, 'round_ends', None)
        s_per      = getattr(self._cfg.llm.adapter, 'sample_num_per_adapter', None)
        if not round_ends or not s_per:
            base = int(self._cfg.federate.sample_client_num)
            return base if base > 0 else len(self.comm_manager.get_neighbors().keys())

        aidx = bisect_right([int(x) for x in round_ends], int(round_idx))  # 0..C-1 (또는 C==eval)
        if aidx >= len(s_per):     # 평가 라운드 등
            aidx = len(s_per) - 1
        return int(s_per[aidx])




    @property
    def client_num(self):
        return self._client_num

    @client_num.setter
    def client_num(self, value):
        self._client_num = value

    @property
    def total_round_num(self):
        return self._total_round_num

    @total_round_num.setter
    def total_round_num(self, value):
        self._total_round_num = value

    def register_noise_injector(self, func):
        self._noise_injector = func

    def run(self): #이거 안쓰이는 거 같음.
        """
        To start the FL course, listen and handle messages (for distributed \
        mode).
        """

        # Begin: Broadcast model parameters and start to FL train
        while self.join_in_client_num < self.client_num: #기대하는 수(client_num)만큼 모든 클라이언트가 join_in 메시지를 보내고 등록될 때까지 기다립니다.
            #receive()로 메시지를 받고, 메시지 타입별로 등록된 콜백(callback_funcs_for_join_in)을 실행해 join_in_client_num을 늘립니다.
            msg = self.comm_manager.receive()
            self.msg_handlers[msg.msg_type](msg) 

        # Running: listen for message (updates from clients),
        # aggregate and broadcast feedbacks (aggregated model parameters)

        min_received_num = self._cfg.asyn.min_received_num \
            if self._cfg.asyn.use else self._cfg.federate.sample_client_num  #매 라운드마다 sample될 클라이언트 수로 지정 # 동기/비동기 최소 응답 수 결정

        num_failure = 0
        time_budget = self._cfg.asyn.time_budget if self._cfg.asyn.use else -1  #asyn 모드가 아니라서 -1로 지정, # 비동기일 때 타임아웃 한계
        with Timeout(time_budget) as time_counter: #비동기 설정 시 “정해진 시간” (time_budget 초) 내에 충분한 업데이트를 못 받으면 타임아웃 처리
            while self.state <= self.total_round_num:
                try: #Timeout(-1) 의 동작: time_budget < 0 이면 타이머를 아예 시작하지 않고 time_counter 에서도 시간을 체크하지 않기 때문에 절대 TimeoutError 가 던져지지 않습니다.
                    msg = self.comm_manager.receive()
                    #callback_funcs_model_para: 모델 파라미터 수집 → 버퍼 저장 → check_and_move_on() 호출
                    #callback_funcs_for_metrics: 평가 결과 수집 → check_and_move_on(check_eval_result=True)
                    move_on_flag = self.msg_handlers[msg.msg_type](msg)
                    if move_on_flag: #“이번 라운드(또는 평가 단계)를 넘길 준비 완료” 신호 → 타이머 리셋
                        time_counter.reset()
                except TimeoutError: #무시해도 됨. 비동기 모드가 아니라서
                    logger.info('Time out at the training round #{}'.format(
                        self.state))
                    move_on_flag_eval = self.check_and_move_on(
                        min_received_num=min_received_num,
                        check_eval_result=True)
                    move_on_flag = self.check_and_move_on(
                        min_received_num=min_received_num)
                    if not move_on_flag and not move_on_flag_eval:
                        num_failure += 1
                        # Terminate the training if the number of failure
                        # exceeds the maximum number (default value: 10)
                        if time_counter.exceed_max_failure(num_failure):
                            logger.info(f'----------- Training fails at round '
                                        f'#{self.state}-------------')
                            break

                        # Time out, broadcast the model para and re-start
                        # the training round
                        logger.info(
                            f'----------- Re-starting the training round ('
                            f'Round #{self.state}) for {num_failure} time '
                            f'-------------')
                        # TODO: Clean the msg_buffer
                        if self.state in self.msg_buffer['train']:
                            self.msg_buffer['train'][self.state].clear()

                        self.comm_manager.send(
                            msg_type='model_para',
                            sample_client_num=self.sample_client_num)
                    else:
                        num_failure = 0
                    time_counter.reset()

        self.terminate(msg_type='finish') #self.state > self.total_round_num 가 되면 루프를 빠져나온 뒤 terminate() 호출 → 마지막 finish 메시지를 모든 클라이언트에게 보내고 서버 종료 처리

    def check_and_move_on(self,
                          check_eval_result=False,
                          min_received_num=None):#클라이언트 메시지가 도착할 때마다 호출되어, 메시지 수가 충분한지 판단하고 집계하고 다음 라운드로 넘어갈지 결정
        """ 
        To check the message_buffer. When enough messages are receiving, \
        some events (such as perform aggregation, evaluation, and move to \
        the next training round) would be triggered.

        Arguments:
            check_eval_result (bool): If True, check the message buffer for \
                evaluation; and check the message buffer for training \
                otherwise.
            min_received_num: number of minimal received message, used for \
                async mode
        """

        #check_eval_result=False : 학습 메시지(model_para)인지, check_eval_result=True : 평가 메시지(metrics)인지

        # 기존 assert min_received_num <= self.sample_client_num  ← -1일 때 깨짐
        effective_sample_size = self.sample_client_num if self.sample_client_num > 0 \
                                else len(self.comm_manager.get_neighbors().keys())


        if min_received_num is None:
            if self._cfg.asyn.use:
                min_received_num = self._cfg.asyn.min_received_num
            else:
                # min_received_num = self._cfg.federate.sample_client_num #이걸로 결정.
                min_received_num = self._current_required_sample_num()


        assert min_received_num <= effective_sample_size

        if check_eval_result and self._cfg.federate.mode.lower(
        ) == "standalone": #eval 단계에서는 전체 클라이언트들에 대해 평가
            
            # in evaluation stage and standalone simulation mode, we assume
            # strong synchronization that receives responses from all clients
            min_received_num = len(self.comm_manager.get_neighbors().keys())#현재 서버가 인식하고 있는 모든 클라이언트들의 ID 목록: self.comm_manager.get_neighbors().keys()

        move_on_flag = True  # 서버가 최종적으로 “다음 라운드로 넘어갑니다(move on)”를 반환할지 여부

        # round or finishing the evaluation
        if self.check_buffer(self.state, min_received_num, check_eval_result): #min_recieved_num만큼 버퍼가 찬 상황
            if not check_eval_result: #Train 상황
                # Receiving enough feedback in the training process
                aggregated_num = self._perform_federated_aggregation() #매 라운드마다 select 된 클라이언트 수
                self.state += 1


                if self.state % self._cfg.eval.freq == 0 and self.state != \
                        self.total_round_num:  #evaluation 실행
                    #  Evaluate
                    logger.info(f'Server: Starting evaluation at the end '
                                f'of round {self.state - 1}.')
                    self.eval() #server 모델을 모든 클라이언트들한테 보냄. self.broadcast_model_para(msg_type='evaluate', filter_unseen_clients=False) 적용됨.

                    self._deferred_after_eval = aggregated_num 

                    move_on_flag = False # 평가 결과를 기다리는 동안 move_on 보류. 이 method는 move_on_flag를 반환하기에 False를 반환.

                else:
                    if self.state < self.total_round_num: #다음 라운드 실행
                        # 바로 다음 라운드 시작
                        logger.info(
                            f'----------- Starting a new training round (Round #{self.state}) -------------'
                        )

                        # Clean the msg_buffer
                        self.msg_buffer['train'][self.state - 1].clear() #self.msg_buffer['train'][self.state - 1]를 빈 딕셔너리 dict()로 만들어줌.
                        self.msg_buffer['train'][self.state] = dict()
                        self.staled_msg_buffer.clear()

                        self._start_new_training_round(aggregated_num)
                    else:
                        # Final Evaluate
                        # logger.info('Server: Training is finished! Starting evaluation.')
                        # self.eval() #server 모델을 모든 클라이언트들한테 보냄. self.broadcast_model_para(msg_type='evaluate', filter_unseen_clients=False) 적용됨.


                        # ★ 평가 없이도 'final_' ckpt 저장
                        if self._cfg.federate.save_to != '' and self.ds_rank == 0:
                            self.aggregator.save_model(
                                add_prefix_to_path('final_', self._cfg.federate.save_to),
                                self.state
                            )
                        logger.info('Server: Training is finished! (skip final evaluation)')
                        self.is_finish = True
                        self.terminate(msg_type='finish')


                
            else:
                # ===== 평가 메시지 수집 완료 → 병합/저장 =====
                self._merge_and_format_eval_results()
                if self.state >= self.total_round_num:
                    self.is_finish = True

                # 연기된 다음 라운드가 있고, 아직 최종 라운드 전이면 지금 시작
                if (self._deferred_after_eval is not None) and (self.state < self.total_round_num):
                    aggregated_num = self._deferred_after_eval
                    self._deferred_after_eval = None
                    logger.info(
                        f'----------- Starting a new training round (Round #{self.state}) -------------'
                    )
                    # Clean the msg_buffer
                    self.msg_buffer['train'][self.state - 1].clear() #self.msg_buffer['train'][self.state - 1]를 빈 딕셔너리 dict()로 만들어줌.
                    self.msg_buffer['train'][self.state] = dict()
                    self.staled_msg_buffer.clear()

                    self._start_new_training_round(aggregated_num)



        else: # 아직 메시지가 충분치 않으면 move_on_flag=False
            move_on_flag = False


        return move_on_flag

    def check_and_save(self):
        """
        To save the results and save model after each evaluation, and check \
        whether to early stop.
        """

        # early stopping 판단. self.history_results: eval 라운드별 서버 formatted_eval_res 로그를 누적한 것 

        """
        self.history_results: 아래의 형태. 
        formatted_logs = {
        "Results_avg": {"val_acc": 0.8166, "val_loss": 0.5, .....},
        "Results_weighted_avg": {...},
        "Results_fairness": {...},
        "Results_raw": {"val_acc": [0.8, 0.75, 0.9],
                        "val_loss": [0.5, 0.55, 0.45], ...}
        }               

        """


        if "Results_weighted_avg" in self.history_results and \
                self._cfg.eval.best_res_update_round_wise_key in \
                self.history_results['Results_weighted_avg']: #True
            should_stop = self.early_stopper.track_and_check(
                self.history_results['Results_weighted_avg'][
                    self._cfg.eval.best_res_update_round_wise_key])  #self.patience=0이라 self.__track_and_check_dummy로 되어 should_stop은 언제나 False인 상황.
            

        #early stop하기로 결정된 경우: 수렴 통지 + 상태 고정

        if should_stop: #False라 실행 안됨.
            self._monitor.global_converged()
            self.comm_manager.send(
                Message(
                    msg_type="converged",
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    timestamp=self.cur_timestamp,
                    state=self.state,
                ))#모든 이웃(보통 클라이언트들)에게 "converged" 메시지 브로드캐스트 → 로컬 업데이트 중단 신호.
            self.state = self.total_round_num + 1

        #주기적 체크포인트 저장 (중간 저장)
        if self.state != self.total_round_num and \
                self.state % self._cfg.federate.save_freq == 0 and \
                self._cfg.federate.save_freq > 0:
            path = add_prefix_to_path(f'{self.state}_',
                                      self._cfg.federate.save_to)
            if self.ds_rank == 0:
                self.aggregator.save_model(path, self.state) #현재 글로벌 모델을 그 시점 메타 정보와 함께 저장한다고 보면 됨.

        #최종 마무리(베스트/클라이언트별 결과 저장, 종료)
        if should_stop or self.state == self.total_round_num: #self.state == self.total_round_num 일때만 실행 될 예정
            logger.info('Server: Final evaluation is finished! Starting '
                        'merging results.')
            # last round or early stopped
            self.save_best_results() #서버 관점에서 베스트 결과를 표준 포맷/경로로 저장(+ 옵션에 따라 final_ 모델도 저장).
            if not self._cfg.federate.make_global_eval: #실행 됨.
                self.save_client_eval_results() #최근 라운드의 클라이언트별 평가 레코드를 eval_results.log로 남김.
            self.terminate(msg_type='finish') #학습 종료 시그널.

        #평가 메시지 버퍼 정리(메모리 관리)
        # Clean the clients evaluation msg buffer
        if not self._cfg.federate.make_global_eval: #실행 됨.
            round = max(self.msg_buffer['eval'].keys())
            self.msg_buffer['eval'][round].clear() #가장 최근 라운드의 eval 버퍼를 비움.

        #분산 모드 루프 종료 보정

        if self.state == self.total_round_num:
            # break out the loop for distributed mode
            self.state += 1




    def _perform_federated_aggregation(self): #model aggregation 수행.
        """
        Perform federated aggregation and update the global model
        (메모리 절약 – in-place parameter copy + GPU 메모리 즉시 해제)
        """
        logger.info(f"[in-place aggregation 시작] round={self.state}")
        train_msg_buffer = self.msg_buffer['train'][self.state]
        aggregated_num   = 0

        for model_idx in range(self.model_num): #self.model_num=1인 상황.
            model      = self.models[model_idx]
            aggregator = self.aggregators[model_idx]

            # ① 클라이언트 피드백 수집
            msg_list = []
            for _, content in train_msg_buffer.items(): #-: client index, content: (sample_size, model)
                if self.model_num == 1: #True
                    msg_list.append(content) #content= (sample_size, model)
                else:
                    n, multi_states = content
                    msg_list.append((n, multi_states[model_idx]))
            aggregated_num = len(msg_list) #aggregate할 갯수.

            # ② Aggregator 호출
            agg_info     = {
                "client_feedback": msg_list,
                "recover_fun"    : self.recover_fun, #self.recover_fun=None 
            }
            result_state = aggregator.aggregate(agg_info)  # GPU 텐서 반환, avg된 모델.

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




    def _start_new_training_round(self, aggregated_num=0):
        """
        The behaviors for starting a new training round
        """

        # ───── 라운드 종료 직후 메모리/버퍼 정리 ─────
        prev_round = self.state            # self.state는 이미 1 증가하기 전 값
        if prev_round in self.msg_buffer['train']:
            self.msg_buffer['train'][prev_round].clear()
        torch.cuda.empty_cache()
        gc.collect()

        # for synchronous training
        self.broadcast_model_para(msg_type='model_para',
                                    sample_client_num=self.sample_client_num)

    def _merge_and_format_eval_results(self): #eval 라운드별 서버 formatted_eval_res 로그를 누적해서 나중에 early stop나 요약용으로 씀.
        """
        The behaviors of server when receiving enough evaluating results
        """
        # Get all the message & aggregate
        # 모인 평가 결과 합치기 → 포맷 만들기. 여기서 클라별 metric들을 모아서 weighted_avg / avg / fairness / raw 같은 집계 포맷을 생성.
        formatted_eval_res = \
            self.merge_eval_results_from_all_clients()
        

        #라운드별 서버 formatted_eval_res 로그를 누적해서 나중에 early stop나 요약용으로 씀.
        self.history_results = merge_dict_of_results(self.history_results,
                                                     formatted_eval_res)
        
            
        #후속 정리/종료 판단
        self.check_and_save()

    def save_best_results(self):
        """
        To Save the best evaluation results.
        """
        # Save final round model
        if self._cfg.federate.save_to != '' and self.ds_rank == 0: #self.ds_rank == 0인 상황.(distributed가 아니라서) 
            self.aggregator.save_model(
                add_prefix_to_path('final_', self._cfg.federate.save_to),
                self.state)
        formatted_best_res = self._monitor.format_eval_res(
            results=self.best_results, #"client_best_individual", "client_summarized_avg", "client_summarized_weighted_avg", "client_summarized_fairness" key로 구성될 예정. 통계량들만 있음. 
            rnd="Final",
            role='Server #',
            forms=["raw"],
            return_raw=True) #eval_results.log에 저장. return_raw=True라서 round_formatted_results_raw를 반환하고 이 dict에  Results_raw key만 반영하고 value로는  self.best_results 반영.
        ##formatted_best_res={'Role': 'Server #', 'Round': 'Final', 'Results_raw': self.best_results}.
        #{'Role': 'Server #', 'Round': 'Final', 'Results_raw': {'client_best_individual': {'test_loss': 17.79407501220703, 'val_total': 11.0, 'val_loss': 7.473792552947998, 'val_avg_loss': 0.5097981135050456, 'val_acc': 0.78125, 'test_total': 40.0, 'test_avg_loss': 0.4448518753051758, 'test_acc': 0.825}, 'client_summarized_weighted_avg': {'test_loss': 25.376989472587155, 'val_total': 124.9622641509434, 'val_loss': 96.82593974400444, 'val_avg_loss': 0.6207614610261037, 'val_acc': 0.6503095274044994, 'test_total': 40.0, 'test_avg_loss': 0.6344247368146788, 'test_acc': 0.6311320754716981}, 'client_summarized_avg': {'test_loss': 25.376989472587155, 'val_total': 124.9622641509434, 'val_loss': 77.57175766746953, 'val_avg_loss': 0.6247532087564949, 'val_acc': 0.6440690982991253, 'test_total': 40.0, 'test_avg_loss': 0.6344247368146788, 'test_acc': 0.6311320754716983}, 'client_summarized_fairness': {'test_loss_entropy': 3.945508185999201, 'test_loss_cos1': 0.9764950031069862, 'test_loss_top10%': 100.19970830281575, 'test_loss_bottom10%': 45.53137702941895, 'test_loss_max': 103.08953094482422, 'test_loss_min': 31.37982749938965, 'test_loss_top_decile': 94.82197570800781, 'test_loss_bottom_decile': 54.21096420288086, 'test_loss_std': 15.919672164015168, 'val_total': 124.9622641509434, 'test_total': 40.0, 'val_loss_std': 112.06515834103715, 'val_loss_bottom_decile': 36.23653030395508, 'val_loss_top_decile': 348.406005859375, 'val_loss_min': 14.605098724365234, 'val_loss_max': 411.28289794921875, 'val_loss_bottom10%': 24.78562355041504, 'val_loss_top10%': 383.567626953125, 'val_loss_cos1': 0.8852770096599365, 'val_loss_entropy': 3.8092242959266636, 'val_avg_loss_std': 0.36090686972101704, 'val_avg_loss_bottom_decile': 1.312889780317034, 'val_avg_loss_top_decile': 2.0635759508287586, 'val_avg_loss_min': 0.9009682337443033, 'val_avg_loss_max': 3.1638114235617896, 'val_avg_loss_bottom10%': 1.0787674480792, 'val_avg_loss_top10%': 2.3460283492285394, 'val_avg_loss_cos1': 0.9786446447696837, 'val_avg_loss_entropy': 3.948730288988393, 'val_acc_std': 0.06604156149737403, 'val_acc_bottom_decile': 0.5782312925170068, 'val_acc_top_decile': 0.7, 'val_acc_min': 0.36363636363636365, 'val_acc_max': 0.8571428571428571, 'val_acc_bottom10%': 0.506494656712048, 'val_acc_top10%': 0.736959574545191, 'val_acc_cos1': 0.9947468136553479, 'val_acc_entropy': 3.964685440472374, 'test_avg_loss_std': 0.3979918041003792, 'test_avg_loss_bottom_decile': 1.3552741050720214, 'test_avg_loss_top_decile': 2.3705493927001955, 'test_avg_loss_min': 0.7844956874847412, 'test_avg_loss_max': 2.5772382736206056, 'test_avg_loss_bottom10%': 1.1382844257354736, 'test_avg_loss_top10%': 2.5049927075703944, 'test_avg_loss_cos1': 0.9764950031069864, 'test_avg_loss_entropy': 3.9455081860266317, 'test_acc_std': 0.06117180749364753, 'test_acc_bottom_decile': 0.55, 'test_acc_top_decile': 0.7, 'test_acc_min': 0.5, 'test_acc_max': 0.8, 'test_acc_bottom10%': 0.5199999999999999, 'test_acc_top10%': 0.7333333333333334, 'test_acc_cos1': 0.9952868697490959, 'test_acc_entropy': 3.9655586896709516}}}

        logger.info(formatted_best_res)
        self._monitor.save_formatted_results(formatted_best_res)

    def save_client_eval_results(self):
        """
        save the evaluation results of each client when the fl course \
        early stopped or terminated
        """
        rnd = max(self.msg_buffer['eval'].keys())
        eval_msg_buffer = self.msg_buffer['eval'][rnd]

        with open(os.path.join(self._cfg.outdir, "eval_results.log"),
                  "a") as outfile:
            for client_id, client_eval_results in eval_msg_buffer.items():
                ##formatted_best_res={'Role': 'Client #1', 'Round': 250, 'Results_raw': client_eval_results}.
                #{'Role': 'Client #1', 'Round': 250, 'Results_raw': {'val_total': 146, 'val_loss': 182.99884033203125, 'val_avg_loss': 1.2534167146029538, 'val_acc': 0.7191780821917808, 'test_total': 40, 'test_loss': 31.37982749938965, 'test_avg_loss': 0.7844956874847412, 'test_acc': 0.8}}
                formatted_res = self._monitor.format_eval_res(
                    client_eval_results,
                    rnd=self.state,
                    role='Client #{}'.format(client_id),
                    return_raw=True)
                logger.info(formatted_res)
                outfile.write(str(formatted_res) + "\n")

    def merge_eval_results_from_all_clients(self): # 같은 라운드에서 각 클라이언트가 보낸 평가 결과(metrics)를 모아 합치는 함수입니다.
        """
        Merge evaluation results from all clients, update best, \
        log the merged results and save them into eval_results.log

        Returns:
            the formatted merged results
        """

        """
        각 클라이언트는 {"val_acc": 0.8, "val_loss": 0.5}처럼 자신만의 성능 결과를 서버에 보냄.

        서버는 이걸 모아 클라이언트별 리스트로 만들고,
        나중에 Results_avg, Results_weighted_avg, Results_raw 등 집계 버전을 생성.    
        """


        #이번 라운드 eval 버퍼 긁어오기
        round = max(self.msg_buffer['eval'].keys())
        eval_msg_buffer = self.msg_buffer['eval'][round]

        """
        eval_msg_buffer = {
            1: {"val_acc": 0.8, "val_loss": 0.5},
            2: {"val_acc": 0.75, "val_loss": 0.55},
            3: {"val_acc": 0.9, "val_loss": 0.45},
        }
        """


        #참여 클라와 미참여(unseen) 클라 분리. unseen_clients_id가 비어있으면 → 전부 eval_res_participated_clients로 감
        eval_res_participated_clients = []
        eval_res_unseen_clients = []

        for client_id in eval_msg_buffer:
            if eval_msg_buffer[client_id] is None: 
                continue
            if client_id in self.unseen_clients_id:
                eval_res_unseen_clients.append(eval_msg_buffer[client_id])
            else: #여기만 걸림.
                eval_res_participated_clients.append(
                    eval_msg_buffer[client_id])

        formatted_logs_all_set = dict()
        #그룹별로 집계 포맷 생성 #formatted_logs->'Role', 'Round', 'Results_weighted_avg', 'Results_avg', 'Results_fairness', 'Results_raw'를 key로 가질 예정
        for merge_type, eval_res_set in [("participated",
                                          eval_res_participated_clients),
                                         ("unseen", eval_res_unseen_clients)]:
            if eval_res_set != []:
                #metrics_all clients 구성
                """
                metrics_all_clients = {
                    "val_acc":  [0.8, 0.75, 0.9],
                    "val_loss": [0.5, 0.55, 0.45],
                    "val_total": ~~
                    "val_avg_loss":~~
                    "test_acc": ~~
                    "test_loss":~~
                    "test_total": ~~
                    "test_avg_loss":~~

                }
                """
                metrics_all_clients = dict()
                for client_eval_results in eval_res_set:
                    for key in client_eval_results.keys():
                        if key not in metrics_all_clients:
                            metrics_all_clients[key] = list()
                        metrics_all_clients[key].append(
                            float(client_eval_results[key]))




                #formatted_logs 생성

                """
                  formatted_logs = {
                    "Results_avg": {"val_acc": 0.8166, "val_loss": 0.5},
                    "Results_weighted_avg": {...},
                    "Results_fairness": {...},

                }               
                """

                formatted_logs = self._monitor.format_eval_res(
                    metrics_all_clients,
                    rnd=round,
                    role='Server #',
                    forms=self._cfg.eval.report) #self._cfg.eval.report=[weighted_avg, avg, fairness, raw]->return_raw=False라 [weighted_avg, avg, fairness]에 한한 결과만 내보낸다. 즉 통계량으로 압축된것만 보냄.
                #{'Role': 'Server #', 'Round': 24, 'Results_weighted_avg': {'val_total': 124.9622641509434, 'val_loss': 99.96387704124663, 'val_avg_loss': 0.6407083052442673, 'val_acc': 0.6385323871357391, 'test_total': 40.0, 'test_loss': 25.894947483854473, 'test_avg_loss': 0.6473736870963619, 'test_acc': 0.6183962264150943}, 'Results_avg': {'val_total': 124.9622641509434, 'val_loss': 80.0643604836374, 'val_avg_loss': 0.6450719820537335, 'val_acc': 0.6339984738853919, 'test_total': 40.0, 'test_loss': 25.894947483854473, 'test_avg_loss': 0.647373687096362, 'test_acc': 0.6183962264150943}, 'Results_fairness': {'val_total': 124.9622641509434, 'test_total': 40.0, 'val_loss_std': 40.26031761787951, 'val_loss_bottom_decile': 17.608741760253906, 'val_loss_top_decile': 130.40792846679688, 'val_loss_min': 6.489111423492432, 'val_loss_max': 143.4759521484375, 'val_loss_bottom10%': 9.588900661468506, 'val_loss_top10%': 135.79025268554688, 'val_loss_cos1': 0.8934065944700241, 'val_loss_entropy': 3.8212773654406504, 'val_avg_loss_std': 0.060552038285135466, 'val_avg_loss_bottom_decile': 0.5847968344362626, 'val_avg_loss_top_decile': 0.7173797607421875, 'val_avg_loss_min': 0.5036511739095052, 'val_avg_loss_max': 0.8567123413085938, 'val_avg_loss_bottom10%': 0.5518826093219575, 'val_avg_loss_top10%': 0.7636649671131263, 'val_avg_loss_cos1': 0.9956232405959634, 'val_avg_loss_entropy': 3.965981491239061, 'val_acc_std': 0.07289379505770262, 'val_acc_bottom_decile': 0.5540540540540541, 'val_acc_top_decile': 0.7192982456140351, 'val_acc_min': 0.36363636363636365, 'val_acc_max': 0.7666666666666667, 'val_acc_bottom10%': 0.4747722778585586, 'val_acc_top10%': 0.7364474950001266, 'val_acc_cos1': 0.9934552236834064, 'val_acc_entropy': 3.96327855153204, 'test_loss_std': 2.737384274028946, 'test_loss_bottom_decile': 23.15328025817871, 'test_loss_top_decile': 29.409709930419922, 'test_loss_min': 19.09391975402832, 'test_loss_max': 31.324756622314453, 'test_loss_bottom10%': 21.632117462158202, 'test_loss_top10%': 30.28991985321045, 'test_loss_cos1': 0.9944589750905024, 'test_loss_entropy': 3.9646973764426416, 'test_avg_loss_std': 0.06843460685072364, 'test_avg_loss_bottom_decile': 0.5788320064544678, 'test_avg_loss_top_decile': 0.7352427482604981, 'test_avg_loss_min': 0.47734799385070803, 'test_avg_loss_max': 0.7831189155578613, 'test_avg_loss_bottom10%': 0.5408029365539551, 'test_avg_loss_top10%': 0.7572479963302613, 'test_avg_loss_cos1': 0.9944589750905025, 'test_avg_loss_entropy': 3.964697376459541, 'test_acc_std': 0.0828659612442278, 'test_acc_bottom_decile': 0.525, 'test_acc_top_decile': 0.725, 'test_acc_min': 0.4, 'test_acc_max': 0.825, 'test_acc_bottom10%': 0.46499999999999997, 'test_acc_top10%': 0.7541666666666668, 'test_acc_cos1': 0.9911409426294011, 'test_acc_entropy': 3.961152079464096}}



                if merge_type == "unseen": #그럴 일 없음.
                    for key, val in copy.deepcopy(formatted_logs).items():
                        if isinstance(val, dict):
                            # to avoid the overrides of results using the
                            # same name, we use new keys with postfix `unseen`:
                            # 'Results_weighted_avg' ->
                            # 'Results_weighted_avg_unseen'
                            formatted_logs[key + "_unseen"] = val
                            del formatted_logs[key]

                logger.info(formatted_logs)
                #두 그룹(participated, unseen) 각각에 대해 만든 formatted_logs를 모두 합침. 여기선 unseen 그룹이 비어있으니 participated만 들어감:
                formatted_logs_all_set.update(formatted_logs)

                """
                self.best_results는 results_type 키에 대해서 metrics_all_client와 같은 형태로  특정 기준 metric(예: test_loss) 기준 가장 낮은 값의 클라이언트에 해당하는 것이 더 낮아질 때마다 해당 시점의 값으로 갱신됩              
                                metrics_all_clients = {
                                    "val_acc":  [0.8, 0.75, 0.9],
                                    "val_loss": [0.5, 0.55, 0.45],
                                    "val_total": ~~
                                    "val_avg_loss":~~
                                    "test_acc": ~~
                                    "test_loss":~~
                                    "test_total": ~~
                                    "test_avg_loss":~~

                                }
                                

                """

                ##### metrics_all_clients 기반으로 self.best_results["client_best_individual"] 업데이트 ############## 모든 클라이언트들의 평균이 아닌 가장 낮게 나오는 클라이언트 값 기준으로 best(test_loss) 갱신. 
                self._monitor.update_best_result(
                    self.best_results,
                    metrics_all_clients,
                    results_type="unseen_client_best_individual"
                    if merge_type == "unseen" else "client_best_individual") #result_type= "client_best_individual". 모든 클라이언트들의 평균이 아닌 가장 낮게 나오는 클라이언트 값 기준으로 best(test_loss) 갱신. 

                self._monitor.save_formatted_results(formatted_logs) #formatted_logs를 eval_results.log에 한줄 추가. 아래와 같은 형태.
                #{'Role': 'Server #', 'Round': 24, 'Results_weighted_avg': {'val_total': 124.9622641509434, 'val_loss': 99.96387704124663, 'val_avg_loss': 0.6407083052442673, 'val_acc': 0.6385323871357391, 'test_total': 40.0, 'test_loss': 25.894947483854473, 'test_avg_loss': 0.6473736870963619, 'test_acc': 0.6183962264150943}, 'Results_avg': {'val_total': 124.9622641509434, 'val_loss': 80.0643604836374, 'val_avg_loss': 0.6450719820537335, 'val_acc': 0.6339984738853919, 'test_total': 40.0, 'test_loss': 25.894947483854473, 'test_avg_loss': 0.647373687096362, 'test_acc': 0.6183962264150943}, 'Results_fairness': {'val_total': 124.9622641509434, 'test_total': 40.0, 'val_loss_std': 40.26031761787951, 'val_loss_bottom_decile': 17.608741760253906, 'val_loss_top_decile': 130.40792846679688, 'val_loss_min': 6.489111423492432, 'val_loss_max': 143.4759521484375, 'val_loss_bottom10%': 9.588900661468506, 'val_loss_top10%': 135.79025268554688, 'val_loss_cos1': 0.8934065944700241, 'val_loss_entropy': 3.8212773654406504, 'val_avg_loss_std': 0.060552038285135466, 'val_avg_loss_bottom_decile': 0.5847968344362626, 'val_avg_loss_top_decile': 0.7173797607421875, 'val_avg_loss_min': 0.5036511739095052, 'val_avg_loss_max': 0.8567123413085938, 'val_avg_loss_bottom10%': 0.5518826093219575, 'val_avg_loss_top10%': 0.7636649671131263, 'val_avg_loss_cos1': 0.9956232405959634, 'val_avg_loss_entropy': 3.965981491239061, 'val_acc_std': 0.07289379505770262, 'val_acc_bottom_decile': 0.5540540540540541, 'val_acc_top_decile': 0.7192982456140351, 'val_acc_min': 0.36363636363636365, 'val_acc_max': 0.7666666666666667, 'val_acc_bottom10%': 0.4747722778585586, 'val_acc_top10%': 0.7364474950001266, 'val_acc_cos1': 0.9934552236834064, 'val_acc_entropy': 3.96327855153204, 'test_loss_std': 2.737384274028946, 'test_loss_bottom_decile': 23.15328025817871, 'test_loss_top_decile': 29.409709930419922, 'test_loss_min': 19.09391975402832, 'test_loss_max': 31.324756622314453, 'test_loss_bottom10%': 21.632117462158202, 'test_loss_top10%': 30.28991985321045, 'test_loss_cos1': 0.9944589750905024, 'test_loss_entropy': 3.9646973764426416, 'test_avg_loss_std': 0.06843460685072364, 'test_avg_loss_bottom_decile': 0.5788320064544678, 'test_avg_loss_top_decile': 0.7352427482604981, 'test_avg_loss_min': 0.47734799385070803, 'test_avg_loss_max': 0.7831189155578613, 'test_avg_loss_bottom10%': 0.5408029365539551, 'test_avg_loss_top10%': 0.7572479963302613, 'test_avg_loss_cos1': 0.9944589750905025, 'test_avg_loss_entropy': 3.964697376459541, 'test_acc_std': 0.0828659612442278, 'test_acc_bottom_decile': 0.525, 'test_acc_top_decile': 0.725, 'test_acc_min': 0.4, 'test_acc_max': 0.825, 'test_acc_bottom10%': 0.46499999999999997, 'test_acc_top10%': 0.7541666666666668, 'test_acc_cos1': 0.9911409426294011, 'test_acc_entropy': 3.961152079464096}}



                ##### formatted_logs[f"Results_{metric_name}"] 기반으로 self.best_results[f"client_summarized_{form}"] 업데이트, (form은 (weighted_avg', 'avg', 'fairness') ##############  저장 여부는 weighted_avg 관점 test_loss 기준으로 결정하게 된다.

                #update_prior_list로 정한 우선순위가 가장 높은 폼에서 베스트 갱신이 있었는지를 기준으로, 그 라운드에 체크포인트 저장 여부를 결정합니다.
                update_prior = -1  #지금까지 “대표로 채택한 form”의 우선순위 인덱스(작을수록 높음). 초기 -1은 아직 아무 것도 채택 안 함. 숫자가 작을수록 더 중요한 form (fairness=0, avg=1, weighted_avg=2)
                update_prior_list = ['fairness', 'avg', 'weighted_avg']
                update_best_this_round = False


                #설정된 self._cfg.eval.report 순서대로 조시하지만, 우선순위는 update_prior_list 기준으로 따로 매긴다.
                #raw는 best 비교 의미 없음 -> 건너뜀.

                for form in self._cfg.eval.report: #['weighted_avg', 'avg', 'fairness', 'raw']
                    if form in update_prior_list:
                        update_prior_tmp = update_prior_list.index(form)
                    else:
                        update_prior_tmp = -1
                    if form != "raw":
                        metric_name = form + "_unseen" if merge_type == \
                                                          "unseen" else form  #form으로 된다.

                        #반환값 update_best_this_round_tmp는 “해당 form에서 이번 라운드에 best가 갱신됐나?” (True/False)
                        #self.best_results의 "client_summarized_weighted_avg"/"client_summarized_avg", "client_summarized_fairness"를 업데이트
                        update_best_this_round_tmp = \
                            self._monitor.update_best_result(
                                self.best_results,
                                formatted_logs[f"Results_{metric_name}"],
                                results_type=f"unseen_client_summarized_{form}"
                                if merge_type == "unseen" else
                                f"client_summarized_{form}") #self.best_results[f"client_summarized_{form}"]를 update함. (클라이언트 별이 아닌 취합된(weighted_avg', 'avg', 'fairness') test_loss 기반으로 업데이트 될 예정) results_type=f"client_summarized_{form}"
                        if update_prior_tmp >= update_prior: #우선순위가 낮은 메트릭이 나오면 갱신. 결국 update_prior=2로 weighted avg의 것으로 결정이 됨.
                            update_prior = update_prior_tmp
                            update_best_this_round = update_best_this_round_tmp
                if update_best_this_round: #weighted avg 결과로 best_result가 업데이트 되었는지 여부
                    # When the frequency of evaluations is high,
                    # the frequency of writing to disk in the early stages
                    # may also be high
                    if self._cfg.federate.save_to != '' and self.ds_rank == 0:
                        self.aggregator.save_model(self._cfg.federate.save_to,
                                                   self.state)

        return formatted_logs_all_set
        """
        아래의 형태를 return. unseen은 존재핮지 않기에.
        {'Role': 'Server #', 'Round': 24, 'Results_weighted_avg': {'val_total': 124.9622641509434, 'val_loss': 99.96387704124663, 'val_avg_loss': 0.6407083052442673, 'val_acc': 0.6385323871357391, 'test_total': 40.0, 'test_loss': 25.894947483854473, 'test_avg_loss': 0.6473736870963619, 'test_acc': 0.6183962264150943}, 'Results_avg': {'val_total': 124.9622641509434, 'val_loss': 80.0643604836374, 'val_avg_loss': 0.6450719820537335, 'val_acc': 0.6339984738853919, 'test_total': 40.0, 'test_loss': 25.894947483854473, 'test_avg_loss': 0.647373687096362, 'test_acc': 0.6183962264150943}, 'Results_fairness': {'val_total': 124.9622641509434, 'test_total': 40.0, 'val_loss_std': 40.26031761787951, 'val_loss_bottom_decile': 17.608741760253906, 'val_loss_top_decile': 130.40792846679688, 'val_loss_min': 6.489111423492432, 'val_loss_max': 143.4759521484375, 'val_loss_bottom10%': 9.588900661468506, 'val_loss_top10%': 135.79025268554688, 'val_loss_cos1': 0.8934065944700241, 'val_loss_entropy': 3.8212773654406504, 'val_avg_loss_std': 0.060552038285135466, 'val_avg_loss_bottom_decile': 0.5847968344362626, 'val_avg_loss_top_decile': 0.7173797607421875, 'val_avg_loss_min': 0.5036511739095052, 'val_avg_loss_max': 0.8567123413085938, 'val_avg_loss_bottom10%': 0.5518826093219575, 'val_avg_loss_top10%': 0.7636649671131263, 'val_avg_loss_cos1': 0.9956232405959634, 'val_avg_loss_entropy': 3.965981491239061, 'val_acc_std': 0.07289379505770262, 'val_acc_bottom_decile': 0.5540540540540541, 'val_acc_top_decile': 0.7192982456140351, 'val_acc_min': 0.36363636363636365, 'val_acc_max': 0.7666666666666667, 'val_acc_bottom10%': 0.4747722778585586, 'val_acc_top10%': 0.7364474950001266, 'val_acc_cos1': 0.9934552236834064, 'val_acc_entropy': 3.96327855153204, 'test_loss_std': 2.737384274028946, 'test_loss_bottom_decile': 23.15328025817871, 'test_loss_top_decile': 29.409709930419922, 'test_loss_min': 19.09391975402832, 'test_loss_max': 31.324756622314453, 'test_loss_bottom10%': 21.632117462158202, 'test_loss_top10%': 30.28991985321045, 'test_loss_cos1': 0.9944589750905024, 'test_loss_entropy': 3.9646973764426416, 'test_avg_loss_std': 0.06843460685072364, 'test_avg_loss_bottom_decile': 0.5788320064544678, 'test_avg_loss_top_decile': 0.7352427482604981, 'test_avg_loss_min': 0.47734799385070803, 'test_avg_loss_max': 0.7831189155578613, 'test_avg_loss_bottom10%': 0.5408029365539551, 'test_avg_loss_top10%': 0.7572479963302613, 'test_avg_loss_cos1': 0.9944589750905025, 'test_avg_loss_entropy': 3.964697376459541, 'test_acc_std': 0.0828659612442278, 'test_acc_bottom_decile': 0.525, 'test_acc_top_decile': 0.725, 'test_acc_min': 0.4, 'test_acc_max': 0.825, 'test_acc_bottom10%': 0.46499999999999997, 'test_acc_top10%': 0.7541666666666668, 'test_acc_cos1': 0.9911409426294011, 'test_acc_entropy': 3.961152079464096}}
        """

    def broadcast_model_para(self,
                             msg_type='model_para',
                             sample_client_num=-1,
                             filter_unseen_clients=True): #서버가 클라이언트들에게 중앙 adapter 모델들만을 보내주는 핵심 함수. (AdapterModel class의 특징)
                            #requires_grad=True인 파라미터 포함인 것 혹은 self.adapter_names에 있는 어댑터 이름이 파라미터 이름 문자열에 포함되면, requires_grad=False라도 포함.
                             # msg_type는 model_para or evaluate로 들어옴.
        """
        To broadcast the message to all clients or sampled clients

        Arguments:
            msg_type: 'model_para' or other user defined msg_type
            sample_client_num: the number of sampled clients in the broadcast \
                behavior. And ``sample_client_num = -1`` denotes to \
                broadcast to all the clients.
            filter_unseen_clients: whether filter out the unseen clients that \
                do not contribute to FL process by training on their local \
                data and uploading their local model update. The splitting is \
                useful to check participation generalization gap in [ICLR'22, \
                What Do We Mean by Generalization in Federated Learning?] \
                You may want to set it to be False when in evaluation stage
        """
        if filter_unseen_clients: #True, self.unseen_clients_id=[]
            # to filter out the unseen clients when sampling
            self.sampler.change_state(self.unseen_clients_id, 'unseen')

        if sample_client_num > 0:

            if self.sampler is None and self._cfg.federate.sampler == 'cluster':
                self.sampler = get_sampler(self._cfg.federate.sampler, self.client_num, None, config=self._cfg)



            # cluster: 샘플링 직전 라운드→허용집합/샘플 수 세팅
            if getattr(self._cfg.federate, 'sampler', 'uniform') == 'cluster' \
               and hasattr(self.sampler, 'set_allowed_for_round'):
                self.sampler.set_allowed_for_round(self.state)

            receiver = self.sampler.sample(size=sample_client_num) #Client selection 일어남.

        else: ## -1 혹은 0 이면 “모두에게”, server.eval()상황에서 벌어짐.
            # broadcast to all clients
            receiver = list(self.comm_manager.neighbors.keys())
            if msg_type == 'model_para':
                self.sampler.change_state(receiver, 'working')

        if self._cfg.federate.share_local_model and not self._cfg.federate.online_aggr: #True
            model_para = copy.deepcopy(self.models[0].state_dict()) #server model 적용 model_para = self.models[0].state_dict())


        # We define the evaluation happens at the end of an epoch
        rnd = self.state - 1 if msg_type == 'evaluate' else self.state

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=receiver,
                    state=min(rnd, self.total_round_num),
                    timestamp=self.cur_timestamp,
                    content=model_para)) #receiver들에게 global model을 보냄. msg_type이  'model_para'이든 'evaluate'든 'adapter_eval'이든 상관 없음.

        if filter_unseen_clients:
            # restore the state of the unseen clients within sampler
            self.sampler.change_state(self.unseen_clients_id, 'seen')

    def broadcast_client_address(self): #클라이언트 모두에게 참여하는 모든 클라이언트 정보들을 보냄. 실제로 안쓰임.
        """
        To broadcast the communication addresses of clients (used for \
        additive secret sharing)
        """

        self.comm_manager.send(
            Message(msg_type='address',
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    state=self.state,
                    timestamp=self.cur_timestamp,
                    content=self.comm_manager.get_neighbors()))

    def check_buffer(self,
                     cur_round,
                     min_received_num,
                     check_eval_result=False): #현재 라운드 cur_round 에 대해, min_received_num (필요 최소 응답 수) 만큼 클라이언트 응답을 받았는지 확인, check_eval_result=True인 경우는 "평가 응답"인지 확인
        """
        To check the message buffer

        Arguments:
            cur_round (int): The current round number
            min_received_num (int): The minimal number of the receiving \
                messages
            check_eval_result (bool): To check training results for \
                evaluation results

        Returns
            bool: Whether enough messages have been received or not
        """

        if check_eval_result:
            if 'eval' not in self.msg_buffer.keys() or len(
                    self.msg_buffer['eval'].keys()) == 0:
                return False

            buffer = self.msg_buffer['eval']
            cur_round = max(buffer.keys())
            cur_buffer = buffer[cur_round]
            
            return len(cur_buffer) >= min_received_num
        else:
            if cur_round not in self.msg_buffer['train']:
                cur_buffer = dict()
            else:
                cur_buffer = self.msg_buffer['train'][cur_round]

            return len(cur_buffer)+len(self.staled_msg_buffer) >= \
                    min_received_num

    def check_client_join_in(self): #전체 클라이언트 수만큼 채워졌는지 확인
        """
        To check whether all the clients have joined in the FL course.
        """

        if len(self._cfg.federate.join_in_info) != 0:
            return len(self.join_in_info) == self.client_num
        else:
            return self.join_in_client_num == self.client_num

    def trigger_for_start(self):
        """
        To start the FL course when the expected number of clients have joined
        """

        if self.check_client_join_in(): ##전체 클라이언트 수만큼 join_in이 반영됐는지 여부
            logger.info('Waited all clients join, start now...')

            #샘플할 클리아언트들에게 서버 모델 broadcast. 서버에서 클라이언트들에게 'model_para' 형태의 message 전달.
            self.trigger_for_feat_engr(
                self.broadcast_model_para, {
                    'msg_type': 'model_para',
                    'sample_client_num': self.sample_client_num
                })

            logger.info(
                '----------- Starting training (Round #{:d}) -------------'.
                format(self.state)) #0라운드 시작한다는 것 표시
            


    def trigger_for_feat_engr(self,
                              trigger_train_func,
                              kwargs_for_trigger_train_func={}): #trigger_for_start에서 쓰임.
        """
        Interface for feature engineering, the default operation is none
        """
        trigger_train_func(**kwargs_for_trigger_train_func)

    # def trigger_for_time_up(self, check_timestamp=None):
    #     """
    #     The handler for time up: modify the currency timestamp \
    #     and check the trigger condition
    #     """
    #     if self.is_finish:
    #         return False

    #     if check_timestamp is not None and \
    #             check_timestamp < self.deadline_for_cur_round:
    #         return False

    #     self.cur_timestamp = self.deadline_for_cur_round
    #     self.check_and_move_on()
    #     return True

    def terminate(self, msg_type='finish'): #run, check_and_save에서 쓰임.
        """
        To terminate the FL course
        """
        self.is_finish = True
        if self.model_num > 1:
            model_para = [model.state_dict() for model in self.models]
        else:
            model_para = self.models[0].state_dict()

        self._monitor.finish_fl() ##🔹 목적: 학습 종료 시, 시스템 지표들을 system_metrics.log 파일에 기록 (단, rank 0에서만 실행) #주요 호출위치: 종료 직전

        self.comm_manager.send(
            Message(msg_type=msg_type,
                    sender=self.ID,
                    receiver=list(self.comm_manager.neighbors.keys()),
                    state=self.state,
                    timestamp=self.cur_timestamp,
                    content=model_para)) #모든 클라이언트들에게 model_para content를 담고 'finish' message 전달.

    def eval(self): #클라이언트에서 평가. 클라이언트에 중앙 모델 브로드캐스팅.
        """
        To conduct evaluation. When ``cfg.federate.make_global_eval=True``, \
        a global evaluation is conducted by the server.
        """
         #(클라이언트에게 평가 위임) 이걸로 적용됨.
        # Preform evaluation in clients
        # 'evaluate' 타입 메시지를 클라이언트로 보내서
        # 클라이언트 단위로 평가를 수행하게 함
        self.broadcast_model_para(msg_type='evaluate',
                                    filter_unseen_clients=False) #sample_client_num=-1로 적용되므로 모든 클라이언트에게 msg_type='evaluate'의 메시지가 전송될 예정.

    def callback_funcs_model_para(self, message: Message): #클라이언트가 로컬 훈련 후 보낸 model parameter를 수신하고, 이를 서버의 버퍼에 저장하거나 직접 집계하며 집계 조건이 충족되면 다음 라운드로 진행
        """
        The handling function for receiving model parameters, which triggers \
        ``check_and_move_on`` (perform aggregation when enough feedback has \
        been received). This handling function is widely used in various FL \
        courses.

        Arguments:
            message: The received message.
        """
        if self.is_finish:
            return 'finish'

        round = message.state
        sender = message.sender
        timestamp = message.timestamp
        content = message.content
        self.sampler.change_state(sender, 'idle') #메시지를 받았으니 해당 클라이언트 상태를 작업 완료로 표시

        # update the currency timestamp according to the received message
        assert timestamp >= self.cur_timestamp  # for test, 도착한 메시지의 timestamp가 현재 시각보다 앞선 메시지면 assert로 에러
        self.cur_timestamp = timestamp #클라이언트 도착 순서대로 업데이트

        if round == self.state: #현재 라운드 → 정상 메시지로 저장
            if round not in self.msg_buffer['train']:
                self.msg_buffer['train'][round] = dict()
            # Save the messages in this round
            self.msg_buffer['train'][round][sender] = content
        elif round >= self.state - self.staleness_toleration: #허용된 지연 메시지 → 별도 버퍼에 저장
            # Save the staled messages
            self.staled_msg_buffer.append((round, sender, content))
        else:
            # Drop the out-of-date messages
            logger.info(f'Drop a out-of-date message from round #{round}')
            self.dropout_num += 1
        move_on_flag = self.check_and_move_on()

        return move_on_flag

    def callback_funcs_for_join_in(self, message: Message): #클라이언트 → 서버로 'join_in' 혹은 'join_in_info' 메시지를 보낼 때 서버가 호출하는 함수. 즉 첫 접속은 join_in, 그 후 서버가 “필요하면 추가 정보 줘”라고 다시 요청하면 클라이언트가 join_in_info 로 응답한다는 구조입니다.
        """
        The handling function for receiving the join in information. The \
        server might request for some information (such as \
        ``num_of_samples``) if necessary, assign IDs for the servers. \
        If all the clients have joined in, the training process will be \
        triggered.

        Arguments:
            message: The received message
        """

        if 'info' in message.msg_type: #join_in_info (추가 정보), 1) payload(content) 검증 2) self.join_in_info 에 저장

            sender, info = message.sender, message.content
            for key in self._cfg.federate.join_in_info:#self._cfg.federate.join_in_info: 클라이언트가 참여할 때 다음 두 정보를 반드시 보내야 한다고 서버가 명시한 것
                assert key in info # key in info.keys()와 동치
            self.join_in_info[sender] = info #sender는 클라이언트의 ID
            logger.info('Server: Client #{:d} has joined in !'.format(sender))


        else: #join_in (처음 접속), 1) 클라이언트 수 카운트 2) 주소(소켓 / gRPC 채널) 등록 3) ID 미보유(sender==-1)면 새 ID 부여→ assign_client_id 전송
            self.join_in_client_num += 1
            sender, address = message.sender, message.content
            if int(sender) == -1:  # assign number to client, 클라이언트가 아직 ID를 할당받지 않은 상태 (sender == -1), 현재 카운트된 순번으로 ID 할당, 클라이언트를 comm_manager에 등록
                sender = self.join_in_client_num
                self.comm_manager.add_neighbors(neighbor_id=sender,
                                                address=address)
                self.comm_manager.send(
                    Message(msg_type='assign_client_id',
                            sender=self.ID,
                            receiver=[sender],
                            state=self.state,
                            timestamp=self.cur_timestamp,
                            content=str(sender)))#클라이언트에게 assign_client_id 메시지로 ID 통보. sender는 client id, self.ID는 서버의 ID.
            else:#이미 ID를 가지고 있다면, 그냥 이 주소로 등록만 진행
                self.comm_manager.add_neighbors(neighbor_id=sender,
                                                address=address)

            if len(self._cfg.federate.join_in_info) != 0:#서버 설정에서 join_in_info 키가 있다면, 클라이언트에게 해당 정보를 다시 요청함
                self.comm_manager.send(
                    Message(msg_type='ask_for_join_in_info',  # ① “이 타입의 메시지”를
                            sender=self.ID, # ② 보내는 사람(=서버 ID)
                            receiver=[sender],  # ③ 받는 사람(=해당 클라이언트)
                            state=self.state, # ④ 현재 라운드
                            timestamp=self.cur_timestamp, # ⑤ 서버 시각
                            content=self._cfg.federate.join_in_info.copy()))# ⑥ ★요청 payload

        self.trigger_for_start() #현재까지 등록된 클라이언트 수를 기준으로, 모든 클라이언트가 참여했는지 확인하고, 맞으면 학습을 시작

    def callback_funcs_for_metrics(self, message: Message): #클라이언트가 평가 결과 (예: test accuracy) 를 서버에 보낼 때 실행되는 함수. self.msg_buffer['eval'] 업데이트 및 라운드를 끝낼지 여부를 반환.
        """
        The handling function for receiving the evaluation results, \
        which triggers ``check_and_move_on`` (perform aggregation when \
        enough feedback has been received).

        Arguments:
            message: The received message
        """

        rnd = message.state #라운드 수
        sender = message.sender #client id가 sender
        content = message.content

        if rnd not in self.msg_buffer['eval'].keys():
            self.msg_buffer['eval'][rnd] = dict()

        self.msg_buffer['eval'][rnd][sender] = content

        return self.check_and_move_on(check_eval_result=True)

    @classmethod
    def get_msg_handler_dict(cls):
        return cls().msg_handlers_str
