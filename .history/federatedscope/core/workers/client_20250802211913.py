import copy
import logging
import sys
import pickle

from federatedscope.core.message import Message
from federatedscope.core.communication import StandaloneCommManager, \
    StandaloneDDPCommManager, gRPCCommManager
from federatedscope.core.monitors.early_stopper import EarlyStopper
from federatedscope.core.auxiliaries.trainer_builder import get_trainer
from federatedscope.core.secret_sharing import AdditiveSecretSharing
from federatedscope.core.auxiliaries.utils import merge_dict_of_results, \
    calculate_time_cost, add_prefix_to_path, get_ds_rank
from federatedscope.core.workers.base_client import BaseClient

logger = logging.getLogger(__name__)
if get_ds_rank() == 0:
    logger.setLevel(logging.INFO)


class Client(BaseClient):
    """
    The Client class, which describes the behaviors of client in an FL \
    course. The behaviors are described by the handling functions (named as \
    ``callback_funcs_for_xxx``)

    Arguments:
        ID: The unique ID of the client, which is assigned by the server
        when joining the FL course
        server_id: (Default) 0
        state: The training round
        config: The configuration
        data: The data owned by the client
        model: The model maintained locally
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
        ss_manager: secret sharing manager
        msg_buffer: dict buffer for storing message
        comm_manager: manager for communication, \
            see ``federatedscope.core.communication``
    """
    def __init__(self,
                 ID=-1,
                 server_id=None,
                 state=-1,
                 config=None,
                 data=None, #ClientData 클래스
                 model=None,
                 device='cpu',
                 strategy=None,
                 is_unseen_client=False,
                 *args,
                 **kwargs):
        super(Client, self).__init__(ID, state, config, model, strategy)

        self.data = data #ClientData 클래스

        # Register message handlers
        self._register_default_handlers()

        # Un-configured worker
        if config is None:
            return

        # the unseen_client indicates that whether this client contributes to
        # FL process by training on its local data and uploading the local
        # model update, which is useful for check the participation
        # generalization gap in
        # [ICLR'22, What Do We Mean by Generalization in Federated Learning?]
        self.is_unseen_client = is_unseen_client

        # Parse the attack_id since we support both 'int' (for single attack)
        # and 'list' (for multiple attacks) for config.attack.attack_id
        parsed_attack_ids = list()
        if isinstance(config.attack.attacker_id, int): #True
            parsed_attack_ids.append(config.attack.attacker_id)
        elif isinstance(config.attack.attacker_id, list):
            parsed_attack_ids = config.attack.attacker_id
        else:
            raise TypeError(f"The expected types of config.attack.attack_id "
                            f"include 'int' and 'list', but we got "
                            f"{type(config.attack.attacker_id)}")

        # Attack only support the stand alone model;
        # Check if is a attacker; a client is a attacker if the
        # config.attack.attack_method is provided
        self.is_attacker = ID in parsed_attack_ids and \
            config.attack.attack_method != '' and \
            config.federate.mode == 'standalone'   #False

        # Build Trainer
        # trainer might need configurations other than those of trainer node
        self.trainer = get_trainer(model=model,
                                   data=data,
                                   device=device,
                                   config=self._cfg,
                                   is_attacker=self.is_attacker,
                                   monitor=self._monitor)#federatedscope.llm.trainer.reward_choice_trainer.RewardChoiceTrainer
        self.device = device

        # For client-side evaluation
        self.best_results = dict()
        self.history_results = dict()

        # in local or global training mode, we do use the early stopper.
        # Otherwise, we set patience=0 to deactivate the local early-stopper
        patience = self._cfg.early_stop.patience if \
            self._cfg.federate.method in [
                "local", "global"
            ] else 0   # 0
        self.early_stopper = EarlyStopper(
            patience, self._cfg.early_stop.delta,
            self._cfg.early_stop.improve_indicator_mode,
            self._monitor.the_larger_the_better) #self._cfg.early_stop.improve_indicator_mode='best'

        # Secret Sharing Manager and message buffer
        self.ss_manager = AdditiveSecretSharing(
            shared_party_num=int(self._cfg.federate.sample_client_num
                                 )) if self._cfg.federate.use_ss else None  #None
        

        
        
        self.msg_buffer = {'train': dict(), 'eval': dict()}

        # train 집계본을 evaluate 때 exp_print로 함께 내보내기 위한 임시 저장소
        self._last_train_results_raw = {}

        # Communication and communication ability
        if 'resource_info' in kwargs and kwargs['resource_info'] is not None: #PASS
            self.comp_speed = float(
                kwargs['resource_info']['computation']) / 1000.  # (s/sample)
            self.comm_bandwidth = float(
                kwargs['resource_info']['communication'])  # (kbit/s)
        else: #여기 걸림
            self.comp_speed = None
            self.comm_bandwidth = None

        if self._cfg.backend == 'torch': #여기 걸림
            try:
                self.model_size = sys.getsizeof(pickle.dumps(
                    self.model)) / 1024.0 * 8.  # kbits
            except Exception as error:
                self.model_size = 1.0
                logger.warning(f'{error} in calculate model size.')
        else:
            # TODO: calculate model size for TF Model
            self.model_size = 1.0
            logger.warning(f'The calculation of model size in backend:'
                           f'{self._cfg.backend} is not provided.')

        # Initialize communication manager
        self.server_id = server_id 

        
        #큐를 이용해 서버·클라이언트 간 메시지 송수신을 담당할 CommManager 인스턴스를 생성
        comm_queue = kwargs['shared_comm_queue'] #deque([])
        if self._cfg.federate.process_num <= 1:
            self.comm_manager = StandaloneCommManager(
                comm_queue=comm_queue, monitor=self._monitor)
        else:
            self.comm_manager = StandaloneDDPCommManager(
                comm_queue=comm_queue, monitor=self._monitor)
        self.local_address = None


        self.logger=logger

    def _ddp_rank_world():
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                return dist.get_rank(), dist.get_world_size()
        except Exception:
            pass
        try:
            from accelerate.state import AcceleratorState
            st = AcceleratorState()
            if getattr(st, "initialized", False):
                # process_index: 0..world-1, num_processes: world
                return st.process_index, st.num_processes
        except Exception:
            pass
        return 0, 1


    def _log_split_metrics(self, role_str, round_idx, split, trainer_ctx):
        """
        split: 'train' | 'val' | 'test'
        trainer_ctx: self.trainer.ctx
        """
        # accelerator 찾기 (client, trainer.ctx, trainer 순으로 시도)
        accel = getattr(self, "accelerator", None)
        if accel is None:
            accel = getattr(trainer_ctx, "accelerator", None)
        if accel is None and hasattr(self, "trainer") and hasattr(self.trainer, "accelerator"):
            accel = self.trainer.accelerator

        world = getattr(accel, "num_processes", 1) if accel is not None else 1
        rank  = getattr(accel, "process_index", 0) if accel is not None else 0

        # ① 각 rank 로컬 스냅샷
        local = getattr(trainer_ctx, "local_results_for_log", {}) or {}
        self.logger.info({
            'Role': role_str, 'Round': round_idx, 'Split': split,
            'Rank': f'{rank}/{world}', 'Local': True,
            'Results': local
        })

        # ② rank0: per-proc 리스트
        if world > 1 and rank == 0:
            per_procs = getattr(trainer_ctx, "per_proc_local_results", None)
            if per_procs:
                self.logger.info({
                    'Role': role_str, 'Round': round_idx, 'Split': split,
                    'Local': False, 'Results_procs': per_procs
                })

        # ③ 집계 결과(해당 split 접두사 4키만; alias 금지)
        agg = getattr(trainer_ctx, "eval_metrics", {}) or {}
        sp  = f"{split}_"
        agg_clean = {k: v for k, v in agg.items() if k.startswith(sp)}
        if agg_clean:
            self.logger.info({
                'Role': role_str, 'Round': round_idx,
                'Results_raw': agg_clean
            })
            # train 라운드의 집계본을 rank0에서만 캐싱해 두었다가,
            # evaluate 시점에 exp_print.raw에 같이 실어 보냅니다.
            if split == 'train' and rank == 0:
                self._last_train_results_raw = dict(agg_clean)  # copy

    def _is_main_process(self) -> bool:
        trainer = getattr(self, 'trainer', None)
        if trainer is not None and hasattr(trainer, 'accelerator') and trainer.accelerator is not None:
            return trainer.accelerator.is_main_process
        return True  # accelerator 미사용 시 단일 프로세스이므로 True        


    def _gen_timestamp(self, init_timestamp, instance_number):
        if init_timestamp is None:
            return None

        comp_cost, comm_cost = calculate_time_cost(
            instance_number=instance_number,
            comm_size=self.model_size,
            comp_speed=self.comp_speed,
            comm_bandwidth=self.comm_bandwidth)
        return init_timestamp + comp_cost + comm_cost

    def _calculate_model_delta(self, init_model, updated_model):
        if not isinstance(init_model, list):
            init_model = [init_model]
            updated_model = [updated_model]

        model_deltas = list()
        for model_index in range(len(init_model)):
            model_delta = copy.deepcopy(init_model[model_index])
            for key in init_model[model_index].keys():
                model_delta[key] = updated_model[model_index][
                    key] - init_model[model_index][key]
            model_deltas.append(model_delta)

        if len(model_deltas) > 1:
            return model_deltas
        else:
            return model_deltas[0]

    def join_in(self):
        """
        To send ``join_in`` message to the server for joining in the FL course.
        """
        self.comm_manager.send(
            Message(msg_type='join_in',
                    sender=self.ID,
                    receiver=[self.server_id],
                    timestamp=0,
                    content=self.local_address))

    def run(self):
        """
        To listen to the message and handle them accordingly (used for \
        distributed mode)
        """
        while True:
            msg = self.comm_manager.receive()
            if self.state <= msg.state:
                self.msg_handlers[msg.msg_type](msg)

            if msg.msg_type == 'finish':
                break

    def run_standalone(self):
        """
        Run in standalone mode
        """
        self.join_in()
        self.run()

    def callback_funcs_for_model_para(self, message: Message): #서버로부터 받은 글로벌 모델 파라미터(혹은 시크릿 셰어 조각)를 처리하고, 로컬 학습을 트리거한 뒤 결과를 서버에 다시 보냅니다.
        """
        The handling function for receiving model parameters, \
        which triggers the local training process. \
        This handling function is widely used in various FL courses.

        Arguments:
            message: The received message
        """
        if 'ss' in message.msg_type: #msg_type 에 'ss' 포함 ⇒ 비밀 분할 조각(fragment) 수신 처리. 해당 안함!!
            # A fragment of the shared secret
            state, content, timestamp = message.state, message.content, \
                                        message.timestamp
            self.msg_buffer['train'][state].append(content)

            if len(self.msg_buffer['train']
                   [state]) == self._cfg.federate.client_num:
                # Check whether the received fragments are enough
                model_list = self.msg_buffer['train'][state]
                sample_size, first_aggregate_model_para = model_list[0]
                single_model_case = True
                if isinstance(first_aggregate_model_para, list):
                    assert isinstance(first_aggregate_model_para[0], dict), \
                        "aggregate_model_para should a list of multiple " \
                        "state_dict for multiple models"
                    single_model_case = False
                else:
                    assert isinstance(first_aggregate_model_para, dict), \
                        "aggregate_model_para should " \
                        "a state_dict for single model case"
                    first_aggregate_model_para = [first_aggregate_model_para]
                    model_list = [[model] for model in model_list]

                for sub_model_idx, aggregate_single_model_para in enumerate(
                        first_aggregate_model_para):
                    for key in aggregate_single_model_para:
                        for i in range(1, len(model_list)):
                            aggregate_single_model_para[key] += model_list[i][
                                sub_model_idx][key]

                self.comm_manager.send(
                    Message(msg_type='model_para',
                            sender=self.ID,
                            receiver=[self.server_id],
                            state=self.state,
                            timestamp=timestamp,
                            content=(sample_size, first_aggregate_model_para[0]
                                     if single_model_case else
                                     first_aggregate_model_para)))

        else: #해당함
            round = message.state # 서버가 보낸 라운드 번호
            sender = message.sender # 메시지를 보낸 주체(서버) ID
            timestamp = message.timestamp # 서버 시각
            content = message.content # 실제 모델 파라미터 (또는 파라미터 델타)

            # dequantization
            if self._cfg.quantization.method == 'uniform': #해당 안함.
                from federatedscope.core.compression import \
                    symmetric_uniform_dequantization
                if isinstance(content, list):  # multiple model
                    content = [
                        symmetric_uniform_dequantization(x) for x in content
                    ]
                else:
                    content = symmetric_uniform_dequantization(content)

            # When clients share the local model, we must set strict=True to
            # ensure all the model params (which might be updated by other
            # clients in the previous local training process) are overwritten
            # and synchronized with the received model
            if self._cfg.federate.process_num > 1:
                for k, v in content.items():
                    content[k] = v.to(self.device)



            self.trainer.update(content,
                                strict=self._cfg.federate.share_local_model) #서버에서 보낸 파라미터로 모델을 덮어씌웁니다.
            


            self.state = round
            skip_train_isolated_or_global_mode = \
                self.early_stopper.early_stopped and \
                self._cfg.federate.method in ["local", "global"] #FALSE
            

            if self.is_unseen_client or skip_train_isolated_or_global_mode: #해당 안함
                # for these cases (1) unseen client (2) isolated_global_mode,
                # we do not local train and upload local model
                sample_size, model_para_all, results = \
                    0, self.trainer.get_model_para(), {}
                if skip_train_isolated_or_global_mode: #FALSE
                    logger.info(
                        f"[Local/Global mode] Client #{self.ID} has been "
                        f"early stopped, we will skip the local training")
                    self._monitor.local_converged()
            else:#해당 
                # 이미 로컬 수렴한 클라이언트라면 훈련을 건너뛰기도 하고…
                if self.early_stopper.early_stopped and \
                        self._monitor.local_convergence_round == 0:
                    logger.info(
                        f"[Normal FL Mode] Client #{self.ID} has been locally "
                        f"early stopped. "
                        f"The next FL update may result in negative effect")
                    self._monitor.local_converged()#self._monitor의 self.local_convergence_wall_time, self.local_convergence_round 지정.


                sample_size, model_para_all, results = self.trainer.train()#이 실제 한 에폭(batch) 치 로컬 업데이트를 실행하고, sample_size (총 샘플 수), model_para_all (업데이트된 state_dict), results (로컬 로그: loss, acc 등) 세 가지를 리턴
                
                
                if self._cfg.federate.share_local_model and not \
                        self._cfg.federate.online_aggr:
                    model_para_all = copy.deepcopy(model_para_all) #안전하게 복사

                rank, world = 0, 1
                if hasattr(self.trainer, 'accelerator') and self.trainer.accelerator is not None:
                    rank  = self.trainer.accelerator.process_index
                    world = self.trainer.accelerator.num_processes

                # ✅ 원하는 포맷의 로그 3종(로컬 / per-proc / 집계) 출력
                self._log_split_metrics(
                    role_str=f'Client #{self.ID}',
                    round_idx=self.state,
                    split='train',
                    trainer_ctx=self.trainer.ctx
                )


                # train_log_res = self._monitor.format_eval_res(
                #     results,
                #     rnd=self.state,
                #     role='Client #{}'.format(self.ID),
                #     return_raw=True)
                

                # logger.info(train_log_res)


                if self._cfg.wandb.use and self._cfg.wandb.client_train_info: #FALSE
                    self._monitor.save_formatted_results(train_log_res,
                                                         save_file_name="")

            # Return the feedbacks to the server after local update
            if self._cfg.federate.use_ss: #해당 안함
                assert not self.is_unseen_client, \
                    "Un-support using secret sharing for unseen clients." \
                    "i.e., you set cfg.federate.use_ss=True and " \
                    "cfg.federate.unseen_clients_rate in (0, 1)"
                single_model_case = True
                if isinstance(model_para_all, list):
                    assert isinstance(model_para_all[0], dict), \
                        "model_para should a list of " \
                        "multiple state_dict for multiple models"
                    single_model_case = False
                else:
                    assert isinstance(model_para_all, dict), \
                        "model_para should a state_dict for single model case"
                    model_para_all = [model_para_all]
                model_para_list_all = []
                for model_para in model_para_all:
                    for key in model_para:
                        model_para[key] = model_para[key] * sample_size
                    model_para_list = self.ss_manager.secret_split(model_para)
                    model_para_list_all.append(model_para_list)
                frame_idx = 0
                for neighbor in self.comm_manager.neighbors:
                    if neighbor != self.server_id:
                        content_frame = model_para_list_all[0][frame_idx] if \
                            single_model_case else \
                            [model_para_list[frame_idx] for model_para_list
                             in model_para_list_all]
                        self.comm_manager.send(
                            Message(msg_type='ss_model_para',
                                    sender=self.ID,
                                    receiver=[neighbor],
                                    state=self.state,
                                    timestamp=self._gen_timestamp(
                                        init_timestamp=timestamp,
                                        instance_number=sample_size),
                                    content=content_frame))
                        frame_idx += 1
                content_frame = model_para_list_all[0][frame_idx] if \
                    single_model_case else \
                    [model_para_list[frame_idx] for model_para_list in
                     model_para_list_all]
                self.msg_buffer['train'][self.state] = [(sample_size,
                                                         content_frame)]
            else: #해당
                if self._cfg.asyn.use or self._cfg.aggregator.robust_rule in \
                        ['krum', 'normbounding', 'median', 'trimmedmean',
                         'bulyan']: #해당 안함
                    # Return the model delta when using asynchronous training
                    # protocol, because the staled updated might be discounted
                    # and cause that the sum of the aggregated weights might
                    # not be equal to 1
                    shared_model_para = self._calculate_model_delta(
                        init_model=content, updated_model=model_para_all)
                else: #해당
                    shared_model_para = model_para_all

                # quantization
                if self._cfg.quantization.method == 'uniform':
                    from federatedscope.core.compression import \
                        symmetric_uniform_quantization
                    nbits = self._cfg.quantization.nbits
                    if isinstance(shared_model_para, list):
                        shared_model_para = [
                            symmetric_uniform_quantization(x, nbits)
                            for x in shared_model_para
                        ]
                    else:
                        shared_model_para = symmetric_uniform_quantization(
                            shared_model_para, nbits)

                self.comm_manager.send(
                    Message(msg_type='model_para', # ↔ 서버가 “train” 단계로 인식
                            sender=self.ID, # 이 클라이언트 ID
                            receiver=[sender], # 앞서 저장한 서버 ID
                            state=self.state, # (같은) 라운드 번호
                            timestamp=self._gen_timestamp(
                                init_timestamp=timestamp,
                                instance_number=sample_size), # → 서버의 time-based staleness 제어용
                            content=(sample_size, shared_model_para))) #데이터 갯수 및 로컬 모델을 content로 담아서 보낸다.

    def callback_funcs_for_assign_id(self, message: Message): #분산 모드에서 서버가 부여한 클라이언트 ID (assign_client_id)를 받아 self.ID 에 설정합니다.
        """
        The handling function for receiving the client_ID assigned by the \
        server (during the joining process), which is used in the \
        distributed mode.

        Arguments:
            message: The received message
        """
        content = message.content
        self.ID = int(content)
        logger.info('Client (address {}:{}) is assigned with #{:d}.'.format(
            self.comm_manager.host, self.comm_manager.port, self.ID))

    def callback_funcs_for_join_in_info(self, message: Message): #볼 필요 없을 듯. 서버가 “참가 정보”를 요청할 때(batch size, 샘플 개수, 리소스 등), 로컬 설정을 읽어 채워서 응답합니다.
        """
        The handling function for receiving the request of join in \
        information (such as ``batch_size``, ``num_of_samples``) during \
        the joining process.

        Arguments:
            message: The received message
        """
        requirements = message.content
        timestamp = message.timestamp
        join_in_info = dict()
        for requirement in requirements:
            if requirement.lower() == 'num_sample':
                if self._cfg.train.batch_or_epoch == 'batch':
                    num_sample = self._cfg.train.local_update_steps * \
                                 self._cfg.dataloader.batch_size
                else:
                    num_sample = self._cfg.train.local_update_steps * \
                                 len(self.trainer.data.train_data)
                join_in_info['num_sample'] = num_sample
                if self._cfg.trainer.type == 'nodefullbatch_trainer':
                    join_in_info['num_sample'] = \
                        self.trainer.data.train_data.x.shape[0]
            elif requirement.lower() == 'client_resource':
                assert self.comm_bandwidth is not None and self.comp_speed \
                       is not None, "The requirement join_in_info " \
                                    "'client_resource' does not exist."
                join_in_info['client_resource'] = self.model_size / \
                    self.comm_bandwidth + self.comp_speed
            else:
                raise ValueError(
                    'Fail to get the join in information with type {}'.format(
                        requirement))
        self.comm_manager.send(
            Message(msg_type='join_in_info',
                    sender=self.ID,
                    receiver=[self.server_id],
                    state=self.state,
                    timestamp=timestamp,
                    content=join_in_info))

    def callback_funcs_for_address(self, message: Message): #볼 필요 없을 듯. (비밀 셰어링 등) 복잡한 토폴로지를 위해 서버가 다른 클라이언트 주소 목록을 보낼 때 처리합니다.

        """
        The handling function for receiving other clients' IP addresses, \
        which is used for constructing a complex topology

        Arguments:
            message: The received message
        """
        content = message.content
        for neighbor_id, address in content.items():
            if int(neighbor_id) != self.ID:
                self.comm_manager.add_neighbors(neighbor_id, address)

    def callback_funcs_for_evaluate(self, message: Message): #서버가 “평가 요청(evaluate)” 메시지를 보낼 때, 로컬 데이터에 대해 평가를 수행하고 결과(metrics)를 서버에 보냅니다.
        """
        The handling function for receiving the request of evaluating

        Arguments:
            message: The received message
        """
        sender, timestamp = message.sender, message.timestamp
        self.state = message.state
        if message.content is not None:
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)#중앙서버 모델 broadcast
        if self.early_stopper.early_stopped and self._cfg.federate.method in [
                "local", "global"
        ]:#FALSE
            metrics = list(self.best_results.values())[0]
        else:
            metrics = {}
            if self._cfg.finetune.before_eval: #FALSE
                self.trainer.finetune()
            for split in self._cfg.eval.split: #['test', 'val'] 
                # TODO: The time cost of evaluation is not considered here
                eval_metrics = self.trainer.evaluate(
                    target_data_split_name=split)

                # ✅ 원하는 포맷의 로그 3종(로컬 / per-proc / 집계) 출력
                self._log_split_metrics(
                    role_str=f'Client #{self.ID}',
                    round_idx=self.state,
                    split=split,
                    trainer_ctx=self.trainer.ctx
                )

                metrics.update(**eval_metrics)
        # 4) 최종 집계 요약은 (원하면) main process에서만 1회 출력
        is_main = True
        if hasattr(self.trainer, 'accelerator') and self.trainer.accelerator is not None:
            is_main = self.trainer.accelerator.is_main_process


        formatted_eval_res = self._monitor.format_eval_res(
            metrics, rnd=self.state, role=f'Client #{self.ID}', forms=['raw'], return_raw=True)

            
        update_best_this_round = self._monitor.update_best_result(
            self.best_results,
            formatted_eval_res['Results_raw'],
            results_type=f"client #{self.ID}",
        )
        
        if update_best_this_round and self._cfg.federate.save_client_model:
            path = add_prefix_to_path(f'client_{self.ID}_',
                                        self._cfg.federate.save_to)
            if self.ds_rank == 0:
                self.trainer.save_model(path, self.state)

            self.history_results = merge_dict_of_results(
                self.history_results, formatted_eval_res['Results_raw'])
            
            #self._cfg.eval.best_res_update_round_wise_key='test_loss'인 상황
            self.early_stopper.track_and_check(self.history_results[
                self._cfg.eval.best_res_update_round_wise_key]) #여기서 self.early_stopper.early_stopped가 False -> True로 바뀔 수 있음. Default는 False
 
        self.comm_manager.send(
            Message(msg_type='metrics',
                    sender=self.ID,
                    receiver=[sender],
                    state=self.state,
                    timestamp=timestamp,
                    content=metrics))

    def callback_funcs_for_finish(self, message: Message): #FL 과정이 “완료(finish)” 신호를 받을 때, 최종 모델을 로드·저장하고 로컬 모니터에 종료를 알립니다.
        """
        The handling function for receiving the signal of finishing the FL \
        course.

        Arguments:
            message: The received message
        """
        logger.info(
            f"================= client {self.ID} received finish message "
            f"=================")

        if message.content is not None:
            self.trainer.update(message.content,
                                strict=self._cfg.federate.share_local_model)

        self._monitor.finish_fl()

    def callback_funcs_for_converged(self, message: Message): #서버가 “조기 수렴(converged)” 신호를 보냈을 때, 로컬 모니터를 통해 더 이상의 업데이트를 멈춥니다.
        """
        The handling function for receiving the signal that the FL course \
        converged

        Arguments:
            message: The received message
        """
        self._monitor.global_converged()

    @classmethod
    def get_msg_handler_dict(cls):
        return cls().msg_handlers_str
