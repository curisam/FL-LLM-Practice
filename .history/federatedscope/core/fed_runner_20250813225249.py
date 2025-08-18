import abc
import logging

from collections import deque
import heapq

import numpy as np

from federatedscope.core.workers import Server, Client
from federatedscope.core.gpu_manager import GPUManager
from federatedscope.core.auxiliaries.model_builder import get_model
from federatedscope.core.auxiliaries.utils import get_resource_info, \
    get_ds_rank
from federatedscope.core.auxiliaries.feat_engr_builder import \
    get_feat_engr_wrapper

logger = logging.getLogger(__name__)


class BaseRunner(object): # Runner의 추상 기반 클래스. Server/Client 생성, 구성 검증, FL 실행 메서드 정의
    """
    This class is a base class to construct an FL course, which includes \
    ``_set_up()`` and ``run()``.

    Args:
        data: The data used in the FL courses, which are formatted as \
        ``{'ID':data}`` for standalone mode. More details can be found in \
        federatedscope.core.auxiliaries.data_builder .
        server_class: The server class is used for instantiating a ( \
        customized) server.
        client_class: The client class is used for instantiating a ( \
        customized) client.
        config: The configurations of the FL course.
        client_configs: The clients' configurations.

    Attributes:
        data: The data used in the FL courses, which are formatted as \
        ``{'ID':data}`` for standalone mode. More details can be found in \
        federatedscope.core.auxiliaries.data_builder .
        server: The instantiated server.
        client: The instantiate client(s).
        cfg : The configurations of the FL course.
        client_cfgs: The clients' configurations.
        mode: The run mode for FL, ``distributed`` or ``standalone``
        gpu_manager: manager of GPU resource
        resource_info: information of resource
    """
    def __init__(self,
                 data,
                 server_class=Server,
                 client_class=Client,
                 config=None,
                 client_configs=None):
        
        #data, server_class, client_class, cfg, client_cfgs 저장

        self.data = data #{client_id: ClientData} 형태의 딕셔너리 
        self.server_class = server_class
        self.client_class = client_class
        assert config is not None, \
            "When using Runner, you should specify the `config` para"   #config 인자가 None이면 바로 AssertionError를 내면서 프로그램을 멈춤.
        if not config.is_ready_for_run:
            config.ready_for_run()
        self.cfg = config
        self.client_cfgs = client_configs





        self.serial_num_for_msg = 0
        self.mode = self.cfg.federate.mode.lower()



        #unseen_clients 설정 (일부 client를 unseen으로 간주)
        #GPUManager, ResourceInfo, FeatEngrWrapper 등 초기화

        self.gpu_manager = GPUManager(gpu_available=self.cfg.use_gpu,
                                      specified_device=self.cfg.device) #사용할 gpu 자원 자동 선택 클래스
        


        self.unseen_clients_id = []
        self.feat_engr_wrapper_client, self.feat_engr_wrapper_server = \
            get_feat_engr_wrapper(config) #feature engineering이 필요한 경우 wrapping 함수 (보통 패스됨)
        

        if self.cfg.federate.unseen_clients_rate > 0: #무시
            self.unseen_clients_id = np.random.choice(
                np.arange(1, self.cfg.federate.client_num + 1),
                size=max(
                    1,
                    int(self.cfg.federate.unseen_clients_rate *
                        self.cfg.federate.client_num)),
                replace=False).tolist()
        # get resource information
        self.resource_info = get_resource_info(
            config.federate.resource_info_file) #None

        # Check the completeness of msg_handler. #check()를 통해 메시지 핸들러 연결성 시각화
        self.check() ##아무것도 안함 ##

        # Set up for Runner
        self._set_up()

    @abc.abstractmethod
    def _set_up(self): #server/client를 생성하고 세팅하는 함수
        """
        Set up and instantiate the client/server.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_server_args(self, resource_info, client_resource_info):
        """
        Get the args for instantiating the server.

        Args:
            resource_info: information of resource
            client_resource_info: information of client's resource

        Returns:
            (server_data, model, kw): None or data which server holds; model \
            to be aggregated; kwargs dict to instantiate the server.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _get_client_args(self, client_id, resource_info):
        """
        Get the args for instantiating the server.

        Args:
            client_id: ID of client
            resource_info: information of resource

        Returns:
            (client_data, kw): data which client holds; kwargs dict to \
            instantiate the client.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def run(self): #FL 실험의 시작점 (client join → main loop 수행 → 결과 반환)
        """
        Launch the FL course

        Returns:
            dict: best results during the FL course
        """
        raise NotImplementedError

    @property
    def ds_rank(self):
        return get_ds_rank()

    def _setup_server(self, resource_info=None, client_resource_info=None): # server 인스턴스 생성
        """
        Set up and instantiate the server.

        Args:
            resource_info: information of resource
            client_resource_info: information of client's resource

        Returns:
            Instantiate server.
        """



        assert self.server_class is not None, \
            "`server_class` cannot be None."
        self.server_id = 0
        server_data, model, kw = self._get_server_args(resource_info,
                                                       client_resource_info)
        
        """
        kw = {
            'shared_comm_queue': deque(),
            'resource_info': None,
            'client_resource_info': None
        }
        
        """

        self._server_device = self.gpu_manager.auto_choice()#“GPU를 쓸 수 있는 상황인가?”, “특정 GPU를 지정했는가?”를 보고 최종적으로 “CPU를 쓸 건지, GPU의 몇 번 장치를 쓸 건지”를 자동으로 결정
        
        server = self.server_class(
            ID=self.server_id,
            config=self.cfg,
            data=server_data,
            model=model,
            client_num=self.cfg.federate.client_num,
            total_round_num=self.cfg.federate.total_round_num,
            device=self._server_device,
            unseen_clients_id=self.unseen_clients_id,
            **kw)#state, strategy는 default 값이 정의되어 있어 생략 가능. unseen_clients_id는 LLMMultiLoRAServer.__init__()에는 정의되어 있지 않지만, **kwargs에 의해 받아들여짐. 그리고 Server.__init__()에서는 실제로 unseen_clients_id를 인자로 명시하고 있기 때문에 정상적으로 처리됨.
        
            #kwargs:  kw = {'shared_comm_queue': self.shared_comm_queue, 'resource_info': resource_info, 'client_resource_info': client_resource_info}
        if self.cfg.nbafl.use: #무시
            from federatedscope.core.trainers.trainer_nbafl import \
                wrap_nbafl_server
            wrap_nbafl_server(server)
        if self.cfg.vertical.use:#무시
            from federatedscope.vertical_fl.utils import wrap_vertical_server
            server = wrap_vertical_server(server, self.cfg)
        if self.cfg.fedswa.use:#무시
            from federatedscope.core.workers.wrapper import wrap_swa_server
            server = wrap_swa_server(server)
        logger.info('Server has been set up ... ')
        return self.feat_engr_wrapper_server(server)# = server

    def _setup_client(self,
                      client_id=-1,
                      client_model=None,
                      resource_info=None): # 각 client 인스턴스 생성
        """
        Set up and instantiate the client.

        Args:
            client_id: ID of client
            client_model: model of client
            resource_info: information of resource

        Returns:
            Instantiate client.
        """


        assert self.client_class is not None, \
            "`client_class` cannot be None"
        self.server_id = 0
        client_data, kw = self._get_client_args(client_id, resource_info) #client_data: 한 개의 ClientDataset 클래스.
        """
        kw = {
        'shared_comm_queue': self.shared_comm_queue,
        'resource_info': None
        }
        
        """



        client_specific_config = self.cfg.clone()
        if self.client_cfgs: #무시
            client_specific_config.defrost()
            client_specific_config.merge_from_other_cfg(
                self.client_cfgs.get('client_{}'.format(client_id)))
            client_specific_config.freeze()

        client_device = self._server_device if \
            self.cfg.federate.share_local_model else \
            self.gpu_manager.auto_choice()
        
        client = self.client_class(
            ID=client_id,
            server_id=self.server_id,
            config=client_specific_config,
            data=client_data,
            model=client_model or get_model(
                client_specific_config, client_data, backend=self.cfg.backend),
            device=client_device,
            is_unseen_client=client_id in self.unseen_clients_id,
            **kw)
 
        if self.cfg.vertical.use:  #무시
            from federatedscope.vertical_fl.utils import wrap_vertical_client
            client = wrap_vertical_client(client, config=self.cfg)

        if client_id == -1:
            logger.info('Client (address {}:{}) has been set up ... '.format(
                self.client_address['host'], self.client_address['port']))
        else:
            logger.info(f'Client {client_id} has been set up ... ')

        return self.feat_engr_wrapper_client(client) #=client

    def check(self):
        """
        Check the completeness of Server and Client.

        """

        if not self.cfg.check_completeness:
            return ####이걸로 걸림 ########
        try:
            import os
            import networkx as nx
            import matplotlib.pyplot as plt
            # Build check graph
            G = nx.DiGraph()
            flags = {0: 'Client', 1: 'Server'}
            msg_handler_dicts = [
                self.client_class.get_msg_handler_dict(),
                self.server_class.get_msg_handler_dict()
            ]
            for flag, msg_handler_dict in zip(flags.keys(), msg_handler_dicts):
                role, oppo = flags[flag], flags[(flag + 1) % 2]
                for msg_in, (handler, msgs_out) in \
                        msg_handler_dict.items():
                    for msg_out in msgs_out:
                        msg_in_key = f'{oppo}_{msg_in}'
                        handler_key = f'{role}_{handler}'
                        msg_out_key = f'{role}_{msg_out}'
                        G.add_node(msg_in_key, subset=1)
                        G.add_node(handler_key, subset=0 if flag else 2)
                        G.add_node(msg_out_key, subset=1)
                        G.add_edge(msg_in_key, handler_key)
                        G.add_edge(handler_key, msg_out_key)
            pos = nx.multipartite_layout(G)
            plt.figure(figsize=(20, 15))
            nx.draw(G,
                    pos,
                    with_labels=True,
                    node_color='white',
                    node_size=800,
                    width=1.0,
                    arrowsize=25,
                    arrowstyle='->')
            fig_path = os.path.join(self.cfg.outdir, 'msg_handler.png')
            plt.savefig(fig_path)
            if nx.has_path(G, 'Client_join_in', 'Server_finish'):
                if nx.is_weakly_connected(G):
                    logger.info(f'Completeness check passes! Save check '
                                f'results in {fig_path}.')
                else:
                    logger.warning(f'Completeness check raises warning for '
                                   f'some handlers not in FL process! Save '
                                   f'check results in {fig_path}.')
            else:
                logger.error(f'Completeness check fails for there is no'
                             f'path from `join_in` to `finish`! Save '
                             f'check results in {fig_path}.')
        except Exception as error:
            logger.warning(f'Completeness check failed for {error}!')
        return


class StandaloneRunner(BaseRunner): #BaseRunner를 상속하여 Standalone 모드 FL 실행을 구현
    def _set_up(self): #server/client를 생성하고 세팅하는 함수
        """
        To set up server and client for standalone mode.
        """
        self.is_run_online = True if self.cfg.federate.online_aggr else False   # False
        self.shared_comm_queue = deque() #메시지 게시판 느낌. 서버와 클라이언트가 공유

        if self.cfg.backend == 'torch':
            import torch
            torch.set_num_threads(1)


        server_resource_info = None
        client_resource_info = None


        self.server = self._setup_server(
            resource_info=server_resource_info,
            client_resource_info=client_resource_info)

        self.client = dict() #clientclass를 value로 가진다.
        # assume the client-wise data are consistent in their input&output
        # shape

        if self.cfg.federate.online_aggr: #False
            self._shared_client_model = get_model(
                self.cfg, self.data[1], backend=self.cfg.backend
            ) if self.cfg.federate.share_local_model else None
        else:
            self._shared_client_model = self.server.model \
                if self.cfg.federate.share_local_model else None # self.cfg.federate.share_local_model=True
        for client_id in range(1, self.cfg.federate.client_num + 1):
            self.client[client_id] = self._setup_client(
                client_id=client_id,
                client_model=self._shared_client_model,
                resource_info=client_resource_info[client_id - 1]
                if client_resource_info is not None else None)

        # in standalone mode, by default, we print the trainer info only
        # once for better logs readability
        trainer_representative = self.client[1].trainer
        if trainer_representative is not None and hasattr(
                trainer_representative, 'print_trainer_meta_info'): #True
            trainer_representative.print_trainer_meta_info() #<bound method Trainer.print_trainer_meta_info of <federatedscope.llm.trainer.reward_choice_trainer.RewardChoiceTrainer object at 0x7f3ca3f6c910>>  

    def _get_server_args(self, resource_info=None, client_resource_info=None):

        import ipdb; ipdb.set_trace(context=15)
        if self.server_id in self.data:
            server_data = self.data[self.server_id]
            model = get_model(self.cfg, server_data, backend=self.cfg.backend)
        else:
            server_data = None
            data_representative = self.data[1]
            model = get_model(
                self.cfg, data_representative, backend=self.cfg.backend
            )  # get the model according to client's data if the server
            # does not own data
        kw = {
            'shared_comm_queue': self.shared_comm_queue,
            'resource_info': resource_info,
            'client_resource_info': client_resource_info
        }
        return server_data, model, kw

    def _get_client_args(self, client_id=-1, resource_info=None):
        client_data = self.data[client_id]
        kw = {
            'shared_comm_queue': self.shared_comm_queue,
            'resource_info': resource_info
        }
        return client_data, kw

    def run(self): #FL 실험의 시작점 (client join → main loop 수행 → 결과 반환)

        #초기 설정(클라이언트에서 id 설정 완료)
        #client:"join_in"- >서바:"assign_client_id"   
        for each_client in self.client:
            # Launch each client
            self.client[each_client].join_in()
        

        #fl 학습 시작:

        #server: model_para-> client: model-para 반복

        #중간에

        #server: "evaluate": 주기적 평가 요청
        #client: "metrics". 로컬 평가 전송

        if self.is_run_online:
            self._run_simulation_online()
        else:
            self._run_simulation()


        # TODO: avoid using private attr
        #server: finish -> 최종 라운드 종료 신호. client에서 더이상 메시지 안보냄. 이로서 self._run_simulation()도 끝난 상황

        
        self.server._monitor.finish_fed_runner(fl_mode=self.mode) #클라이언트들의 평균/std 집계하여 "system_metrics.log"에 두 줄 추가.
        return self.server.best_results

    def _handle_msg(self, msg, rcv=-1): #메시지를 처리 (server or client에게 dispatch)
        """
        To simulate the message handling process (used only for the \
        standalone mode)
        """


        if rcv != -1: #언제나 -1로 받아 해당할 일 없을 거 같음.
            # simulate broadcast one-by-one
            self.client[rcv].msg_handlers[msg.msg_type](msg)
            return

        _, receiver = msg.sender, msg.receiver
        download_bytes, upload_bytes = msg.count_bytes() #download_bytes: 메시지 다운받을 때 크기, upload_bytes: 동일 메시지를 모든 reciver에게 전달하는데 드는 cost
        if not isinstance(receiver, list):
            receiver = [receiver]
        for each_receiver in receiver:
            if each_receiver == 0: #서버에서 처리
                self.server.msg_handlers[msg.msg_type](msg)
                self.server._monitor.track_download_bytes(download_bytes) #inplace로 self.total_download_bytes에 download_bytes 추가
            else: #클라이언트에서 처리
                self.client[each_receiver].msg_handlers[msg.msg_type](msg)
                self.client[each_receiver]._monitor.track_download_bytes(
                    download_bytes) #inplace로 self.total_download_bytes에 download_bytes 추가

    def _run_simulation_online(self): #online aggregation 지원 (client마다 순차적 broadcast 처리). 해당 안함.
        """
        Run for online aggregation.
        Any broadcast operation would be executed client-by-clien to avoid \
        the existence of #clients messages at the same time. Currently, \
        only consider centralized topology \
        """
        def is_broadcast(msg):
            return len(msg.receiver) >= 1 and msg.sender == 0

        cached_bc_msgs = []
        cur_idx = 0
        while True:
            if len(self.shared_comm_queue) > 0:
                msg = self.shared_comm_queue.popleft()
                if is_broadcast(msg):
                    cached_bc_msgs.append(msg)
                    # assume there is at least one client
                    msg = cached_bc_msgs[0]
                    self._handle_msg(msg, rcv=msg.receiver[cur_idx])
                    cur_idx += 1
                    if cur_idx >= len(msg.receiver):
                        del cached_bc_msgs[0]
                        cur_idx = 0
                else:
                    self._handle_msg(msg)
            elif len(cached_bc_msgs) > 0:
                msg = cached_bc_msgs[0]
                self._handle_msg(msg, rcv=msg.receiver[cur_idx])
                cur_idx += 1
                if cur_idx >= len(msg.receiver):
                    del cached_bc_msgs[0]
                    cur_idx = 0
            else:
                # finished
                break

    def _run_simulation(self): #메시지 스케줄러 역할->실제 일(학습·집계)은 Server._handle_msg, Client._handle_msg 안에서.
        """
        Run for standalone simulation (W/O online aggr)
        """
        server_msg_cache = list()
        while True:
            if len(self.shared_comm_queue) > 0:  # (A) 메시지가 큐에 존재할 때. 공유 메시지 큐를 전부 비워내는게 목표

                """
                ### 메시지가 shared_comm_queue에 존재 → 가장 일반적인 루트 ###

                클라이언트가 서버로 보내는 메시지 (예: model_para)

                서버가 클라이언트로 보내는 메시지 (예: sync_model)

                클라이언트 join-in 요청, early stop 알림 등등도 여기에 포함됨

                #########  동작 ###########
                메시지의 receiver가 [self.server_id]면 → server_msg_cache에 저장

                아니면 → 바로 _handle_msg() 호출 (클라이언트나 다른 모듈에게)

                """

                #타임스탬프가 아니라 도착(append) 순이 처리 순서의 기준

                #self.shared_comm_queue: 서버와 클라이언트가 주고받는 모든 Message들을 담는 공통 큐
                #shared_comm_queue (collections.deque) 에 append() 로 들어온 순서대로(=FIFO) 꺼냄과 동시에 self.shared_comm_queue 에서 제거
                msg = self.shared_comm_queue.popleft()  


                # 1) 서버로 향하는 i.e. 서버가 처리해야 하는  메시지인지 판별
                if  msg.receiver == [self.server_id]: 
                    # For the server, move the received message to a
                    # cache for reordering the messages according to
                    # the timestamps

                    # ① 서버 앞으로 온 메시지인지 확인
                        #    msg.receiver == [self.server_id] 인 메시지는 서버가 처리해야 할 메시지입니다.


                    # ② 순서를 붙여 우선순위 큐에 넣기 전에 일련번호 할당
                         #    여러 클라이언트에서 거의 동시에 온 메시지들이 있을 때, 동일한 타임스탬프(timestamp) 를 가진 메시지끼리는 어느 것을 먼저 처리해야 할지 결정할 기준이 필요합니다.
                         #    이렇게 하면 (타임스탬프, 순번) 쌍으로 메시지들의 우선순위를 정할 수 있게 되고, 순서가 보장됩니다.                 
                    msg.serial_num = self.serial_num_for_msg
                    self.serial_num_for_msg += 1

                    # ③ heapq(최소 힙)에 메시지 저장
                    heapq.heappush(server_msg_cache, msg) #꺼낸 msg 객체가 server_msg_cache안에 추가(append). heappush 과정에서 내부적으로 여러 번 msg_a < msg_b 를 비교하면서 “최소 속성” 순으로 노드를 배치. heappop 하면, 위 __lt__ 로 정의된 순서에 따라 (timestamp, serial_num) 가 가장 작은 메시지가 꺼내집니다


                # → 클라이언트로 향하는 메시지일 땐
                #   즉시 해당  클라이언트의 핸들러로 전달
                else: 
                    self._handle_msg(msg) #클라이언트의 핸들러로 바로 처리하면서 비워낸다.


            elif len(server_msg_cache) > 0: # (B) shared_comm_queue는 비었지만 서버 메시지 캐시에 남은 게 있을 때. 서버측에서 처리할 것. 즉 클라이언트에서 보낸것을 서버가 다 파악하여 server_msg_cache에 담은 상황.

                """
                ### shared_comm_queue는 비었지만 서버용 메시지 캐시에 쌓여 있을 때 ###

                여러 클라이언트의 메시지가 동시에 도착했고 → 시간 순 재정렬을 위해 server_msg_cache에 들어간 상태

                위 (A) 단계에서 shared_comm_queue는 비었지만 아직 server 쪽 처리가 끝나지 않음

                #########  상황 ###########
                일반적인 FedAvg: 클라이언트 K명 중 일부만 먼저 업데이트를 보냄

                FedBiscuit: 각 클라이언트가 adapter N개를 학습 후 메시지 보냄 → server_msg_cache로 들어감

                동기 FL이면 모든 메시지 수집 후 aggregate() 실행

                """
                # 2) 서버용 메시지가 쌓여 있으면, 우선순위 큐에서 꺼내 처리
                msg = heapq.heappop(server_msg_cache)#메시지를 우선순위 큐(priority queue) 형태로 관리” 하기 위해서 heapq를 쓴다.


                self._handle_msg(msg) #서버의 핸들러로 처리

            else: # (C) 큐가 완전히 비었을 때 → 종료 조건 확인
                """
                ### 종료 조건 체크용 (남은 작업이 있는지 확인) ###

                shared_comm_queue도 비었고

                server_msg_cache도 모두 비었을 때

                #########  상황 ###########
                FL 전체 라운드가 끝나서 클라이언트와 서버가 더 이상 메시지를 생성하지 않을 때

                또는 비동기 FL에서 모든 클라이언트가 타임아웃돼 집계가 더 이상 불가능할 때

                """


                # terminate when shared_comm_queue and
                # server_msg_cache are all empty
                break


class DistributedRunner(BaseRunner):
    def _set_up(self):
        """
        To set up server or client for distributed mode.
        """
        # sample resource information
        if self.resource_info is not None:
            sampled_index = np.random.choice(list(self.resource_info.keys()))
            sampled_resource = self.resource_info[sampled_index]
        else:
            sampled_resource = None

        self.server_address = {
            'host': self.cfg.distribute.server_host,
            'port': self.cfg.distribute.server_port + self.ds_rank
        }
        if self.cfg.distribute.role == 'server':
            self.server = self._setup_server(resource_info=sampled_resource)
        elif self.cfg.distribute.role == 'client':
            # When we set up the client in the distributed mode, we assume
            # the server has been set up and number with #0
            self.client_address = {
                'host': self.cfg.distribute.client_host,
                'port': self.cfg.distribute.client_port + self.ds_rank
            }
            self.client = self._setup_client(resource_info=sampled_resource)

    def _get_server_args(self, resource_info, client_resource_info):
        server_data = self.data
        model = get_model(self.cfg, server_data, backend=self.cfg.backend)
        kw = self.server_address
        kw.update({'resource_info': resource_info})
        return server_data, model, kw

    def _get_client_args(self, client_id, resource_info):
        client_data = self.data
        kw = self.client_address
        kw['server_host'] = self.server_address['host']
        kw['server_port'] = self.server_address['port']
        kw['resource_info'] = resource_info
        return client_data, kw

    def run(self):
        if self.cfg.distribute.role == 'server':
            self.server.run()
            return self.server.best_results
        elif self.cfg.distribute.role == 'client':
            self.client.join_in()
            self.client.run()


# TODO: remove FedRunner (keep now for forward compatibility)
class FedRunner(object):
    """
    This class is used to construct an FL course, which includes `_set_up`
    and `run`.

    Arguments:
        data: The data used in the FL courses, which are formatted as \
        ``{'ID':data}`` for standalone mode. More details can be found in \
        federatedscope.core.auxiliaries.data_builder .
        server_class: The server class is used for instantiating a ( \
        customized) server.
        client_class: The client class is used for instantiating a ( \
        customized) client.
        config: The configurations of the FL course.
        client_configs: The clients' configurations.

    Warnings:
        ``FedRunner`` will be removed in the future, consider \
        using ``StandaloneRunner`` or ``DistributedRunner`` instead!
    """
    def __init__(self,
                 data,
                 server_class=Server,
                 client_class=Client,
                 config=None,
                 client_configs=None):
        logger.warning('`federate.core.fed_runner.FedRunner` will be '
                       'removed in the future, please use'
                       '`federate.core.fed_runner.get_runner` to get '
                       'Runner.')
        self.data = data
        self.server_class = server_class
        self.client_class = client_class
        assert config is not None, \
            "When using FedRunner, you should specify the `config` para"
        if not config.is_ready_for_run:
            config.ready_for_run()
        self.cfg = config
        self.client_cfgs = client_configs

        self.mode = self.cfg.federate.mode.lower() #standalone
        self.gpu_manager = GPUManager(gpu_available=self.cfg.use_gpu,
                                      specified_device=self.cfg.device) 

        self.unseen_clients_id = []
        if self.cfg.federate.unseen_clients_rate > 0:
            self.unseen_clients_id = np.random.choice(
                np.arange(1, self.cfg.federate.client_num + 1),
                size=max(
                    1,
                    int(self.cfg.federate.unseen_clients_rate *
                        self.cfg.federate.client_num)),
                replace=False).tolist()
        # get resource information
        self.resource_info = get_resource_info(
            config.federate.resource_info_file)

        # Check the completeness of msg_handler.
        self.check()

    def setup(self):
        if self.mode == 'standalone':
            self.shared_comm_queue = deque()
            self._setup_for_standalone()
            # in standalone mode, by default, we print the trainer info only
            # once for better logs readability
            trainer_representative = self.client[1].trainer
            if trainer_representative is not None:
                trainer_representative.print_trainer_meta_info()
        elif self.mode == 'distributed':
            self._setup_for_distributed()

    def _setup_for_standalone(self):
        """
        To set up server and client for standalone mode.
        """
        if self.cfg.backend == 'torch':
            import torch
            torch.set_num_threads(1)

        assert self.cfg.federate.client_num != 0, \
            "In standalone mode, self.cfg.federate.client_num should be " \
            "non-zero. " \
            "This is usually cased by using synthetic data and users not " \
            "specify a non-zero value for client_num"

        if self.cfg.federate.method == "global":
            self.cfg.defrost()
            self.cfg.federate.client_num = 1
            self.cfg.federate.sample_client_num = 1
            self.cfg.freeze()

        # sample resource information
        if self.resource_info is not None:
            if len(self.resource_info) < self.cfg.federate.client_num + 1:
                replace = True
                logger.warning(
                    f"Because the provided the number of resource information "
                    f"{len(self.resource_info)} is less than the number of "
                    f"participants {self.cfg.federate.client_num+1}, one "
                    f"candidate might be selected multiple times.")
            else:
                replace = False
            sampled_index = np.random.choice(
                list(self.resource_info.keys()),
                size=self.cfg.federate.client_num + 1,
                replace=replace)
            server_resource_info = self.resource_info[sampled_index[0]]
            client_resource_info = [
                self.resource_info[x] for x in sampled_index[1:]
            ]
        else:
            server_resource_info = None
            client_resource_info = None

        self.server = self._setup_server(
            resource_info=server_resource_info,
            client_resource_info=client_resource_info)

        self.client = dict()

        # assume the client-wise data are consistent in their input&output
        # shape
        self._shared_client_model = get_model(
            self.cfg, self.data[1], backend=self.cfg.backend
        ) if self.cfg.federate.share_local_model else None

        for client_id in range(1, self.cfg.federate.client_num + 1):
            self.client[client_id] = self._setup_client(
                client_id=client_id,
                client_model=self._shared_client_model,
                resource_info=client_resource_info[client_id - 1]
                if client_resource_info is not None else None)

    def _setup_for_distributed(self):
        """
        To set up server or client for distributed mode.
        """

        # sample resource information
        if self.resource_info is not None:
            sampled_index = np.random.choice(list(self.resource_info.keys()))
            sampled_resource = self.resource_info[sampled_index]
        else:
            sampled_resource = None

        self.server_address = {
            'host': self.cfg.distribute.server_host,
            'port': self.cfg.distribute.server_port
        }
        if self.cfg.distribute.role == 'server':
            self.server = self._setup_server(resource_info=sampled_resource)
        elif self.cfg.distribute.role == 'client':
            # When we set up the client in the distributed mode, we assume
            # the server has been set up and number with #0
            self.client_address = {
                'host': self.cfg.distribute.client_host,
                'port': self.cfg.distribute.client_port
            }
            self.client = self._setup_client(resource_info=sampled_resource)

    def run(self):
        """
        To run an FL course, which is called after server/client has been
        set up.
        For the standalone mode, a shared message queue will be set up to
        simulate ``receiving message``.
        """
        self.setup()
        if self.mode == 'standalone':
            # trigger the FL course
            for each_client in self.client:
                self.client[each_client].join_in()
            if self.cfg.federate.online_aggr:
                # any broadcast operation would be executed client-by-client
                # to avoid the existence of #clients messages at the same time.
                # currently, only consider centralized topology
                self._run_simulation_online()

            else:
                self._run_simulation()

            self.server._monitor.finish_fed_runner(fl_mode=self.mode)

            return self.server.best_results

        elif self.mode == 'distributed':
            if self.cfg.distribute.role == 'server':
                self.server.run()
                return self.server.best_results
            elif self.cfg.distribute.role == 'client':
                self.client.join_in()
                self.client.run()

    def _run_simulation_online(self):
        def is_broadcast(msg):
            return len(msg.receiver) >= 1 and msg.sender == 0

        cached_bc_msgs = []
        cur_idx = 0
        while True:

            if len(self.shared_comm_queue) > 0:

                msg = self.shared_comm_queue.popleft()
                if is_broadcast(msg):
                    cached_bc_msgs.append(msg)
                    # assume there is at least one client
                    msg = cached_bc_msgs[0]
                    self._handle_msg(msg, rcv=msg.receiver[cur_idx])
                    cur_idx += 1
                    if cur_idx >= len(msg.receiver):
                        del cached_bc_msgs[0]
                        cur_idx = 0
                else:
                    self._handle_msg(msg)
            elif len(cached_bc_msgs) > 0:

                msg = cached_bc_msgs[0]
                self._handle_msg(msg, rcv=msg.receiver[cur_idx])
                cur_idx += 1
                if cur_idx >= len(msg.receiver):
                    del cached_bc_msgs[0]
                    cur_idx = 0
            else:
                # finished
                break

    def _run_simulation(self):
        server_msg_cache = list()
        while True:
            if len(self.shared_comm_queue) > 0:
                msg = self.shared_comm_queue.popleft()
                if msg.receiver == [self.server_id]:
                    # For the server, move the received message to a
                    # cache for reordering the messages according to
                    # the timestamps
                    heapq.heappush(server_msg_cache, msg)
                else:
                    self._handle_msg(msg)
            elif len(server_msg_cache) > 0:
                msg = heapq.heappop(server_msg_cache)
                if self.cfg.asyn.use and self.cfg.asyn.aggregator \
                        == 'time_up':
                    # When the timestamp of the received message beyond
                    # the deadline for the currency round, trigger the
                    # time up event first and push the message back to
                    # the cache
                    if self.server.trigger_for_time_up(msg.timestamp):
                        heapq.heappush(server_msg_cache, msg)
                    else:
                        self._handle_msg(msg)
                else:
                    self._handle_msg(msg)
            else:
                if self.cfg.asyn.use and self.cfg.asyn.aggregator \
                        == 'time_up':
                    self.server.trigger_for_time_up()
                    if len(self.shared_comm_queue) == 0 and \
                            len(server_msg_cache) == 0:
                        break
                else:
                    # terminate when shared_comm_queue and
                    # server_msg_cache are all empty
                    break

    def _setup_server(self, resource_info=None, client_resource_info=None):
        """
        Set up the server
        """
        self.server_id = 0
        if self.mode == 'standalone':
            if self.server_id in self.data:
                server_data = self.data[self.server_id]
                model = get_model(self.cfg,
                                  server_data,
                                  backend=self.cfg.backend)
            else:
                server_data = None
                data_representative = self.data[1]
                model = get_model(
                    self.cfg, data_representative, backend=self.cfg.backend
                )  # get the model according to client's data if the server
                # does not own data
            kw = {
                'shared_comm_queue': self.shared_comm_queue,
                'resource_info': resource_info,
                'client_resource_info': client_resource_info
            }
        elif self.mode == 'distributed':
            server_data = self.data
            model = get_model(self.cfg, server_data, backend=self.cfg.backend)
            kw = self.server_address
            kw.update({'resource_info': resource_info})
        else:
            raise ValueError('Mode {} is not provided'.format(
                self.cfg.mode.type))

        if self.server_class:
            self._server_device = self.gpu_manager.auto_choice()
            server = self.server_class(
                ID=self.server_id,
                config=self.cfg,
                data=server_data,
                model=model,
                client_num=self.cfg.federate.client_num,
                total_round_num=self.cfg.federate.total_round_num,
                device=self._server_device,
                unseen_clients_id=self.unseen_clients_id,
                **kw)

            if self.cfg.nbafl.use:
                from federatedscope.core.trainers.trainer_nbafl import \
                    wrap_nbafl_server
                wrap_nbafl_server(server)

        else:
            raise ValueError

        logger.info('Server has been set up ... ')

        return server

    def _setup_client(self,
                      client_id=-1,
                      client_model=None,
                      resource_info=None):
        """
        Set up the client
        """
        self.server_id = 0
        if self.mode == 'standalone':
            client_data = self.data[client_id]
            kw = {
                'shared_comm_queue': self.shared_comm_queue,
                'resource_info': resource_info
            }
        elif self.mode == 'distributed':
            client_data = self.data
            kw = self.client_address
            kw['server_host'] = self.server_address['host']
            kw['server_port'] = self.server_address['port']
            kw['resource_info'] = resource_info
        else:
            raise ValueError('Mode {} is not provided'.format(
                self.cfg.mode.type))

        if self.client_class:
            client_specific_config = self.cfg.clone()
            if self.client_cfgs and \
                    self.client_cfgs.get('client_{}'.format(client_id)):
                client_specific_config.defrost()
                client_specific_config.merge_from_other_cfg(
                    self.client_cfgs.get('client_{}'.format(client_id)))
                client_specific_config.freeze()
            client_device = self._server_device if \
                self.cfg.federate.share_local_model else \
                self.gpu_manager.auto_choice()
            client = self.client_class(ID=client_id,
                                       server_id=self.server_id,
                                       config=client_specific_config,
                                       data=client_data,
                                       model=client_model
                                       or get_model(client_specific_config,
                                                    client_data,
                                                    backend=self.cfg.backend),
                                       device=client_device,
                                       is_unseen_client=client_id
                                       in self.unseen_clients_id,
                                       **kw)
        else:
            raise ValueError

        if client_id == -1:
            logger.info('Client (address {}:{}) has been set up ... '.format(
                self.client_address['host'], self.client_address['port']))
        else:
            logger.info(f'Client {client_id} has been set up ... ')

        return client

    def _handle_msg(self, msg, rcv=-1):
        """
        To simulate the message handling process (used only for the
        standalone mode)
        """
        if rcv != -1:
            # simulate broadcast one-by-one
            self.client[rcv].msg_handlers[msg.msg_type](msg)
            return

        _, receiver = msg.sender, msg.receiver
        download_bytes, upload_bytes = msg.count_bytes()
        if not isinstance(receiver, list):
            receiver = [receiver]
        for each_receiver in receiver:
            if each_receiver == 0:#server
                self.server.msg_handlers[msg.msg_type](msg)
                self.server._monitor.track_download_bytes(download_bytes)
            else:#client
                self.client[each_receiver].msg_handlers[msg.msg_type](msg)
                self.client[each_receiver]._monitor.track_download_bytes(
                    download_bytes)

    def check(self):
        """
        Check the completeness of Server and Client.

        """
        if not self.cfg.check_completeness:
            return
        try:
            import os
            import networkx as nx
            import matplotlib.pyplot as plt
            # Build check graph
            G = nx.DiGraph()
            flags = {0: 'Client', 1: 'Server'}
            msg_handler_dicts = [
                self.client_class.get_msg_handler_dict(),
                self.server_class.get_msg_handler_dict()
            ]
            for flag, msg_handler_dict in zip(flags.keys(), msg_handler_dicts):
                role, oppo = flags[flag], flags[(flag + 1) % 2]
                for msg_in, (handler, msgs_out) in \
                        msg_handler_dict.items():
                    for msg_out in msgs_out:
                        msg_in_key = f'{oppo}_{msg_in}'
                        handler_key = f'{role}_{handler}'
                        msg_out_key = f'{role}_{msg_out}'
                        G.add_node(msg_in_key, subset=1)
                        G.add_node(handler_key, subset=0 if flag else 2)
                        G.add_node(msg_out_key, subset=1)
                        G.add_edge(msg_in_key, handler_key)
                        G.add_edge(handler_key, msg_out_key)
            pos = nx.multipartite_layout(G)
            plt.figure(figsize=(20, 15))
            nx.draw(G,
                    pos,
                    with_labels=True,
                    node_color='white',
                    node_size=800,
                    width=1.0,
                    arrowsize=25,
                    arrowstyle='->')
            fig_path = os.path.join(self.cfg.outdir, 'msg_handler.png')
            plt.savefig(fig_path)
            if nx.has_path(G, 'Client_join_in', 'Server_finish'):
                if nx.is_weakly_connected(G):
                    logger.info(f'Completeness check passes! Save check '
                                f'results in {fig_path}.')
                else:
                    logger.warning(f'Completeness check raises warning for '
                                   f'some handlers not in FL process! Save '
                                   f'check results in {fig_path}.')
            else:
                logger.error(f'Completeness check fails for there is no'
                             f'path from `join_in` to `finish`! Save '
                             f'check results in {fig_path}.')
        except Exception as error:
            logger.warning(f'Completeness check failed for {error}!')
        return
