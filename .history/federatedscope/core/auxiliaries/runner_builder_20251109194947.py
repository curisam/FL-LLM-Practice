from federatedscope.core.fed_runner import StandaloneRunner, DistributedRunner
from federatedscope.core.parallel.parallel_runner import \
    StandaloneMultiGPURunner


def get_runner(server_class,
               client_class,
               config,
               client_configs=None,
               data=None):
    """
    Instantiate a runner based on a configuration file

    Args:
        server_class: server class
        client_class: client class
        config: configurations for FL, see ``federatedscope.core.configs``
        client_configs: client-specific configurations

    Returns:
        An instantiated FedRunner to run the FL course.

    Note:
      The key-value pairs of built-in runner and source are shown below:
        =============================  ===============================
        Mode                                          Source
        =============================  ===============================
        ``standalone``                 ``core.fed_runner.StandaloneRunner``
        ``distributed``                ``core.fed_runner.DistributedRunner``
        ``standalone(process_num>1)``  ``core.auxiliaries.parallel_runner.``
                                       ``StandaloneMultiGPURunner``
        =============================  ===============================
    """

    mode = config.federate.mode.lower()
    process_num = config.federate.process_num
    if mode == 'standalone':

        runner_cls = StandaloneRunner

        # # 단일 프로세스(또는 단일 GPU) → StandaloneRunner
        # if process_num <= 1:
        #     runner_cls = StandaloneRunner
        # # 다중 GPU로 병렬 실행하고 싶으면 → StandaloneMultiGPURunner
        # else:# GPU수가 CLIENT 수보다 적으니 지양
        #     runner_cls = StandaloneMultiGPURunner #client를 여러 GPU에 분산 (e.g., 0번 GPU에 client 1, 1번 GPU에 client 2)
    elif mode == 'distributed':
        # 진짜 분산 환경(gRPC 등) → DistributedRunner
        runner_cls = DistributedRunner

    # Multi-GPU standalone 의 경우 data를 다시 로드할 수 있도록 None
    if runner_cls is StandaloneMultiGPURunner:
        data = None

    return runner_cls(data=data,
                      server_class=server_class,
                      client_class=client_class,
                      config=config,
                      client_configs=client_configs)
