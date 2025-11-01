import logging
from federatedscope.core.configs import constants

logger = logging.getLogger(__name__)


def get_aggregator(method, model=None, device=None, online=False, config=None):
    """
    This function builds an aggregator, which is a protocol for aggregate \
    all clients' model(s).

    Arguments:
        method: key to determine which aggregator to use
        model:  model to be aggregated
        device: where to aggregate models (``cpu`` or ``gpu``)
        online: ``True`` or ``False`` to use online aggregator.
        config: configurations for FL, see ``federatedscope.core.configs``

    Returns:
        An instance of aggregator (see ``core.aggregator`` for details)

    Note:
      The key-value pairs of ``method`` and aggregators:
        ==================================  ===========================
        Method                              Aggregator
        ==================================  ===========================
        ``tensorflow``                      ``cross_backends.FedAvgAggregator``
        ``local``                           \
        ``core.aggregators.NoCommunicationAggregator``
        ``global``                          \
        ``core.aggregators.NoCommunicationAggregator``
        ``fedavg``                          \
        ``core.aggregators.OnlineClientsAvgAggregator`` or \
        ``core.aggregators.AsynClientsAvgAggregator`` or \
        ``ClientsAvgAggregator``
        ``pfedme``                          \
        ``core.aggregators.ServerClientsInterpolateAggregator``
        ``ditto``                           \
        ``core.aggregators.OnlineClientsAvgAggregator`` or \
        ``core.aggregators.AsynClientsAvgAggregator`` or \
        ``ClientsAvgAggregator``
        ``fedsageplus``                     \
        ``core.aggregators.OnlineClientsAvgAggregator`` or \
        ``core.aggregators.AsynClientsAvgAggregator`` or \
        ``ClientsAvgAggregator``
        ``gcflplus``                        \
        ``core.aggregators.OnlineClientsAvgAggregator`` or \
        ``core.aggregators.AsynClientsAvgAggregator`` or \
        ``ClientsAvgAggregator``
        ``fedopt``                          \
        ``core.aggregators.FedOptAggregator``
        ==================================  ===========================
    """

    # 1) server_type를 먼저 확인
    server_type = getattr(config.federate, 'server_type', '').lower()
    if server_type == 'llmfullmoeserver':
        from federatedscope.llm.aggregator.full_moe_aggregator import FullMoEAggregator
        return FullMoEAggregator(model=model, device=device, config=config)
    if server_type == 'llmfusionmoeserver':
        from federatedscope.llm.aggregator.fusion_moe_aggregator import FusionMoEAggregator
        return FusionMoEAggregator(model=model, device=device, config=config)



    if config.backend == 'tensorflow': #False, torch이다.
        from federatedscope.cross_backends import FedAvgAggregator
        return FedAvgAggregator(model=model, device=device)
    else:
        from federatedscope.core.aggregators import ClientsAvgAggregator, \
            OnlineClientsAvgAggregator, ServerClientsInterpolateAggregator, \
            FedOptAggregator, NoCommunicationAggregator, \
            AsynClientsAvgAggregator, KrumAggregator, \
            MedianAggregator, TrimmedmeanAggregator, \
            BulyanAggregator,  NormboundingAggregator
        from federatedscope.llm.llm_local.aggregator import \
            MultiLoRAAvgAggregator

    STR2AGG = {
        'fedavg': ClientsAvgAggregator,
        'krum': KrumAggregator,
        'median': MedianAggregator,
        'bulyan': BulyanAggregator,
        'trimmedmean': TrimmedmeanAggregator,
        'normbounding': NormboundingAggregator
    }




    if method.lower() in constants.AGGREGATOR_TYPE: #fedavg이다.
        aggregator_type = constants.AGGREGATOR_TYPE[method.lower()] #"clients_avg" or "no_communication"
    else: #pass
        aggregator_type = "clients_avg"
        logger.warning(
            'Aggregator for method {} is not implemented. Will use default one'
            .format(method))

    if config.data.type.lower() == 'hetero_nlp_tasks' and \
            not config.federate.atc_vanilla: #PASS
        from federatedscope.nlp.hetero_tasks.aggregator import ATCAggregator
        return ATCAggregator(model=model, config=config, device=device)

    if config.fedopt.use or aggregator_type == 'fedopt': #PASS
        return FedOptAggregator(config=config, model=model, device=device)
    elif aggregator_type == 'clients_avg': #True
        if online: #PASS
            return OnlineClientsAvgAggregator(
                model=model,
                device=device,
                config=config,
                src_device=device
                if config.federate.share_local_model else 'cpu')
        elif config.asyn.use: #PASS
            return AsynClientsAvgAggregator(model=model,
                                            device=device,
                                            config=config)
        elif config.llm.adapter.count > 1: #<---------------------------------------------FedBiscuit
            return MultiLoRAAvgAggregator(model=model,
                                          device=device,
                                          config=config)
        else:
            if config.aggregator.robust_rule not in STR2AGG:  #config.aggregator.robust_rule=fedavg
                logger.warning(
                    f'The specified {config.aggregator.robust_rule} aggregtion\
                    rule has not been supported, the vanilla fedavg algorithm \
                    will be used instead.')
            return STR2AGG.get(config.aggregator.robust_rule,
                               ClientsAvgAggregator)(model=model,
                                                     device=device,
                                                     config=config)  #config.aggregator.robust_rule가 STR2AGG의 key가 없을떄 ClientsAvgAggregator를 retrun한다. #ClientsAvgAggregator가 결국 쓰임.<---------------------------------------------FedBis

    elif aggregator_type == 'server_clients_interpolation': #PASS
        return ServerClientsInterpolateAggregator(
            model=model,
            device=device,
            config=config,
            beta=config.personalization.beta)
    elif aggregator_type == 'no_communication': #<---------------------------------------------LOCAL ONLY의 경우 여기 걸릴듯!! 
        return NoCommunicationAggregator(model=model, 
                                         device=device,
                                         config=config)
    else:
        raise NotImplementedError(
            "Aggregator {} is not implemented.".format(aggregator_type))
