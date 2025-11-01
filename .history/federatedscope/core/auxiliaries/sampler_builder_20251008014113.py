import logging

from federatedscope.core.sampler import UniformSampler, GroupSampler, \
    ResponsivenessRealtedSampler

logger = logging.getLogger(__name__)


    
def get_sampler(sample_strategy='uniform', client_num=None, client_info=None, bins=10, config=None):

    """
    This function builds a sampler for sampling clients who should join the \
    aggregation per communication round.

    Args:
        sample_strategy: Sampling strategy of sampler
        client_num: total number of client joining the FL course
        client_info: client information
        bins: size of bins for group sampler

    Returns:
        An instantiated Sampler to sample during aggregation.

    Note:
      The key-value pairs of built-in sampler and source are shown below:
        ===================================  ==============================
        Sampling strategy                    Source
        ===================================  ==============================
        ``uniform``                          ``core.sampler.UniformSampler``
        ``group``                            ``core.sampler.GroupSampler``
        ===================================  ==============================
    """
    if sample_strategy == 'uniform':
        return UniformSampler(client_num=client_num)
    elif sample_strategy == 'responsiveness':
        return ResponsivenessRealtedSampler(client_num=client_num,
                                            client_info=client_info)
    elif sample_strategy == 'group':
        return GroupSampler(client_num=client_num,
                            client_info=client_info,
                            bins=bins)
    

    elif sample_strategy == 'cluster':
        assert config is not None, "cluster sampler에는 config가 필요합니다."
        clusters = list(getattr(config.llm.adapter, 'clusters'))  # 0-based
        boundaries = list(getattr(config.llm.adapter, 'boundaries'))
        s_per = list(getattr(config.llm.adapter, 'sample_num_per_adapter')) 
        from federatedscope.core.sampler import ClusterUniformSampler
        return ClusterUniformSampler(client_num=client_num,
                                   clusters=clusters,
                                   boundaries=boundaries,
                                   sample_num_per_adapter=s_per)

    else:
        raise ValueError(
            f"The sample strategy {sample_strategy} has not been provided.")
