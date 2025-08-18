from federatedscope.core.monitors.monitor import Monitor
from federatedscope.core.auxiliaries.utils import get_ds_rank


class Worker(object):
    """
    The base worker class, the parent of ``BaseClient`` and ``BaseServer``

    Args:
        ID: ID of worker
        state: the training round index
        config: the configuration of FL course
        model: the model maintained locally

    Attributes:
        ID: ID of worker
        state: the training round index
        model: the model maintained locally
        cfg: the configuration of FL course
        mode: the run mode for FL, ``distributed`` or ``standalone``
        monitor: monite FL course and record metrics
    """
    def __init__(self, ID=-1, state=0, config=None, model=None, strategy=None):

        """
        self._ID	현재 워커의 ID (Client 또는 Server)
        self._state	현재 진행 중인 라운드 번호 (FL Round)
        self._cfg	전체 FL 설정 정보 (federated, train, eval, 등 포함된 cfg 객체)
        self._model	로컬에서 사용하는 모델
        self._strategy	커스텀 aggregation 전략 등이 있을 경우 설정
        self._mode	"standalone" vs "distributed" 모드
        self._monitor	MetricCalculator와 연동하여 loss, acc, flops 등 추적
        """

        self._ID = ID
        self._state = state

        self._model = model
        self._cfg = config
        self._strategy = strategy
        if self._cfg is not None:
            self._mode = self._cfg.federate.mode.lower()
            self._monitor = Monitor(config, monitored_object=self)

    @property
    def ID(self):
        return self._ID

    @ID.setter
    def ID(self, value):
        self._ID = value

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, value):
        self._strategy = value

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        self._mode = value

    @property
    def ds_rank(self):
        return get_ds_rank()
