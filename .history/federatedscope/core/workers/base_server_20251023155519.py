import abc
from federatedscope.core.workers import Worker


class BaseServer(Worker): #FL 서버의 추상 베이스 클래스로 메시지 처리 함수 등록 기능 제공, 메시지 타입 → 처리 함수 매핑 구조 제공
    def __init__(self, ID, state, config, model, strategy):
        super(BaseServer, self).__init__(ID, state, config, model, strategy)
        #메시지 핸들러 관리용 딕셔너리 초기화, 기본 메시지 처리 구조를 위한 틀만 제공함.
        self.msg_handlers = dict() #메시지 타입 → 처리 함수 연결 딕셔너리
        self.msg_handlers_str = dict() #메시지 처리 함수명 및 후속 메시지 구조 정의용

    def register_handlers(self, msg_type, callback_func, send_msg=[None]): #서버가 어떤 메시지를 어떻게 처리할지를 등록하는 매핑 함수
        """
        To bind a message type with a handling function.

        Arguments:
            msg_type (str): The defined message type
            callback_func: The handling functions to handle the received \
                message
        """
        self.msg_handlers[msg_type] = callback_func #특정 메시지 타입에 대해 어떤 함수로 처리할지 매핑.
        self.msg_handlers_str[msg_type] = (callback_func.__name__, send_msg) #디버깅이나 로깅용으로 함수 이름과 전송 메시지 리스트를 문자열로 저장.

    def _register_default_handlers(self):
        """
        Register default handler dic to handle message, which includes \
        sender, receiver, state, and content. More detail can be found in \
        ``federatedscope.core.message``.

        Note:
          the default handlers to handle messages and related callback \
          function are shown below:
            ============================ ==================================
            Message type                 Callback function
            ============================ ==================================
            ``join_in_info``             ``callback_funcs_for_join_in()`` #쓸 일 없음
            ``join_in``                  ``callback_funcs_for_join_in()`` ################# 중요 #################
            ``model_para``               ``callback_funcs_model_para()``  ################# 중요 #################
            ``metrics``                  ``callback_funcs_for_metrics``   ################# 중요 #################
            ``grouping``                 ``callback_funcs_for_grouping``  ################# 중요 #################
            ============================ ==================================
        """
        self.register_handlers('join_in', self.callback_funcs_for_join_in, [
            'assign_client_id', 'ask_for_join_in_info', 'address', 'model_para'
        ])#client: 'join_in' 처리 -> Server: 'assign_client_id', 'ask_for_join_in_info', 'address', 'model_para'로 처리

        self.register_handlers('join_in_info', self.callback_funcs_for_join_in,
                               ['address', 'model_para'])
        
        self.register_handlers('model_para', self.callback_funcs_model_para,
                               ['model_para', 'evaluate', 'finish'])
        
        self.register_handlers('metrics', self.callback_funcs_for_metrics,
                               ['converged'])

    @abc.abstractmethod
    def run(self):
        """
        To start the FL course, listen and handle messages (for distributed \
        mode).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_model_para(self, message):
        """
        The handling function for receiving model parameters, which triggers \
        ``check_and_move_on`` (perform aggregation when enough feedback has \
        been received). This handling function is widely used in various FL \
        courses.

        Arguments:
            message: The received message.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_join_in(self, message):
        """
        The handling function for receiving the join in information. The \
        server might request for some information (such as \
        ``num_of_samples``) if necessary, assign IDs for the servers. \
        If all the clients have joined in, the training process will be \
        triggered.

        Arguments:
            message: The received message
        """
        raise NotImplementedError

    @abc.abstractmethod
    def callback_funcs_for_metrics(self, message):
        """
        The handling function for receiving the evaluation results, \
        which triggers ``check_and_move_on`` (perform aggregation when \
        enough feedback has been received).

        Arguments:
            message: The received message
        """
        raise NotImplementedError
