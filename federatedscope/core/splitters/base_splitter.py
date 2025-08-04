import abc
import inspect


class BaseSplitter(abc.ABC):
    """
    This is an abstract base class for all splitter, which is not \
    implemented with ``__call__()``.

    Attributes:
        client_num: Divide the dataset into ``client_num`` pieces.
    """
    def __init__(self, client_num):
        self.client_num = client_num

    @abc.abstractmethod
    def __call__(self, dataset, *args, **kwargs): #__call__ 메서드를 구현해 놓으면, “객체 이름 뒤에 (...)” 를 붙였을 때 파이썬이 알아서 그 __call__ 을 실행
        raise NotImplementedError

    def __repr__(self): #파이썬에서 객체를 출력하거나 디버깅 목적으로 문자열로 변환할 때 호출되는 특별한 메서드.
        """

        Returns: Meta information for `Splitter`.

        """
        sign = inspect.signature(self.__init__).parameters.values()
        meta_info = tuple([(val.name, getattr(self, val.name))
                           for val in sign])
        return f'{self.__class__.__name__}{meta_info}'
