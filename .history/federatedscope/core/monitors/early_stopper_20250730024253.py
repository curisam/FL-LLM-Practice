import operator
import numpy as np


# TODO: make this as a sub-module of monitor class
class EarlyStopper(object):
    """
    Track the history of metric (e.g., validation loss), \
    check whether should stop (training) process if the metric doesn't \
    improve after a given patience.

    Args:
        patience (int): (Default: 5) How long to wait after last time the \
            monitored metric improved. Note that the \
            ``actual_checking_round = patience * cfg.eval.freq``
        delta (float): (Default: 0) Minimum change in the monitored metric to \
            indicate an improvement.
        improve_indicator_mode (str): Early stop when no improve to \
            last ``patience`` round, in ``['mean', 'best']``
    """

    """
    patience: 지표가 개선되지 않아도 기다릴 최대 라운드 수.

    delta: “개선”으로 간주하기 위한 최소 변화량.

    improve_indicator_mode: 개선 판단 기준.

        'best': 과거 최고(best) 값 대비 비교

        'mean': 최근 patience 라운드 평균 대비 비교

    the_larger_the_better: 지표가 클수록 좋은 경우(True, 예: accuracy) / 작을수록 좋은 경우(False, 예: loss)

    내부 변수로는

        self.best_metric: 지금까지 기록된 최고(또는 최저) 값

        self.counter_no_improve: 연속으로 개선이 없었던 횟수

        self.early_stopped: 멈춰야 하는지 최종 플래그

    를 저장합니다.


    정리: 이 클래스를 활용하면, 훈련 중 자동으로 “과적합” 직전에 학습을 멈추도록 제어할 수 있어 모델의 일반화 성능을 높이는 데 도움을 줍니다.

        patience 만큼 개선이 없으면 멈춤

        delta 이상으로 좋아져야 “개선”으로 인정

        best vs mean: 과거 최고값 vs 최근 평균값 비교 방식

        the_larger_the_better: 지표가 클수록 좋은지/작을수록 좋은지    
    """

    def __init__(self,
                 patience=5,
                 delta=0,
                 improve_indicator_mode='best',
                 the_larger_the_better=True):
        assert 0 <= patience == int(
            patience
        ), "Please use a non-negtive integer to indicate the patience"
        assert delta >= 0, "Please use a positive value to indicate the change"
        assert improve_indicator_mode in [
            'mean', 'best'
        ], "Please make sure `improve_indicator_mode` is 'mean' or 'best']"

        self.patience = patience
        self.counter_no_improve = 0
        self.best_metric = None
        self.early_stopped = False
        self.the_larger_the_better = the_larger_the_better
        self.delta = delta
        self.improve_indicator_mode = improve_indicator_mode
        # For expansion usages of comparisons
        self.comparator = operator.lt
        self.improvement_operator = operator.add
        """
        self.comparator = operator.lt

            operator.lt(a, b) 는 파이썬의 a < b 와 동일합니다.

            EarlyStopper 에서는 “새로운 지표가 더 좋아졌는지” 혹은 “더 나빠졌는지(=멈출 조건인지)” 판단할 때 두 값을 비교해야 하는데, 이 비교식을 직접 하드코딩하지 않고 self.comparator 로 할당해 두면, 필요에 따라 operator.lt 대신 operator.gt 등을 바꿔 쓸 수도 있습니다.

        self.improvement_operator = operator.add

            operator.add(x, y) 는 x + y 와 동일합니다.

            지표 개선 기준을 best_metric ± delta 처럼 쓰려면 “best_metric 에 delta 를 더하거나 빼는” 연산이 필요한데, 이 역시 operator.add 로 일반화해 둔 겁니다.

            예를 들어 loss(낮아야 좋은 지표)를 모니터링할 때 “이전 최저(best_metric) 에서 -delta 만큼 더 작아져야 개선으로 본다” 는 로직에서 self.improvement_operator(best_metric, -delta) 를 쓰면 됩니다.
        """



    def __track_and_check_dummy(self, new_result):
        """
        Dummy stopper, always return false

        Args:
            new_result:

        Returns:
            False
        """

        """
        patience=0 이거나 모드가 잘못 설정된 경우, 늘 False 반환

        """


        self.early_stopped = False
        return self.early_stopped

    def __track_and_check_best(self, history_result):
        """
        Tracks the best result and checks whether the patience is exceeded.

        Args:
            history_result: results of all evaluation round

        Returns:
            Bool: whether stop
        """


        """
        매번 new_result 를 self.best_metric 과 비교

        만약 (더 좋아야 할 방향에 따라) new_result < best_metric + delta (또는 반대) 이면
        → 개선 없음, counter_no_improve += 1

        그렇지 않으면 개선 있음 → best_metric = new_result, counter_no_improve = 0

        마지막으로 counter_no_improve >= patience 면 early_stopped = True

        """

        new_result = history_result[-1]
        if self.best_metric is None:
            self.best_metric = new_result
        elif not self.the_larger_the_better and self.comparator(
                self.improvement_operator(self.best_metric, -self.delta),
                new_result):
            # add(best_metric, -delta) < new_result
            self.counter_no_improve += 1
        elif self.the_larger_the_better and self.comparator(
                new_result,
                self.improvement_operator(self.best_metric, self.delta)):
            # new_result < add(best_metric, delta)
            self.counter_no_improve += 1
        else:
            self.best_metric = new_result
            self.counter_no_improve = 0

        self.early_stopped = self.counter_no_improve >= self.patience
        return self.early_stopped

    def __track_and_check_mean(self, history_result):
        """
        최근 patience 라운드의 평균(np.mean(history_result[-patience-1:-1]))과 new_result 를 비교

        개선 기준(delta, the_larger_the_better)에 따라
        → 평균 대비 개선되지 않았다면 early_stopped = True

        """

        new_result = history_result[-1]
        if len(history_result) > self.patience:
            if not self.the_larger_the_better and self.comparator(
                    self.improvement_operator(
                        np.mean(history_result[-self.patience - 1:-1]),
                        -self.delta), new_result):
                self.early_stopped = True
            elif self.the_larger_the_better and self.comparator(
                    new_result,
                    self.improvement_operator(
                        np.mean(history_result[-self.patience - 1:-1]),
                        self.delta)):
                self.early_stopped = True
        else:
            self.early_stopped = False

        return self.early_stopped

    def track_and_check(self, new_result):
        """
        Checks the new result and if it improves it returns True.

        Args:
            new_result: new evaluation result

        Returns:
            Bool: whether stop
        """

        track_method = self.__track_and_check_dummy  # do nothing
        if self.patience == 0:
            track_method = self.__track_and_check_dummy
        elif self.improve_indicator_mode == 'best':
            track_method = self.__track_and_check_best
        elif self.improve_indicator_mode == 'mean':
            track_method = self.__track_and_check_mean

        return track_method(new_result)
