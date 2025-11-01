import os, json, math
from copy import deepcopy
from itertools import chain

def _load_clusters(cfg):
    c = getattr(cfg.llm.adapter, 'clusters', None)
    if c:
        return c
    path = getattr(cfg.llm.adapter, 'clusters_file', '')
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)['clusters']
    raise ValueError("llm.adapter.clusters 또는 clusters_file이 필요합니다.")

def _norm_to_zero_based(clusters, client_num):
    flat = list(chain.from_iterable(clusters))
    if not flat:
        return clusters, False
    mn, mx = min(flat), max(flat)
    # 1..N 이면 0..N-1로 변환
    if 1 <= mn and mx <= client_num:
        return [[cid - 1 for cid in g] for g in clusters], True
    # 이미 0-based 라고 가정
    return clusters, False

def compute_round_schedule(cfg):
    """
    규칙:
      - 총 논리 라운드 T = round_budget (기본 200)
      - 매 라운드 목표 선택 인원 K = target_per_round (기본 5)
      - w[i] = max(K, c[i]) ; L[i] = ceil(T * w[i] / sum w)
      - f[i] = 1 (c[i]>=K) else ceil(K / c[i])
      - P[i] = L[i] * f[i] (물리 라운드 수)
      - s[i] = c[i] (c[i]<=K) else K (라운드별 샘플 수)
      - boundaries = 누적합(P)
    산출물을 cfg에 주입하고, outdir/schedules/cluster_schedule.json 으로 저장.
    """
    if getattr(cfg.federate, 'sampler', 'uniform') != 'cluster':
        return cfg  # 분기: cluster 가 아닐 땐 아무 것도 하지 않음

    new_cfg = deepcopy(cfg)
    clusters_raw = _load_clusters(new_cfg)
    client_num = int(new_cfg.federate.client_num)

    # 내부 계산은 0-based로 통일
    clusters_idx0, converted = _norm_to_zero_based(clusters_raw, client_num)
    card = [len(g) for g in clusters_idx0]
    if len(card) != int(new_cfg.llm.adapter.count):
        raise ValueError(f"클러스터 수({len(card)}) != adapter.count({new_cfg.llm.adapter.count})")

    K = int(getattr(new_cfg.llm.adapter, 'target_per_round', 5))
    T = int(getattr(new_cfg.llm.adapter, 'round_budget', 200))

    w = [max(K, ci) for ci in card]
    W = sum(w)
    L = [math.ceil(T * wi / W) for wi in w]
    f = [1 if ci >= K else math.ceil(K / max(1, ci)) for ci in card]
    P = [Li * fi for Li, fi in zip(L, f)]
    s = [ci if ci <= K else K for ci in card]

    boundaries, acc = [], 0
    for pi in P:
        acc += int(pi)
        boundaries.append(acc)
    total_rounds = int(sum(P))

    # 1-based 표현(저장/개인화용)
    clusters_idx1 = [[cid + 1 for cid in g] for g in clusters_idx0]
    client2adapter_1b = {str(cid): a for a, g in enumerate(clusters_idx1) for cid in g}

    # 저장 경로
    outdir = getattr(new_cfg, 'outdir', 'exp')
    sched_dir = os.path.join(outdir, 'schedules')
    os.makedirs(sched_dir, exist_ok=True)
    sched_path = os.path.join(sched_dir, 'cluster_schedule.json')

    payload = {
        "adapter_count": int(new_cfg.llm.adapter.count),
        "client_num": client_num,
        "round_budget": T,
        "target_per_round": K,
        "clusters_1_based": clusters_idx1,
        "clusters_0_based": clusters_idx0,
        "cardinalities": card,
        "weights_w": w,
        "logical_rounds_L": L,
        "small_cluster_factor_f": f,
        "physical_rounds_P": P,
        "sample_num_per_adapter_s": s,
        "boundaries": boundaries,
        "total_physical_rounds": total_rounds,
        "client2adapter_1_based": client2adapter_1b,
        "note": "IDs 내부계산은 0-based, 저장/개인화는 1-based."
    }
    with open(sched_path, 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    # CFG 주입(런타임 접근용)
    new_cfg.llm.adapter.boundaries = boundaries
    new_cfg.llm.adapter.rounds_physical = P
    new_cfg.llm.adapter.sample_num_per_adapter = s
    new_cfg.llm.adapter.clusters = clusters_idx0  # 내부 0-based
    new_cfg.llm.adapter.cluster_runtime = {
        "schedule_file": sched_path,
        "client2adapter": client2adapter_1b,  # 1-based
    }
    new_cfg.federate.total_round_num = total_rounds
    return new_cfg
