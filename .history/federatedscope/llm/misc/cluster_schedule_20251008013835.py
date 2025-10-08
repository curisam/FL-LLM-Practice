# federatedscope/llm/misc/cluster_schedule.py
import os, json, math
from copy import deepcopy
from itertools import chain

def _load_clusters(cfg):
    c = getattr(cfg.llm.adapter, 'clusters', None)
    if c: return c
    path = getattr(cfg.llm.adapter, 'clusters_file', '')
    if path and os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)['clusters']
    raise ValueError("llm.adapter.clusters 또는 clusters_file 필요")

def _norm_to_one_based(clusters):
    # 입력이 0-based/1-based 섞여 있어도 1-based로 정규화
    flat = list(chain.from_iterable(clusters))
    if not flat: return clusters
    mn = min(flat)
    return [[cid+1 for cid in g] for g in clusters] if mn == 0 else clusters

def compute_round_schedule(cfg):
    # cluster 분기 아닐 때는 건드리지 않음
    if getattr(cfg.federate, 'sampler', 'uniform') != 'cluster' or int(cfg.llm.adapter.count) <= 1:
        return cfg

    new_cfg = deepcopy(cfg)
    clusters_1b = _norm_to_one_based(_load_clusters(new_cfg))  # 내부는 1-based 유지(UniformSampler와 호환)
    card = [len(g) for g in clusters_1b]
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

    # 저장
    outdir = getattr(new_cfg, 'outdir', 'exp')
    sched_dir = os.path.join(outdir, 'schedules'); os.makedirs(sched_dir, exist_ok=True)
    sched_path = os.path.join(sched_dir, 'cluster_schedule.json')
    client2adapter = {str(cid): a for a, g in enumerate(clusters_1b) for cid in g}
    payload = {
        "adapter_count": int(new_cfg.llm.adapter.count),
        "client_num": int(new_cfg.federate.client_num),
        "round_budget": T, "target_per_round": K,
        "clusters_1_based": clusters_1b,
        "cardinalities": card, "weights_w": w,
        "logical_rounds_L": L, "small_cluster_factor_f": f,
        "physical_rounds_P": P, "sample_num_per_adapter_s": s,
        "boundaries": boundaries, "total_physical_rounds": total_rounds,
        "client2adapter_1_based": client2adapter
    }
    with open(sched_path, 'w') as f: json.dump(payload, f, indent=2, ensure_ascii=False)

    # CFG 주입(런타임 접근용) — 1-based 유지
    new_cfg.llm.adapter.boundaries = boundaries
    new_cfg.llm.adapter.rounds_physical = P
    new_cfg.llm.adapter.sample_num_per_adapter = s
    new_cfg.llm.adapter.clusters = clusters_1b
    new_cfg.llm.adapter.cluster_runtime = {"schedule_file": sched_path, "client2adapter": client2adapter}
    new_cfg.federate.total_round_num = total_rounds
    return new_cfg
