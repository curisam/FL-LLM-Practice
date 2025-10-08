# federatedscope/llm/misc/cluster_schedule.py
import os, json, math
from copy import deepcopy

def _load_clusters(cfg):
    # JSON에 주석 금지! 순수 JSON만 허용
    with open(cfg.llm.adapter.clusters_file, 'r') as f:
        obj = json.load(f)
    return obj['clusters']  # 1-based 클라이언트 ID 목록의 리스트

def compute_round_schedule(cfg):
    new_cfg = deepcopy(cfg)

    clusters_1b = _load_clusters(new_cfg)               # [[1,6,...], [2,4,...], ...]
    N = int(new_cfg.federate.client_num)                # 전체 클라 수 (예: 53)
    B = int(new_cfg.llm.adapter.round_budget)           # 200
    T = int(new_cfg.llm.adapter.target_per_round)       # 5

    sizes = [len(g) for g in clusters_1b]               # n_i
    s_per = [min(n, T) for n in sizes]                  # s_i = min(n_i, 5)

    # 균등 기대치 유지: 각 클라의 기대 참여 횟수를 E*로 맞춤
    E_star = (B * T) / float(N)

    # 각 어댑터(=클러스터)의 라운드 수
    rounds = [int(math.ceil(E_star * n / s)) for n, s in zip(sizes, s_per)]

    # boundaries: bisect_right용 "포함 끝 인덱스(0-based)"
    boundaries, acc = [], 0
    for r in rounds:
        acc += r
        boundaries.append(acc - 1)

    # cfg에 반영 (총 라운드는 sum(rounds)로 덮어씀 — 200을 초과 가능)
    if hasattr(new_cfg, 'defrost'): new_cfg.defrost()
    new_cfg.llm.adapter.clusters = clusters_1b
    new_cfg.llm.adapter.sample_num_per_adapter = s_per
    new_cfg.llm.adapter.boundaries = boundaries
    new_cfg.llm.adapter.per_client_target = E_star
    new_cfg.federate.total_round_num = int(sum(rounds))

    # 스케줄 JSON 저장 (예: exp/.../cluster_schedule/cluster_schedule_u3.json)
    sched_dir = os.path.join(new_cfg.outdir, "cluster_schedule")
    os.makedirs(sched_dir, exist_ok=True)
    sched_path = os.path.join(
        sched_dir, f'cluster_schedule_u{int(new_cfg.llm.adapter.count)}.json'
    )
    with open(sched_path, 'w') as f:
        json.dump({
            "clusters_1_based": clusters_1b,
            "sizes": sizes,
            "sample_num_per_adapter": s_per,
            "rounds_per_adapter": rounds,
            "boundaries": boundaries,
            "per_client_target": E_star,
            "meta": {
                "round_budget": B,
                "target_per_round": T,
                "sum_rounds": int(sum(rounds))
            }
        }, f, indent=2)
    new_cfg.llm.adapter.cluster_runtime = {"schedule_file": sched_path}
    if hasattr(new_cfg, 'freeze'): new_cfg.freeze()
    return new_cfg
