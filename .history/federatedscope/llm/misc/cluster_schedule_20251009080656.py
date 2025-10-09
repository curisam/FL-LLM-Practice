import json, os, math
import numpy as np
from copy import deepcopy

def compute_round_schedule(cfg):
    new = deepcopy(cfg)

    # 1) clusters/file 읽기 (1-based client IDs)
    with open(new.llm.adapter.clusters_file, 'r') as f:
        clusters_1b = json.load(f)['clusters']

    N = int(new.federate.client_num)            # 53
    B = int(new.llm.adapter.round_budget)       # 200 (또는 1)
    T = int(new.llm.adapter.target_per_round)   # 5

    sizes = [len(g) for g in clusters_1b]       # n_i
    s_per = [min(n, T) for n in sizes]          # s_i
    E_star = (B * T) / float(N)                 # 균등 기대치

    # 2) 각 어댑터 라운드 수 Ri = ceil(E* * n_i / s_i)
    r_per = [int(math.ceil(E_star * n / s)) for n, s in zip(sizes, s_per)]
    if B == 1:                                  # B=1 테스트 케이스(옵션)
        r_per = [1]*len(sizes)

    # 3) ★ Exclusive ends (끝점 누적)
    #    adapter a 는 라운드 [start[a], end[a]-1]
    round_ends = np.cumsum(r_per).astype(int).tolist()   # [19,65,92,...]
    sum_rounds = int(round_ends[-1])                     # total_round_num

    # 4) client → adapter (0-based) 매핑
    c2a = {}
    for aidx, group in enumerate(clusters_1b):   # aidx: 0..C-1
        for cid in group:                         # cid: 1..N
            c2a[str(int(cid))] = int(aidx)

    # 5) cfg에 주입
    try: new.defrost()
    except: pass
    new.llm.adapter.clusters = clusters_1b
    new.llm.adapter.sample_num_per_adapter = s_per
    new.llm.adapter.round_ends = round_ends      # ★ boundaries 대신 round_ends
    new.federate.total_round_num = sum_rounds
    new.llm.adapter.per_client_target = E_star

    # 6) JSON 저장
    sched_dir = os.path.join(new.outdir, "cluster_schedule")
    os.makedirs(sched_dir, exist_ok=True)
    sched_path = os.path.join(sched_dir, f"cluster_schedule_u{int(new.llm.adapter.count)}.json")
    with open(sched_path, 'w') as f:
        json.dump({
            "clusters_1_based": clusters_1b,
            "sizes": sizes,
            "sample_num_per_adapter": s_per,
            "rounds_per_adapter": r_per,
            "round_ends": round_ends,                    # ★ 저장
            "per_client_target": E_star,
            "meta": {"round_budget": B, "target_per_round": T, "sum_rounds": sum_rounds},
            "client2adapter_1_based": c2a               # ★ 저장
        }, f, indent=2)
    new.llm.adapter.cluster_runtime = {"schedule_file": sched_path}
    try: new.freeze()
    except: pass
    return new
