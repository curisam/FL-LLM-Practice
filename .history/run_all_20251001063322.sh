#!/usr/bin/env bash
set -e  # 중간에 실패하면 바로 종료

bash pfl_local_only.sh
bash cluster_stat.sh
