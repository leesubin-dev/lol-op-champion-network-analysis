# LoL OP Champion Network Analysis

League of Legends pro match data를 크롤링하고, OP 챔피언 조합과 챔피언 시너지/카운터 네트워크를 분석한 비정형 데이터 분석 프로젝트입니다.

> 핵심 질문: OP 챔피언은 정말 승리를 보장하는가?  
> 결론: OP 챔피언은 중요한 메타 신호지만, 프로 경기 승패는 OP 단독보다 조합 구조와 상대 조합 맥락을 함께 봐야 더 잘 설명됩니다.

## Project Overview

이 프로젝트는 2026년 26.09~26.11 패치 프로 경기 데이터를 대상으로 다음을 분석합니다.

- Leaguepedia 기반 프로 경기 픽/밴/승패 데이터 수집
- OP.GG 패치별 OP 티어를 이용한 OP 챔피언 정의
- OP 포함/제외 조합 성과 비교
- OP 챔피언을 이긴 상대 조합 분석
- 챔피언 시너지 네트워크 구축
- 챔피언 카운터 네트워크 구축
- 팀-챔피언 2-mode network 분석
- 네트워크 feature를 추가한 승패 예측 실험

## Key Results

| 항목 | 결과 |
|---|---:|
| Raw collected matches | 2,207 |
| Excluded incomplete-pick matches | 190 |
| Final analyzed matches | 2,017 |
| Patch range | 26.09, 26.10, 26.11 |
| Date range | 2026-04-30 18:09:00 ~ 2026-06-15 16:40:00 |
| Matches where exactly one team picked OP | 143 |
| OP team win rate | 53.1% |
| 95% confidence interval | 45.5% ~ 60.8% |
| p-value vs 50% | 0.504 |
| Synergy network | 131 nodes / 1,293 edges |
| Counter network | 148 nodes / 3,202 edges |
| Test set size | 404 matches |

Model comparison:

| Model | Feature set | Accuracy | F1-score | ROC-AUC |
|---|---|---:|---:|---:|
| Logistic Regression | baseline | 0.557 | 0.656 | 0.589 |
| Logistic Regression | network | 0.517 | 0.586 | 0.477 |
| Random Forest | baseline | 0.545 | 0.613 | 0.552 |
| Random Forest | network | 0.520 | 0.573 | 0.538 |

네트워크 feature를 추가했지만 승패 예측 성능 향상은 확인되지 않았습니다. 따라서 예측 모델은 주 결론이 아니라 네트워크 지표의 보조 검증으로 해석했습니다.

## Repository Structure

```text
lol-op-champion-network-analysis/
├── README.md
├── requirements.txt
├── src/
│   └── pipeline_from_notebook.py
├── code/
│   └── make_notebook.py
├── notebooks/
│   └── 비정형네트워크프로젝트.ipynb
├── data/
│   ├── README.md
│   ├── raw/
│   │   └── raw_match_history.csv
│   └── processed/
│       ├── analysis_match_history.csv
│       ├── champion_edges_counter.csv
│       ├── champion_edges_synergy.csv
│       ├── champion_global_stats.csv
│       └── opgg_champion_tiers.csv
├── outputs/
│   └── analysis result CSV files
├── figures/
│   └── analysis figures
├── reports/
│   └── report markdown files
└── docs/
    ├── GITHUB_UPLOAD_CHECKLIST.md
    ├── REPRODUCIBILITY.md
    ├── DATA_SOURCES.md
    └── manifest.json
```

## Main Figures

| Figure | Description |
|---|---|
| `figures/03_synergy_network.png` | Champion synergy network |
| `figures/04_counter_network.png` | Counter / matchup-advantage network |
| `figures/13_best_compositions_by_op_status.png` | Best compositions by OP status |
| `figures/14_op_loss_opponent_combos.png` | Opponent compositions that beat OP teams |
| `figures/16_op_status_winrate_summary.png` | OP team win-rate summary |
| `figures/19_team_champion_two_mode_network.png` | Team-champion 2-mode network |

## How To Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/비정형네트워크프로젝트.ipynb
```

또는 노트북 생성 스크립트를 사용할 수 있습니다.

```bash
python code/make_notebook.py
```

웹 크롤링은 사이트 구조 변경, 접근 제한, 네트워크 상태에 영향을 받을 수 있습니다. 그래서 재현과 검토를 위해 주요 CSV 결과를 `data/`, `outputs/`, `figures/`에 함께 포함했습니다.

재현 관련 주의사항은 `docs/REPRODUCIBILITY.md`에 따로 정리했습니다.

## Method Summary

1. **Data collection**
   - Leaguepedia에서 경기별 patch, team, winner, picks/bans 수집
   - OP.GG에서 패치별 OP tier champion-position 수집
   - GOL.GG에서 챔피언 통계 보조 자료 수집
   - Riot Data Dragon은 챔피언 아이콘 매핑에 활용

2. **Preprocessing**
   - 불완전 픽 경기 190개 제외
   - 완전한 5v5 픽 경기 2,017개 사용
   - 챔피언명 표준화
   - 패치 26.09~26.11 구간 분석

3. **Network construction**
   - Synergy network: 같은 팀에서 함께 픽된 챔피언 쌍
   - Counter network: 승리팀 챔피언 -> 패배팀 챔피언 방향 관계
   - 2-mode network: 팀과 챔피언을 서로 다른 node type으로 둔 이원 네트워크

4. **Modeling**
   - Baseline: champion win-rate/pick-rate/ban-rate 기반 feature
   - Network model: synergy/counter/centrality feature 추가
   - Chronological split으로 train/test 분리

## Interpretation Notes

- 본 프로젝트의 "counter network"는 순수 1:1 라인전 카운터가 아닙니다. 경기 결과 기반의 상대 조합 우위 네트워크로 해석해야 합니다.
- OP.GG OP tier는 솔로랭크 기반 외부 지표이므로 프로 메타와 완전히 일치하지 않을 수 있습니다.
- 일부 상위 조합은 표본 수가 작기 때문에 "확정 조합"이 아니라 "후보 조합"으로 해석했습니다.
- 관찰 데이터 기반 분석이므로 인과관계를 단정하지 않습니다.

## Data And Asset Notice

League of Legends, champion names, champion icons, and related assets belong to Riot Games. This repository is an academic/non-commercial analysis project. Raw web data and generated figures are included only for reproducibility and review. See `docs/DATA_SOURCES.md` and `NOTICE.md` for details.
