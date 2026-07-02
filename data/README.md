# Data

This folder contains the minimum CSV files needed to review or rerun the analysis.

## Included

- `raw/raw_match_history.csv`: collected match-history table before complete-pick filtering
- `processed/analysis_match_history.csv`: final 2,017 complete-pick matches used for analysis
- `processed/champion_edges_synergy.csv`: champion pair synergy edges
- `processed/champion_edges_counter.csv`: directed counter/matchup-advantage edges
- `processed/champion_global_stats.csv`: champion-level auxiliary statistics
- `processed/opgg_champion_tiers.csv`: OP.GG patch-tier reference table

## Excluded From GitHub Version

- `data/cache/`: local web request cache
- `data/champion_icons/`: downloaded Riot Data Dragon champion icon images
- very large intermediate files that are not required for portfolio review

The full local project may contain more intermediate files than this GitHub-ready package.
