# Multidimensional_root_cause_analysis
- 基于蒙特卡洛树（MCTS）的多维监控异常根因分析
- 本项目基于[HotSpot: Anomaly Localization for Additive KPIs With Multi-Dimensional Attributes](https://www.researchgate.net/publication/323087892_HotSpot_Anomaly_Localization_for_Additive_KPIs_with_Multi-Dimensional_Attributes)
- 原作者是我本科的同学，在他的帮助指导下，我实现了二维的根因分析，具体蒙特卡洛树搜索的细节与论文中有少许出入。
- 例如当我们监测播放质量这个指标，它会和多个维度关联，包括app版本、运营商、省份、cdn等等，当播放质量发生突降时，我们需要快速定位到生成这个问题的维度组合。
