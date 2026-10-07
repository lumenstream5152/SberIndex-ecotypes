# Словарь данных outputs/main

Каждый артефакт — что это, ключи, расшифровка колонок. Индекс «артефакт → команда → run» — в README.md этого же каталога.

## Общие термины

- **МО** — муниципальное образование; `territory_id` — целочисленный id из справочника СберИндекса.
- **CLR** — центрированное лог-отношение долей 6 категорий расходов (Aitchison): `clr_<s> = log(share_s) − mean_j log(share_j)`. Категории: `prod` продовольствие, `health` здоровье, `market` маркетплейсы, `food` общепит, `transp` транспорт, `proch` прочее (= 1 − Σ₅ по построению данных, не мусор).
- **Два пространства меток** (важно, не путать): **макро-слой** — статическая типология k=3 (Leiden-консенсус, метки 0/1/2, артефакты `labels.parquet`, `passports_macro.parquet`); **помесячный слой** — реестр типов динамики (9 типов, id 0–8, артефакты `type_registry.parquet`, `events_*.parquet`, `passports_subtypes.parquet`). Карта соответствия: `type_id_map.parquet`.
- `dclr_*` — изменение CLR за месяц; `seed_agreement` — доля из 25 seed-прогонов Leiden, где узел попал в тот же кластер консенсуса; `displacement` — CLR-дистанция профиля месяц-к-месяцу.
- ICVI: SW ↑ (silhouette), CH_over_N ↑ (Calinski–Harabasz/n), S_Dbw ↓, AVI ↑ (average isolability, сеть), AVU ↓ (average unifiability, сеть), MQ ↑ (modularity quality, вариант Mancoridis 1998).

## labels.parquet (2016 строк)
Метки всех методов зоопарка на прод-графе M2. `row_idx` — позиция в node_index графа; далее по колонке на метод (суффикс `_k6` = K=6 из k-протокола). `leiden_consensus` — прод-типология.

## plateau_table.parquet (13 строк)
Выбор γ для Leiden по плато-правилу: `gamma`, `k_med/k_lo/k_hi` — число кластеров (медиана/квантили по 25 seeds), `ari_med/ari_min` — попарная seed-стабильность, `q_med` — модульность, `singleton_share/max_share` — доли вырождений, `icvi_*` — панель ICVI на точке, `icvi_comp` — сводка.

## table_B_measures.parquet (11 строк)
Бенчмарк мер сходства M1–M11: ноги Q (качество ICVI), S (стабильность к бутстрэпу рёбер), T (темпоральная согласованность odd/even), R (восстановление истины на синтетике), H (ценность для типологии/интерпретируемость структуры); `z_*` — z-нормировки по кандидатам, `composite` — 0.30Q+0.25S+0.20T+0.15R+0.10H (prereg), `ci_lo/ci_hi` — парный бутстрэп, `verdict_gate` — гейт прошла/нет.

## table_C_ranks.parquet
Ранги мер по каждой ноге + Borda + композитный ранг (два правила ранжирования рядом — расхождение M9/M3 по Borda задокументировано в отчёте).

## table_methods.parquet
Зоопарк методов × ICVI (все 6: SW, CH_over_N, S_Dbw, AVI, AVU, MQ-mancoridis) × стабильность (`boot_ari_mean`, `seed_ari_iqr`, `n_boot_ok`) × `nmi_synth` (медиана по сетке A синтетики, `nmi_synth_cells` — покрытие ячеек) × тайминг. `composite_with_zero_interp` / `composite_renormalized` — две версии композита (без ноги интерпретируемости / с перенормировкой); `circular_note` — флаги циркулярных пар ICVI↔метод (Leiden↔MQ, kmeans↔SW/CH).

## synth_grid_summary.parquet
Синтетическая сетка (PP-Dir + LFR): `cell` — ячейка (сетка:K:дрейф:α:μ), `method`, `nmi_med/nmi_mean`, `ari_med`, `per_snapshot_nmi_mean`, `f1_mean` (детекция смены типа; ≡0 у всех — структура статических методов, см. отчёт), `replicas_ok`.

## icvi_null.parquet
Перестановочный нуль ICVI прод-типологии (200 перестановок меток, кардинальности сохранены): `obs`, `null_mean`, `null_std`, `z`, `percentile`, `higher_better`. У AVU нуль почти без дисперсии (n≈2000) — z читать вместе с null_std.

## stability.json
Динамика: `pairs` — по каждой паре соседних месяцев `ari_cross` (ARI типологий) vs `ari_perturb_med` (медиана нуля 5%-пертурбации рёбер), `flagged` (true = отличия внутри шумового конверта); `summary` — агрегаты (23/23 flagged — публикуется как доказательство устойчивости, PREREG_DEVIATIONS №12).

## transitions_quarterly.csv
Квартальные матрицы переходов помесячного слоя (smoothed-метки): `quarter_from/to`, `type_from/to`, `n_movers`, `n_base`.

## type_registry.parquet (9 типов)
Реестр помесячного слоя: `birth_month/death_month`, `parent_id` (split), `merged_into` (merge), `n_nodes_by_month` (JSON), `lifetime` (месяцев).

## type_id_map.parquet
Карта макро-слой ↔ помесячный слой: для каждого типа реестра — распределение макро-типов его узлов (доля), и обратно. Ключ: `(layer, type_id)`.

## events_all.parquet / events_admitted.parquet (177)
Все smoothed-переходы и прошедшие узловой скрин (№12): `month`, `territory_id`, `type_from/to` (id реестра!), `top3_delta_clr` (JSON: топ-3 сдвига категорий), `displacement`, `admitted`, `reject_reason`. events_all — полная таблица со скрином (нужна для sensitivity к порогам).

## transition_cards.parquet (177)
Карточки переходов: поимённые МО, `driver1..5` — топ-5 признаков по |SHAP| со значениями в сырых единицах (`value`, `share_value`) и вкладом (`shap`); `card_text` — человекочитаемая строка; `p`/`p_raw` — вероятность модели (калиброванная/сырая), `prob_source`.

## event_study.parquet
Event-study ±6 мес вокруг admitted-переходов против matched-контроля (k=5 NN, тот же type_from): `scope` (all / канал A→B), `feature`, `rel_month` (−6..+6), `mean_movers/mean_controls`, `cohens_d`, `n_events`. Формулировки — только «ассоциировано».

## model_metrics.json
Драйверы переходов: rolling-origin+эмбарго; `h1`/`h3` — фолды и pooled для `lgbm`, `margin_rank`, `logreg5` (PR-AUC + baseline=prevalence, ROC-AUC, lift@decile, Brier); `calibration` (isotonic), `shap_top15` (и `_full_model`), `verdict.publish` — что публикуем; `published_features` — финальный список признаков (без seed_agreement — утечка убрана, JR1 подтвердил); `h1_no_seed` — абляция без seed_agreement.

## radar_watchlist.parquet
Радар смены типа на 2024-12: `p_move_h1` (калиброванная), `p_move_h1_raw`, `p_move_h3`, `decile`, `watch` (топ-дециль), **`unverified=true` всегда** (2025 в данных нет — out-of-time верификация невозможна), `risk_low_seed_agreement` (нижний дециль — риск артефакта), `top3_drivers` (JSON).

## passports_macro.parquet (3) / passports_subtypes.parquet (6)
Паспорта типов: `name_draft` (утверждает владелец; «тип K*» = протокол именования честно отказал), `core_features` (JSON: Миркин-профиль + правило дерева + fidelity), `geography`, `population`, `dynamics`, `monotowns`, `exemplars` (3 ближайших к центроиду), `boundary_mos` (пограничные), `practical_note`, `agreement_score` (согласие 3 описателей; <0.5 = спорный, помечено в note), `stability_note` (у подтипов — пониженная seed-стабильность).

## agreement.parquet (9)
Согласие описателей по типам: 0.5·Spearman(ранги признаков) + 0.5·Jaccard@5 (топ-5 списки), `verdict` текстом.

## validation_engel.parquet (3)
Энгель-чек: `wage_rub` (медиана зарплаты типа), `share_prod` (средняя доля продовольствия). Spearman=−1.00 — качественный чек: n=3, точный p=1/3 (см. оговорку в отчёте и в metrics).

## validation_four_russias.parquet
Диалог с «Четырьмя Россиями» Зубаревич: по типам × России — `n`, `share_in_type`, `lift` (доля в типе / доля в корпусе).

## validation_monotowns.parquet
Моногорода (перечень 1398-р): по типам × категориям — `n`, `share_in_type`, `lift`. lift≈1.14 = честный негатив (моногорода не отделяются).

## validation_aznakay.parquet (1)
Кейс Азнакаевский район: зарплата vs медиана типа и корпуса, профильные доли. Вывод: «зарплата ≠ безналичное потребление» (+31% к медиане своего типа при периферийном профиле).

## convergence_summary.json
Клубы сходимости Phillips–Sul (свой порт log-t, HAC QS-Эндрюс): по рядам mpfood (маркетплейсы+продовольствие), food, proch — клубы, критерии (3/3), σ/β-конвергенция. Оговорка о мощности T=24 — в отчёте.

## clubs_x_types.csv
Кросс-таб клубы × макро-типы (ортогональность структур).

## lead_summary.json
Лаг-лидерство: 4437 направленных рёбер (phase-randomization суррогаты + BH-FDR), `top_beacons`/`top_followers`, `geo` (доля рёбер по шоссе), контрольный срез. Подача: «глубже всех в поле», не «уникально» (у proknulo есть зачаточный lead-lag по типам).
