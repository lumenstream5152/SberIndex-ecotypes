# outputs/main — каноничные артефакты (коммитим; CC BY-SA 4.0, см. NOTICE)

Каждая строка воспроизводится командой из корня репо. Полные прогоны (со снапшотами
конфига и run.log) — в `outputs/<run_id>/` (не коммитятся, см. .gitignore).
Словарь колонок — `DATA_DICTIONARY.md`. Два пространства меток (макро k=3 vs
помесячный реестр 0–8) разведены в `type_id_map.parquet` — не путать.

Срез: 07.10.2026, после аудита JR1 (19 независимых проверок) и перезапусков
05/07/07b/06(skip-measures) с багфиксами (PREREG_DEVIATIONS №13).

| Артефакт | Что | Команда | Исходный run |
|---|---|---|---|
| labels.parquet | метки всех методов зоопарка на прод-графе (M2) | `make cluster` | 20261006_183609 |
| plateau_table.parquet | плато-таблица выбора γ (13 точек сетки) | `make cluster` | 20261006_183609 |
| metrics_04_cluster.json | γ*=0.293 (fallback), k=3, seed ARI med 0.931, NMI-матрица | `make cluster` | 20261006_183609 |
| table_A_gate.parquet | гейт мер M1–M11 + анти-примеры X1–X3 (с вердиктами; M11 fail) | `make measures` | data/processed/measures |
| table_topology.parquet | топология графов мер (степени, компоненты, хаб-коллапс) | `make measures` | data/processed/measures |
| table_B_measures.parquet | бенчмарк мер: 11 мер × ноги Q/S/T/R/H + композит + CI | `make icvi` | 20261006_150618 |
| table_C_ranks.parquet | ранги мер по ногам + Borda (Borda-победитель M9 ≠ композит M3 — расхождение раскрыто в отчёте) | `make icvi` | 20261006_150618 |
| table_methods.parquet | зоопарк × все 6 ICVI (вкл. AVI/AVU) × стабильность × NMI_synth × обе версии композита + circular_note | `make icvi` | 20261007_143127 |
| synth_grid_summary.parquet | сетка синтетики: 13 ячеек × методы, NMI/ARI/F1 | `make synth` | outputs/synth_grid (отдельный каталог прогона) |
| icvi_null.parquet | ICVI прод-типологии против 200 перестановок меток (z, percentile, null_mean/std) | `make icvi-null` | 20261007_133353 |
| stability.json | динамика: ARI(t,t+1) vs нуль 5%-пертурбации, flagged 23/23 (публикуется как есть) | `make dynamics` | data/processed/dynamics |
| transitions_quarterly.csv | квартальные матрицы переходов (публичные, smoothed) | `make dynamics` | data/processed/dynamics |
| type_registry.parquet | реестр типов помесячного слоя (рождение/смерть/размеры) | `make dynamics` | data/processed/dynamics |
| events_all.parquet | все 1363 smoothed-перехода со скрином (admitted/reject_reason) — нужен для sensitivity | `make dynamics` | data/processed/dynamics |
| events_admitted.parquet | 177 переходов, прошедших узловой скрин (PREREG_DEVIATIONS №12) | `make dynamics` | data/processed/dynamics |
| events_sensitivity.csv | число admitted при пороге displacement q50…q95 (q75→177) | `make aux` | 20261007_144949 |
| type_id_map.parquet | карта макро-слой k=3 × помесячный реестр 0–8 (доли в обе стороны) | `make aux` | 20261007_144949 |
| passports_macro.parquet | паспорта макро-типов (k=3), согласие описателей + пометки спорности | `make interpret` | 20261007_142436 |
| passports_subtypes.parquet | паспорта подтипов (помесячный слой, пометка пониженной стабильности) | `make interpret` | 20261007_142436 |
| agreement.parquet | согласие 3 описателей (Миркин/дерево/SHAP) по типам | `make interpret` | 20261007_142436 |
| agreement_pairs_*.parquet | попарное согласие описателей (подложка agreement) | `make interpret` | 20261007_142436 |
| validation_*.parquet | внешняя валидация: моногорода / «Четыре России» / Энгель / Азнакаево / курорты | `make interpret` | 20261007_142436 |
| model_metrics.json | драйверы переходов: PR-AUC, базлайны, калибровка, вердикт публикации | `make drivers` | 20261007_142730 |
| transition_cards.parquet | карточки 177 переходов (поимённые, драйверы в сырых единицах) | `make drivers` | 20261007_142730 |
| event_study.parquet | event-study ±6 мес вокруг admitted-переходов (matched-контроль) | `make drivers` | 20261007_142730 |
| radar_watchlist.parquet | радар смены типа (топ-дециль, unverified=true) | `make drivers` | 20261007_142730 |
| channel_profiles.parquet | SHAP-профили каналов переходов A→B | `make drivers` | 20261007_142730 |
| shap_profiles.parquet | глобальный SHAP-профиль драйверов | `make drivers` | 20261007_142730 |
| convergence_summary.json | клубы сходимости: log-t/σ/β по трём рядам | `make convergence` | 20261006_201742 |
| clubs_x_types.csv | кросс-таб клубы × типы | `make convergence` | 20261006_201742 |
| lead_summary.json | лаг-лидерство: 4437 направленных рёбер (FDR), маяки/последователи | `make laglead` | 20261006_201752 |
| graph_summary.json | константы построения сети: E*=12 900 (12 759 FDR-пересечение + 141 safety), 0 изолятов, 1 компонента, медианы корреляций 0.906→−0.006 | `make aux` | 20261006_183551 |
