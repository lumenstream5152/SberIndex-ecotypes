# outputs/main — каноничные артефакты (коммитим; CC BY-SA 4.0, см. NOTICE)

Каждая строка воспроизводится командой из корня репо. Полные прогоны (со снапшотами
конфига и run.log) — в `outputs/<run_id>/` (не коммитятся).

| Артефакт | Что | Команда | Исходный run |
|---|---|---|---|
| labels.parquet | метки всех методов зоопарка на прод-графе (M2) | `make cluster` | 20261006_183609 |
| plateau_table.parquet | плато-таблица выбора γ (13 точек сетки) | `make cluster` | 20261006_183609 |
| metrics_04_cluster.json | γ*=0.293 (fallback), k=3, seed ARI med 0.931, NMI-матрица | `make cluster` | 20261006_183609 |
| table_B_measures.parquet | бенчмарк мер: 11 мер × ноги Q/S/T/R/H + композит + CI | `make icvi` | 20261006_150618 |
| table_C_ranks.parquet | ранги мер по ногам + Borda | `make icvi` | 20261006_150618 |
| table_methods.parquet | зоопарк × ICVI × стабильность × NMI_synth × композит | `make icvi` | 20261006_150618 |
| synth_grid_summary.parquet | сетка синтетики: 13 ячеек × методы, NMI/ARI/F1 | `scripts/06b_synth_grid.py` | 20261006_1117 |
| stability.json | динамика: ARI(t,t+1) vs нуль 5%-пертурбации, flagged 23/23 (публикуется как есть) | `make dynamics` | data/processed/dynamics |
| transitions_quarterly.csv | квартальные матрицы переходов (публичные) | `make dynamics` | data/processed/dynamics |
| type_registry.parquet | реестр типов (рождение/смерть/размеры по месяцам) | `make dynamics` | data/processed/dynamics |
| events_admitted.parquet | 177 переходов, прошедших узловой скрин (PREREG_DEVIATIONS №12) | `make dynamics` | data/processed/dynamics |
| passports_macro.parquet | паспорта макро-типов (k=3), 10 полей | `make interpret` | 20261006_200607 |
| passports_subtypes.parquet | паспорта подтипов (помесячный слой, пометка стабильности) | `make interpret` | 20261006_200607 |
| agreement.parquet | согласие 3 описателей (Миркин/дерево/SHAP) по типам | `make interpret` | 20261006_200607 |
| validation_*.parquet | внешняя валидация: моногорода / «Четыре России» / Энгель / Азнакаево | `make interpret` | 20261006_200607 |
| model_metrics.json | драйверы переходов: PR-AUC, базлайны, калибровка | `scripts/07b_drivers.py` | 20261006_201057 |
| transition_cards.parquet | карточки 177 переходов (драйверы в сырых единицах) | `scripts/07b_drivers.py` | 20261006_201057 |
| event_study.parquet | event-study ±6 мес вокруг admitted-переходов | `scripts/07b_drivers.py` | 20261006_201057 |
| radar_watchlist.parquet | радар смены типа (202 МО, unverified=true) | `scripts/07b_drivers.py` | 20261006_201057 |
| convergence_summary.json | клубы сходимости: log-t/σ/β по трём рядам | `scripts/07c_convergence.py` | 20261006_201742 |
| clubs_x_types.csv | кросс-таб клубы × типы | `scripts/07c_convergence.py` | 20261006_201742 |
| lead_summary.json | лаг-лидерство: 4437 направленных рёбер, маяки/последователи | `scripts/07d_laglead.py` | 20261006_201752 |
