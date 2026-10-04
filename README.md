# sberindex-ecotypes

Типы локальных экономик России по потреблению СберИндекс 2023–24: динамическая
атрибутированная сеть 2016 муниципальных образований × 24 месяца, кластеры,
их эволюция и интерпретация. Полная воспроизводимость одной командой.

![ci](https://github.com/OWNER/sberindex-ecotypes/actions/workflows/ci.yml/badge.svg)

## Главный результат

Воспроизводится командами ниже; итоговые таблицы — `outputs/main/`
(метод × ICVI × NMI на синтетике; паспорта типов). Предрегистрация
экспериментальных решений: [PREREG.md](PREREG.md) (тег `prereg-v1`).

## Quickstart

```bash
uv sync                 # или: pip install -r requirements.txt
make data               # скачать данные (~20 МБ, проверка SHA256)
make smoke              # полный пайплайн на подвыборке, <5 минут
make reproduce          # полный прогон (часы CPU)
```

## Структура

```
configs/    default.yaml (все гиперпараметры), smoke.yaml, prereg.yaml
src/ecotypes/  пакет: panel, graphs, cluster, dynamics, icvi, synthetic, interpret, site
scripts/    01_download_data … 09_make_report — тонкие раннеры
tests/      pytest: контракты данных, инварианты, детерминизм, smoke
outputs/main/  итоговые таблицы (в git)
report/     методологический отчёт (рус.)
site/       интерактивный лендинг (single-file)
```

## Результат → артефакт → команда

(таблица заполняется по мере прогонов; каждая строка — `make <таргет>`)

## Данные и лицензии

Код — MIT. Данные СберИндекса и производные таблицы — CC BY-SA 4.0,
источники и хэши: [DATA.md](DATA.md). GPL-3 зависимости: [NOTICE](NOTICE).
Конкурс СберИндекс 2026, направление «Кластеризация».
