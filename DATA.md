# DATA.md — источники, хэши, атрибуции

## Политика
Сырые данные **не хранятся в git** (data/raw/ в .gitignore). Они приезжают
`make data` (скрипт `scripts/01_download_data.py`) с проверкой SHA256.
Производные таблицы `outputs/main/` (метки, метрики, паспорта типов, <2 МБ)
коммитим — они доступны по CC BY-SA 4.0 (см. NOTICE).

## Источники и SHA256 (зафиксировано 04.10.2026)

| Файл | Источник | SHA256 |
|---|---|---|
| hackathonlicence.zip (consumption/market_access/connection.parquet + лицензия PDF) | https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip | a9f932ff4096a7df797d1547987937f34d3995ac445b4748177114488d12b010 |
| hackathon/hackathonlicence/consumption.parquet | внутри zip | 9833ddaaee7b2a182ed4cceeed16469031700ea87d508bd976150c6770ef8a61 |
| hackathon/hackathonlicence/connection.parquet | внутри zip | 20cbd5213d3ac1d0a867f097b485431811614b283d7366eb5cba9f65dc80493d |
| hackathon/hackathonlicence/market_access.parquet | внутри zip | 434258afe322b7e6e6610b2552d129ae47094613de72dae3d13f6965a28d1dc1 |
| t_dict_municipal_districts.xlsx | https://www.sberbank.com/common/files/t_dict_municipal.rar | 4150658c3298fbc87ed79838f503f3a5b9a28da33257ff803c7d9231ddb775d6 |
| t_dict_municipal_districts_poly.gpkg | там же | e62027630d48e4a13f9b6d173dd074d7358706743fe859ca3b7fa30fefa9813a |

Замечание: sberbank.com отдаёт нестандартную цепочку сертификатов (НУЦ).
Скрипт сначала пробует обычный TLS, при ошибке сертификата — `curl -k`
с громким предупреждением; целостность гарантирует сверка SHA256.
Ручной фолбэк: скачать по ссылкам выше и положить в `data/raw/`.

## Атрибуции (CC BY-SA 4.0)

RU: «Потребительские безналичные расходы на уровне муниципальных образований.
СберИндекс. Данные доступны по адресу https://sberindex.ru/ru/research/data-sense
(данные скачаны 04.10.2026)».
RU: «Автодорожные и железнодорожные связи между муниципальными образованиями.
СберИндекс. https://sberindex.ru/ru/research/data-sense (данные скачаны 04.10.2026)».
RU: «Данные о границах и преобразованиях муниципальных образований. СберИндекс.
https://sberindex.ru/ru/research/data-sense (данные скачаны 04.10.2026)».
Лицензия: CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/legalcode.ru

EN: “Consumer cashless spending at the municipal level. SberIndex.
Available at https://sberindex.ru/ru/research/data-sense (downloaded 04.10.2026)”.
License: CC BY-SA 4.0.

Росстат, БД показателей муниципальных образований — через каталог «Если быть
точным» (tochno-st): https://storage.yandexcloud.net/tochno-st-catalog/Rosstat/ — CC BY 4.0.

data/external/ (коммитится — малые валидационные файлы, нужны make panel/interpret
на чистом клоне; research/34):
rosstat_pmo_population_2022_2024_compact.csv, rosstat_pmo_employment_okved_annual.csv,
rosstat_pmo_wages_total_annual.csv — Росстат БД ПМО, выгрузка tochno.st v20250918,
скачано 04.10.2026, CC BY 4.0 (индикаторы Y48112027, Y48423005, Y48423007 за 2022–2024).
monotowns_1398r_full.csv — перечень моногородов (распоряжение Правительства РФ
№1398-р), 312 строк; monotowns_1398r_matched.csv / _unmatched.csv — результат
матчинга к справочнику СберИндекса, воспроизводится `make`-шагом
`scripts/01b_match_monotowns.py` (эвристика задокументирована в нём).
resort_mo_kurortny_sbor.csv — МО эксперимента курортного сбора (16 МО).
