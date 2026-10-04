# Единая точка входа. Каждый скрипт идемпотентен: выход существует и конфиг
# не менялся → skip (хэш-метка конфига в артефакте).
# Конвенция раннеров: --config [--overrides] [--out].
CONFIG ?= configs/default.yaml
OVERRIDES ?=
OUT ?= data/processed
ARGS = --config $(CONFIG) $(if $(OVERRIDES),--overrides $(OVERRIDES))
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

.PHONY: env data panel graphs cluster dynamics icvi interpret site report reproduce smoke test

env:       ; uv sync --locked && uv run python -c "import ecotypes; print('env ok')"
data:      ; uv run python scripts/01_download_data.py
panel:     ; uv run python scripts/02_build_panel.py $(ARGS) --out $(OUT)
graphs:    ; uv run python scripts/03_build_graphs.py $(ARGS) --out $(OUT)
cluster:   ; uv run python scripts/04_cluster.py $(ARGS) --out $(OUT)
dynamics:  ; uv run python scripts/05_dynamics.py $(ARGS) --out $(OUT)
icvi:      ; uv run python scripts/06_icvi_compare.py $(ARGS) --out $(OUT)
interpret: ; uv run python scripts/07_interpret.py $(ARGS) --out $(OUT)
site:      ; uv run python scripts/08_build_site.py $(ARGS)
report:    ; uv run python scripts/09_make_report.py $(ARGS)
reproduce: data panel graphs cluster dynamics icvi interpret site report
# smoke: полный пайплайн на подвыборке 200 МО, отдельный выходной каталог — полные артефакты не затираются
smoke:     ; $(MAKE) reproduce OVERRIDES=configs/smoke.yaml OUT=data/processed_smoke
test:      ; uv run pytest -q
