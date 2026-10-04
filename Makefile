# Единая точка входа. Каждый скрипт идемпотентен: выход существует и конфиг
# не менялся → skip (хэш-метка конфига в артефакте).
CONFIG ?= configs/default.yaml
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

.PHONY: env data panel graphs cluster dynamics icvi interpret site report reproduce smoke test

env:       ; uv sync --locked && uv run python -c "import ecotypes; print('env ok')"
data:      ; uv run python scripts/01_download_data.py
panel:     ; uv run python scripts/02_build_panel.py --config $(CONFIG)
graphs:    ; uv run python scripts/03_build_graphs.py --config $(CONFIG)
cluster:   ; uv run python scripts/04_cluster.py --config $(CONFIG)
dynamics:  ; uv run python scripts/05_dynamics.py --config $(CONFIG)
icvi:      ; uv run python scripts/06_icvi_compare.py --config $(CONFIG)
interpret: ; uv run python scripts/07_interpret.py --config $(CONFIG)
site:      ; uv run python scripts/08_build_site.py --config $(CONFIG)
report:    ; uv run python scripts/09_make_report.py --config $(CONFIG)
reproduce: data panel graphs cluster dynamics icvi interpret site report
smoke:     ; $(MAKE) reproduce CONFIG=configs/smoke.yaml
test:      ; uv run pytest -q
