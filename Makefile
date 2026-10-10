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

.PHONY: env data match-monotowns panel graphs cluster dynamics measures icvi icvi-null synth \
        interpret drivers convergence laglead aux map report deck reproduce smoke test

env:             ; uv sync --locked && uv run python -c "import ecotypes; print('env ok')"
data:            ; uv run python scripts/01_download_data.py
match-monotowns: ; uv run python scripts/01b_match_monotowns.py $(ARGS)
panel:           ; uv run python scripts/02_build_panel.py $(ARGS) --out $(OUT)
graphs:          ; uv run python scripts/03_build_graphs.py $(ARGS) --out $(OUT)
cluster:         ; uv run python scripts/04_cluster.py $(ARGS) --out $(OUT)
dynamics:        ; uv run python scripts/05_dynamics.py $(ARGS) --out $(OUT)
measures:        ; uv run python scripts/06a_measures.py $(ARGS) --out $(OUT)
icvi:            ; uv run python scripts/06_icvi_compare.py $(ARGS) --out $(OUT)
icvi-null:       ; uv run python scripts/06c_icvi_null.py $(ARGS) --out $(OUT)
synth:           ; uv run python scripts/06b_synth_grid.py $(ARGS)
interpret:       ; uv run python scripts/07_interpret.py $(ARGS) --out $(OUT)
drivers:         ; uv run python scripts/07b_drivers.py $(ARGS) --out $(OUT)
convergence:     ; uv run python scripts/07c_convergence.py $(ARGS)
laglead:         ; uv run python scripts/07d_laglead.py $(ARGS) --out $(OUT)
aux:             ; uv run python scripts/07e_publish_aux.py $(ARGS) --out $(OUT)
map:             ; uv run python scripts/10_figure_map.py
report:          ; uv run python scripts/09_make_report.py $(ARGS)
# Презентация собирается из соседнего каталога deck/ (HTML → PDF, локальные
# шрифты, chrome-headless-shell) и кладётся в report/presentation.pdf.
deck:            ; cd ../deck && node build.mjs && cp presentation.pdf ../sberindex-ecotypes/report/presentation.pdf
# Воспроизведение полного численного конвейера
reproduce: data panel graphs cluster dynamics measures icvi icvi-null synth interpret drivers convergence laglead aux report
# smoke: полный пайплайн на подвыборке 200 МО, отдельный выходной каталог — полные артефакты не затираются
smoke:         ; $(MAKE) reproduce OVERRIDES=configs/smoke.yaml OUT=data/processed_smoke
test:          ; uv run pytest -q
