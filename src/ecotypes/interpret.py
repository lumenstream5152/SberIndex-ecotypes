"""Движок описаний типов (research/33_descriptor_engine.md, 05–06.10.2026).

Три независимых описателя + согласие + паспорта + именование + bootstrap:
- Миркин (§1): B_kv = N_k·c_kv² на z-стандартизованных признаках; первичный
  описатель — выводится из критерия кластеризации, не из суррогата. Инвариант
  B + W = T проверяется в тестах (и прогнан на синтетике 33 §8).
- Суррогатное дерево (§2): DecisionTree(max_depth, min_samples_leaf=20),
  fidelity глобальная И per-class (глобальная маскирует слабые типы),
  пороги откатываются в сырые единицы.
- TreeSHAP (§3): LGBMClassifier → shap_values multiclass = ndarray (N, p, K),
  НЕ list (ловушка старых туториалов). Fallback — permutation one-vs-rest (§3.4).
- Согласие (§4): Spearman/Kendall на полных векторах + Jaccard топ-3/5;
  сводный балл 0.5·Spearman + 0.5·Jaccard_топ5 (пороги 0.6 / 0.4).
- Паспорта (§5): 10 машинных полей на тип, оба слоя (макро k=3 + подтипы с
  пометкой пониженной seed-стабильности, PREREG_DEVIATIONS №8).
- Именование (§6): prevalence ≥ 0.6 и lift ≥ 2 (+ внешний факт: lift ≥ 2 и
  покрытие списка ≥ 30%); драфт в configs/type_names_draft.yaml, approved=false —
  утверждает владелец, не движок.
- Bootstrap (§7): recall@5 топ-признаков при перевыборке МО, метки фиксированы
  (устойчивость ОПИСАНИЙ, не разбиения); порог публикации ≥ 0.8.
- Внешняя валидация (research/34): моногорода 1398-р × тип, «Четыре России»
  (Р4 → Р1 → Р2 → Р3, именно в этом порядке), Энгель-чек на зарплате МО,
  курортные МО vs summer_amp_food, кейс Азнакаевский (tid 357).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, export_text

log = logging.getLogger(__name__)

__all__ = [
    "FEATURE_RU",
    "mirkin_profiles",
    "mirkin_decomposition",
    "surrogate_tree",
    "shap_profiles",
    "agreement_table",
    "build_passports",
    "naming_protocol",
    "bootstrap_stability",
    "mo_stats",
    "four_russias_labels",
    "validation_monotowns",
    "validation_four_russias",
    "validation_engel",
    "validation_resorts",
    "aznakay_case",
    "resolve_external_dir",
    "find_run",
    "run_all",
    "PASSPORT_FIELDS",
    "FORBIDDEN_NAMES",
]

# Подписи признаков для правил/паспортов (X_static, cluster.FEATURE_COLS).
FEATURE_RU = {
    "clr_mean_prod": "CLR доли продовольствия",
    "clr_mean_health": "CLR доли здоровья",
    "clr_mean_market": "CLR доли маркетплейсов",
    "clr_mean_food": "CLR доли общепита",
    "clr_mean_transp": "CLR доли транспорта",
    "clr_mean_proch": "CLR доли прочего",
    "level_mean": "log уровня трат",
    "yoy_all_med": "медианный г/г рост трат",
    "yoy_market_med": "г/г рост маркетплейсов",
    "dclr_market": "сдвиг CLR маркетплейсов за 24 мес",
    "dec_amp_all": "декабрьская амплитуда трат",
    "season_std": "сезонная волатильность",
}

# Имена эталонного конкурента (33 §6.3, configs/types.yaml Frolovskiy) — запрет.
FORBIDDEN_NAMES = [
    "Столичные агломерации", "Северные и ресурсные", "Крупные и средние города",
    "Сельские и малые", "Общепит и услуги", "Дорогая корзина без онлайна",
    "Базовая корзина малых городов", "Продукты и маркетплейсы",
    "Транспортная корзина",
]

PASSPORT_FIELDS = [
    "name_draft", "core_features", "geography", "population", "n_mo",
    "dynamics", "monotowns", "exemplars", "boundary_mos", "practical_note",
]

# «Четыре России» (34 §4): Р4 — 9 республик, приоритетно до остальных правил.
_R4_SUBSTR = ("дагестан", "ингушет", "чеченск", "кабардино-балкарск",
              "карачаево-черкесск", "северная осетия", "калмыки", "тыва",
              "республика алтай")
_R1_REGIONS = ("москва", "санкт-петербург", "севастополь")


# ------------------------------------------------------------------ утилиты

def _classes_of(labels: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(labels))


def _top_idx(row: np.ndarray, k: int) -> list[int]:
    row = np.nan_to_num(row, nan=-np.inf)
    return [int(i) for i in np.argsort(-row)[:k] if np.isfinite(row[i])]


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def find_run(outputs: str | Path, key: str, require: str | None = None,
             allow_smoke: bool = False) -> Path:
    """Последний outputs/<run>/metrics.json с ключом key (как в scripts/06).
    По умолчанию игнорирует smoke-прогоны, защищая прод от перезаписи артефактов."""
    for d in sorted(Path(outputs).iterdir(), reverse=True):
        if not allow_smoke and "smoke" in d.name:
            continue
        m = d / "metrics.json"
        if not m.exists() or (require is not None and not (d / require).exists()):
            continue
        try:
            if key in json.loads(m.read_text(encoding="utf-8")):
                return d
        except Exception:
            continue
    raise FileNotFoundError(f"в {outputs} нет прогона с metrics.json[{key!r}]")


def resolve_external_dir() -> Path:
    """data/external: сначала в репо, затем в родительском проекте (34: файлы
    моногородов/курортов лежат в ~/сбериндекс/data/external)."""
    repo = Path(__file__).resolve().parents[2]
    for base in [repo, *repo.parents]:
        cand = base / "data" / "external"
        if (cand / "monotowns_1398r_matched.csv").exists():
            return cand
    return repo / "data" / "external"  # дефолт (пусть падает с ясным именем)


# ------------------------------------------------------------------- Миркин

def mirkin_decomposition(X: np.ndarray, labels: np.ndarray,
                         classes: np.ndarray | None = None):
    """B_kv = N_k·c_kv² на z-стандартизованном X; rel = B_kv / T_v; знаки.
    Возвращает (B, rel, sign, W, T) — инвариант B.sum() + W == T (33 §1.2)."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    if classes is None:
        classes = _classes_of(labels)
    grand = X.mean(axis=0)
    T_v = (X ** 2).sum(axis=0)
    K, p = len(classes), X.shape[1]
    B = np.zeros((K, p))
    sign = np.zeros_like(B)
    W = 0.0
    for ki, k in enumerate(classes):
        idx = labels == k
        if not idx.any():
            B[ki] = np.nan
            continue
        c = X[idx].mean(axis=0) - grand
        B[ki] = idx.sum() * c ** 2
        sign[ki] = np.sign(c)
        W += float(((X[idx] - (c + grand)) ** 2).sum())
    rel = B / np.where(T_v == 0, 1.0, T_v)
    return B, rel, sign, W, float(T_v.sum())


def mirkin_profiles(X: np.ndarray, labels: np.ndarray, feature_names: list[str],
                    X_raw: np.ndarray | None = None) -> pd.DataFrame:
    """Длинная таблица вкладов: type_id, feature, B_kv, rel (B_kv/T_v), sign,
    d_pct (% к среднему по РФ на сырых; NaN при |mean|≈0 — CLR), rank в типе.
    Доля объяснённого рассеяния типа — type_explained_share (33 §1.3)."""
    classes = _classes_of(labels)
    B, rel, sign, W, T = mirkin_decomposition(X, labels, classes)
    rows = []
    for ki, k in enumerate(classes):
        order = _top_idx(rel[ki], len(feature_names))
        explained = float(np.nansum(B[ki]) / T) if T > 0 else np.nan
        for rank, v in enumerate(order, 1):
            d_pct = np.nan
            if X_raw is not None:
                xr = np.asarray(X_raw, dtype=np.float64)
                m_all = xr[:, v].mean()
                m_k = xr[np.asarray(labels) == k, v].mean()
                if abs(m_all) > 1e-9:
                    d_pct = float((m_k - m_all) / abs(m_all) * 100.0)
            rows.append(dict(
                type_id=int(k), feature=feature_names[v], B_kv=float(B[ki, v]),
                rel=float(rel[ki, v]), sign=float(sign[ki, v]), d_pct=d_pct,
                rank_in_type=rank, type_explained_share=explained))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- дерево

def _tree_class_importances(tree: DecisionTreeClassifier, labels: np.ndarray,
                            classes: np.ndarray, p: int) -> np.ndarray:
    """(K, p): суммарное снижение impurity по признаку на путях, ведущих в
    листья класса k, взвешенное размером листа (33 §4.1); строки нормированы."""
    t = tree.tree_
    K = len(classes)
    imp = np.zeros((K, p))
    # снижение impurity узла: weighted impurity decrease (sklearn хранит
    # impurity и n_node_samples; decrease = I·n − I_l·n_l − I_r·n_r)
    dec = np.zeros(t.node_count)
    for n in range(t.node_count):
        if t.children_left[n] != -1:
            l, r = t.children_left[n], t.children_right[n]
            dec[n] = (t.impurity[n] * t.n_node_samples[n]
                      - t.impurity[l] * t.n_node_samples[l]
                      - t.impurity[r] * t.n_node_samples[r])
    # пути до листьев
    def walk(n: int, path: list[int]):
        if t.children_left[n] == -1:  # лист
            maj = int(np.argmax(t.value[n][0]))
            leaf_n = t.n_node_samples[n]
            for m in path:
                f = t.feature[m]
                if f >= 0:
                    imp[maj, f] += dec[m] * leaf_n
            return
        walk(t.children_left[n], path + [n])
        walk(t.children_right[n], path + [n])

    walk(0, [])
    s = imp.sum(axis=1, keepdims=True)
    return imp / np.where(s == 0, 1.0, s)


def _tree_rules(tree: DecisionTreeClassifier, labels: np.ndarray,
                classes: np.ndarray, feature_names: list[str],
                means: np.ndarray | None,
                stds: np.ndarray | None) -> dict[int, dict]:
    """Правило класса = конъюнкция условий пути к доминирующему листу (max
    поддержка среди листьев класса); покрытие < 50% МО класса → +второй лист
    (дизъюнкция, 33 §2.1). Пороги откачены в сырые единицы (θ·std + mean)."""
    t = tree.tree_
    leaves = [n for n in range(t.node_count) if t.children_left[n] == -1]
    # родительские связи для восстановления пути
    parent = {}
    for n in range(t.node_count):
        if t.children_left[n] != -1:
            parent[t.children_left[n]] = (n, "left")
            parent[t.children_right[n]] = (n, "right")

    def path_conds(leaf: int) -> list[str]:
        conds = []
        n = leaf
        while n in parent:
            m, side = parent[n]
            thr = t.threshold[m]
            if means is not None and stds is not None:
                thr = thr * stds[t.feature[m]] + means[t.feature[m]]
            op = "<=" if side == "left" else ">"
            conds.append(f"{feature_names[t.feature[m]]} {op} {thr:.3f}")
            n = m
        return list(reversed(conds))

    rules: dict[int, dict] = {}
    for ki, k in enumerate(classes):
        my = [(n, t.n_node_samples[n]) for n in leaves
              if int(np.argmax(t.value[n][0])) == ki]
        n_k = int((labels == k).sum())
        if not my:
            rules[int(k)] = dict(rule=None, coverage=0.0,
                                 note="дерево не выделило лист классу")
            continue
        my.sort(key=lambda x: -x[1])
        best, cov = my[0]
        conds = path_conds(best)
        coverage = cov / max(n_k, 1)
        rule = " и ".join(conds)
        note = None
        if coverage < 0.5 and len(my) > 1:
            second, cov2 = my[1]
            rule = f"({rule}) ИЛИ ({' и '.join(path_conds(second))})"
            coverage = (cov + cov2) / max(n_k, 1)
            note = "дизъюнкция двух листьев: доминирующий < 50% класса"
        rules[int(k)] = dict(rule=rule, coverage=float(coverage), note=note)
    return rules


def surrogate_tree(X: np.ndarray, labels: np.ndarray,
                   feature_names: list[str], max_depth: int = 4,
                   min_samples_leaf: int = 20, seed: int = 0,
                   feature_means: np.ndarray | None = None,
                   feature_stds: np.ndarray | None = None) -> dict:
    """Суррогатное дерево меток (33 §2). fidelity глобальная + per class +
    macro-F1; export_text; правила классов в сырых единицах; (K,p) важности."""
    labels = np.asarray(labels)
    classes = _classes_of(labels)
    tree = DecisionTreeClassifier(max_depth=max_depth,
                                  min_samples_leaf=min_samples_leaf,
                                  random_state=seed)
    tree.fit(X, labels)
    pred = tree.predict(X)
    fid = float((pred == labels).mean())
    per_class = {int(k): float((pred[labels == k] == k).mean())
                 if (labels == k).any() else np.nan for k in classes}
    # macro-F1 без меток-пустышек
    f1s = []
    for k in classes:
        tp = int(((pred == k) & (labels == k)).sum())
        fp = int(((pred == k) & (labels != k)).sum())
        fn = int(((pred != k) & (labels == k)).sum())
        f1s.append(2 * tp / max(2 * tp + fp + fn, 1))
    return dict(
        model=tree,
        text=export_text(tree, feature_names=feature_names, decimals=2),
        fidelity=fid, fidelity_per_class=per_class,
        macro_f1=float(np.mean(f1s)),
        rules=_tree_rules(tree, labels, classes, feature_names, feature_means,
                          feature_stds),
        class_importances=pd.DataFrame(
            _tree_class_importances(tree, labels, classes, X.shape[1]),
            index=pd.Index([int(k) for k in classes], name="type_id"),
            columns=feature_names),
        params=dict(max_depth=max_depth, min_samples_leaf=min_samples_leaf),
    )


# --------------------------------------------------------------------- SHAP

def _fit_lgbm(X: np.ndarray, labels: np.ndarray, seed: int,
              n_estimators: int = 200):
    import lightgbm as lgb

    clf = lgb.LGBMClassifier(n_estimators=n_estimators, num_leaves=15,
                             learning_rate=0.05, random_state=seed,
                             deterministic=True, force_col_wise=True,
                             verbose=-1)
    clf.fit(X, labels)
    return clf


def _permutation_profiles(X, labels, classes, feature_names, seed,
                          n_repeats: int) -> np.ndarray:
    """Запасной путь §3.4: K бинарных one-vs-rest LGBM + permutation_importance
    → (K, p) важности. Медленнее и шумнее TreeSHAP, без внешних зависимостей."""
    from sklearn.inspection import permutation_importance

    K, p = len(classes), X.shape[1]
    out = np.zeros((K, p))
    for ki, k in enumerate(classes):
        y_bin = (labels == k).astype(int)
        if y_bin.min() == y_bin.max():
            out[ki] = np.nan
            continue
        clf = _fit_lgbm(X, y_bin, seed + ki, n_estimators=100)
        r = permutation_importance(clf, X, y_bin, n_repeats=n_repeats,
                                   random_state=seed + 1000 + ki)
        out[ki] = r.importances_mean
    return out


def shap_profiles(X: np.ndarray, labels: np.ndarray, feature_names: list[str],
                  seed: int = 0, permutation_repeats: int = 30) -> dict:
    """LGBM-суррогат меток → per-class mean|SHAP| (33 §3). shap 0.52:
    multiclass shap_values = ndarray (N, p, K) — нормализуем list/2D.
    При недоступности shap — permutation one-vs-rest (method='permutation').
    SHAP описывает СУРРОГАТ, не кластеризацию → fidelity печатается всегда."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    classes = _classes_of(labels)
    clf = _fit_lgbm(X, labels, seed)
    pred = clf.predict(X)
    fidelity = float((pred == labels).mean())
    per_class_acc = {int(k): float((pred[labels == k] == k).mean())
                     if (labels == k).any() else np.nan for k in classes}
    method = "treeshap"
    try:
        import shap

        sv = shap.TreeExplainer(clf).shap_values(X)
        if isinstance(sv, list):           # старые версии: list по классам
            sv = np.stack(sv, axis=-1)     # (N, p, K)
        sv = np.asarray(sv)
        if sv.ndim == 2:                   # бинарный случай: (N, p) класса 1
            sv = np.stack([-sv, sv], axis=-1)
        # sv: (N, p, K); столбец j отвечает classes[j] (clf.classes_ отсортирован)
        prof = np.abs(sv).mean(axis=0).T   # (K, p)
    except Exception as e:                 # ImportError и любые сбои shap
        log.warning("TreeSHAP недоступен (%s) → permutation fallback", e)
        method = "permutation"
        prof = _permutation_profiles(X, labels, classes, feature_names, seed,
                                     permutation_repeats)
    prof_df = pd.DataFrame(prof,
                           index=pd.Index([int(k) for k in classes],
                                          name="type_id"),
                           columns=feature_names)
    long = prof_df.stack().rename("mean_abs_shap").reset_index()
    long.columns = ["type_id", "feature", "mean_abs_shap"]
    long["rank_in_type"] = long.groupby("type_id")["mean_abs_shap"] \
        .rank(ascending=False).astype(int)
    return dict(profiles=long, matrix=prof_df, fidelity=fidelity,
                per_class_accuracy=per_class_acc, method=method,
                classes=[int(k) for k in classes])


# ------------------------------------------------------------------ согласие

def agreement_table(mirkin: pd.DataFrame | np.ndarray,
                    tree_imp: pd.DataFrame | np.ndarray,
                    shap_prof: pd.DataFrame | np.ndarray,
                    feature_names: list[str], top: tuple[int, ...] = (3, 5)
                    ) -> dict[str, pd.DataFrame]:
    """Согласие трёх описателей (33 §4). Входы — (K, p) матрицы или wide/long
    DataFrame с type_id; все три — ранжируемые оценки значимости признаков.
    pairs: Spearman/Kendall (полные векторы) + Jaccard топ-3/5 на трёх парах.
    summary: СОГЛАСИЕ_ТИПА = среднее по парам (0.5·Spearman + 0.5·Jaccard_топ5);
    пороги ≥0.6 «консистентно», 0.4–0.6 «ядро согласовано», <0.4 «спорят».
    На малых p смотрим на величину, не на p-value (33 §4.2)."""
    def to_mat(x) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            if "type_id" in x.columns:  # long → wide
                vals = [c for c in x.columns
                        if c not in ("type_id", "feature")]
                val = next((c for c in ("rel", "mean_abs_shap", "importance")
                            if c in vals), vals[-1])
                return (x.pivot(index="type_id", columns="feature",
                                values=val)[feature_names])
            return x[feature_names]
        return pd.DataFrame(x, columns=feature_names)

    M, Tm, S = to_mat(mirkin), to_mat(tree_imp), to_mat(shap_prof)
    classes = M.index
    pairs = [("mirkin_tree", M, Tm), ("mirkin_shap", M, S), ("tree_shap", Tm, S)]
    rows = []
    for ki, k in enumerate(classes):
        for pname, A_, B_ in pairs:
            a, b = A_.iloc[ki].to_numpy(float), B_.iloc[ki].to_numpy(float)
            ok = np.isfinite(a) & np.isfinite(b)
            rho = stats.spearmanr(a[ok], b[ok]).statistic if ok.sum() > 2 \
                else np.nan
            tau = stats.kendalltau(a[ok], b[ok]).statistic if ok.sum() > 2 \
                else np.nan
            row = dict(type_id=int(k), pair=pname,
                       spearman=float(rho), kendall=float(tau))
            for t in top:
                row[f"jaccard_top{t}"] = _jaccard(set(_top_idx(a, t)),
                                                  set(_top_idx(b, t)))
            rows.append(row)
    pairs_df = pd.DataFrame(rows)
    t5 = f"jaccard_top{top[-1]}"
    score = (pairs_df.groupby("type_id")
             .apply(lambda g: float(np.nanmean(
                 0.5 * g.spearman + 0.5 * g[t5])), include_groups=False)
             .rename("agreement_score").reset_index())
    score["verdict"] = pd.cut(score.agreement_score, [-np.inf, 0.4, 0.6, np.inf],
                              labels=["описания спорят",
                                      "ядро согласовано, хвост расходится",
                                      "консистентно тремя методами"])
    return dict(pairs=pairs_df, summary=score)


# ------------------------------------------------------------------ bootstrap

def bootstrap_stability(X: np.ndarray, labels: np.ndarray,
                        feature_names: list[str], B: int = 100, top: int = 5,
                        seed: int = 0, tree_params: dict | None = None,
                        shap_replicas: int = 0) -> pd.DataFrame:
    """Устойчивость ОПИСАНИЙ (33 §7): перевыборка МО с возвратом, метки
    фиксированы. Миркин — все B реплик; дерево — все B (дёшево); SHAP —
    shap_replicas реплик (дорогой). Метрики per тип per описатель: recall@top
    и Jaccard@top к исходному топ-набору; порог публикации recall ≥ 0.8."""
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    classes = _classes_of(labels)
    rng = np.random.default_rng(seed)
    tp = dict(max_depth=4, min_samples_leaf=20) | (tree_params or {})

    def base_tops() -> dict[str, dict[int, set[int]]]:
        _, rel, _, _, _ = mirkin_decomposition(X, labels, classes)
        tr = surrogate_tree(X, labels, feature_names, seed=seed, **tp)
        out = {"mirkin": {int(k): set(_top_idx(rel[ki], top))
                          for ki, k in enumerate(classes)},
               "tree": {int(k): set(_top_idx(
                   tr["class_importances"].iloc[ki].to_numpy(float), top))
                   for ki, k in enumerate(classes)}}
        if shap_replicas > 0:
            sh = shap_profiles(X, labels, feature_names, seed=seed)
            out["shap"] = {int(k): set(_top_idx(
                sh["matrix"].iloc[ki].to_numpy(float), top))
                for ki, k in enumerate(classes)}
        return out

    base = base_tops()
    hits: dict[str, dict[int, list[tuple[float, float]]]] = {
        d: {int(k): [] for k in classes} for d in base}
    n_shap_done = 0
    for b in range(B):
        idx = rng.choice(len(labels), len(labels), replace=True)
        Xb, lb = X[idx], labels[idx]
        _, rel_b, _, _, _ = mirkin_decomposition(Xb, lb, classes)
        tr_b = surrogate_tree(Xb, lb, feature_names,
                              seed=int(rng.integers(2**31)), **tp)
        rep = {"mirkin": rel_b,
               "tree": tr_b["class_importances"].to_numpy(float)}
        if shap_replicas > 0 and n_shap_done < shap_replicas:
            rep["shap"] = shap_profiles(
                Xb, lb, feature_names, seed=int(rng.integers(2**31))
            )["matrix"].to_numpy(float)
            n_shap_done += 1
        for desc, mat in rep.items():
            for ki, k in enumerate(classes):
                tops_b = set(_top_idx(mat[ki], top))
                inter = len(base[desc][int(k)] & tops_b)
                recall = inter / max(len(base[desc][int(k)]), 1)
                jac = _jaccard(base[desc][int(k)], tops_b)
                hits[desc][int(k)].append((recall, jac))
    rows = []
    for desc, per_k in hits.items():
        n_rep = shap_replicas if desc == "shap" else B
        for ki, k in enumerate(classes):
            if not per_k[int(k)]:
                continue
            rec = float(np.mean([r for r, _ in per_k[int(k)]]))
            jac = float(np.mean([j for _, j in per_k[int(k)]]))
            rows.append(dict(type_id=int(k), descriptor=desc,
                             recall_at_top=rec, jaccard_at_top=jac,
                             top=top, n_replicas=int(n_rep),
                             verdict=("устойчиво" if rec >= 0.8 else
                                      "устойчиво с оговоркой" if rec >= 0.6
                                      else "нестабильно")))
    return pd.DataFrame(rows)


# ----------------------------------------------------------- статистика МО

def mo_stats(panel: pd.DataFrame, nodes: pd.DataFrame) -> pd.DataFrame:
    """Средние за 24 мес доли категорий + log уровня + контекст nodes (B5 —
    только интерпретация/валидация, не кластеризация)."""
    agg = (panel.groupby("territory_id")
           .agg(share_prod=("share_prod", "mean"),
                share_food=("share_food", "mean"),
                share_market=("share_market", "mean"),
                share_health=("share_health", "mean"),
                share_transp=("share_transp", "mean"),
                level_panel=("log_all", "mean"))
           .reset_index())
    keep = ["territory_id", "name", "region_name", "mun_type", "pop_2024",
            "urban_share", "log_wage", "empl_pc", "log_ma", "level_mean",
            "summer_amp_food", "dec_amp_all", "season_std", "dclr_market"]
    return agg.merge(nodes[[c for c in keep if c in nodes.columns]],
                     on="territory_id", how="right")


# ------------------------------------------------------------------ паспорта

def _centroid_dynamics(panel: pd.DataFrame, labels_s: pd.Series,
                       classes: np.ndarray) -> dict[int, dict]:
    """Дрейф центроида типа по месяцам в z-пространстве 4 помесячных признаков
    (log_all + 3 доли): ||c_2024-12 − c_2023-01||₂ + топ-3 признака по |тренду|
    (линейный наклон центроида, 33 §5 п.5)."""
    feats = ["log_all", "share_prod", "share_food", "share_market"]
    df = panel[["territory_id", "month", *feats]].copy()
    z = df[feats].to_numpy(float)
    z = (z - np.nanmean(z, 0)) / np.where(np.nanstd(z, 0) == 0, 1,
                                          np.nanstd(z, 0))
    df[feats] = z
    df["type"] = df.territory_id.map(labels_s)
    months = sorted(df.month.unique())
    out: dict[int, dict] = {}
    tt = np.arange(len(months))
    for k in classes:
        sub = df[df.type == k]
        if sub.empty:
            continue
        c = sub.groupby("month")[feats].mean().reindex(months)
        c0, c1 = c.iloc[0].to_numpy(), c.iloc[-1].to_numpy()
        ok = np.isfinite(c0) & np.isfinite(c1)
        drift = float(np.linalg.norm((c1 - c0)[ok])) if ok.any() else np.nan
        trends = []
        for f in feats:
            y = c[f].to_numpy()
            m = np.isfinite(y)
            slope = float(np.polyfit(tt[m], y[m], 1)[0]) if m.sum() > 2 \
                else np.nan
            trends.append(dict(feature=f, slope=slope))
        trends = sorted(trends, key=lambda d: -abs(d["slope"] or 0))[:3]
        out[int(k)] = dict(drift_norm=drift, top_trends=trends)
    return out


def _switch_stats(dyn: pd.DataFrame, labels_s: pd.Series,
                  classes: np.ndarray) -> dict[int, dict]:
    """Доля МО типа, сменивших помесячный подтип (первый vs последний месяц
    присутствия), + среднее seed_agreement — слой подтипов с пониженной
    seed-стабильностью помечается (PREREG_DEVIATIONS №8)."""
    if dyn is None or dyn.empty:
        return {}
    d = dyn[dyn.present].sort_values("month") if "present" in dyn else \
        dyn.sort_values("month")
    g = d.groupby("territory_id").agg(first=("type_id_smooth", "first"),
                                      last=("type_id_smooth", "last"),
                                      seed_agr=("seed_agreement", "mean"))
    g["changed"] = (g.first != g.last).astype(float)
    g["type"] = g.index.map(labels_s)
    out = {}
    for k in classes:
        sub = g[g.type == k]
        if len(sub):
            out[int(k)] = dict(share_changed_subtype=float(sub.changed.mean()),
                               mean_seed_agreement=float(sub.seed_agr.mean()))
    return out


def build_passports(X_std: np.ndarray, X_raw: np.ndarray,
                    feature_names: list[str], labels: np.ndarray,
                    nodes: pd.DataFrame, *, layer: str,
                    mo_stats_df: pd.DataFrame | None = None,
                    panel: pd.DataFrame | None = None,
                    dyn: pd.DataFrame | None = None,
                    mono: pd.DataFrame | None = None,
                    mirkin_df: pd.DataFrame | None = None,
                    tree: dict | None = None,
                    shap: dict | None = None,
                    agreement: pd.DataFrame | None = None,
                    names: dict[int, str] | None = None,
                    stability_note: str | None = None,
                    top: int = 5) -> pd.DataFrame:
    """Паспорта типов — 10 машинных полей (33 §5, порядок PASSPORT_FIELDS):
    имя-черновик; ядро-признаки (топ-5 Миркина с d% + правило дерева в сырых
    единицах + отметка согласия SHAP); география; население; число МО;
    динамика (дрейф центроида + доля перешедших); моногорода; эталонные МО;
    граничные МО; практическая заметка-заготовка. Сложные поля — JSON-строки."""
    X_std = np.asarray(X_std, dtype=np.float64)
    labels = np.asarray(labels)
    classes = _classes_of(labels)
    nodes = nodes.reset_index(drop=True)
    labels_s = pd.Series(labels, index=nodes.territory_id)
    centroids = np.stack([X_std[labels == k].mean(0) for k in classes])
    if mirkin_df is None:
        mirkin_df = mirkin_profiles(X_std, labels, feature_names, X_raw=X_raw)
    shap_top = {}
    if shap is not None:
        for k in classes:
            if int(k) in shap["matrix"].index:
                row = shap["matrix"].loc[int(k)].to_numpy(float)
                shap_top[int(k)] = {feature_names[i]
                                    for i in _top_idx(row, top)}
    agr_score = {}
    if agreement is not None and len(agreement):
        agr_score = dict(zip(agreement.type_id, agreement.agreement_score))
    dyn_c = _centroid_dynamics(panel, labels_s, classes) \
        if panel is not None else {}
    dyn_s = _switch_stats(dyn, labels_s, classes)
    mono_by_tid = {}
    if mono is not None:  # в перечне бывает дубль МО (Закаменский ×2) — первый
        mono_by_tid = (mono.drop_duplicates("territory_id")
                       .set_index("territory_id")["category"].to_dict())
    mono_flags = nodes.territory_id.map(mono_by_tid) if mono is not None else None
    share_mono_all = float(mono_flags.notna().mean()) if mono is not None else 0.0

    rows = []
    for ki, k in enumerate(classes):
        idx = np.where(labels == k)[0]
        sub = nodes.iloc[idx]
        mk = mirkin_df[mirkin_df.type_id == k].sort_values("rank_in_type")
        core = []
        for _, r in mk.head(top).iterrows():
            core.append(dict(
                feature=r.feature, feature_ru=FEATURE_RU.get(r.feature,
                                                             r.feature),
                rel=round(r.rel, 4), sign=int(r.sign),
                d_pct=None if pd.isna(r.d_pct) else round(r.d_pct, 1),
                in_shap_top=bool(shap_top.get(int(k), set())
                                 and r.feature in shap_top[int(k)])))
        rule_raw = tree["rules"].get(int(k), {}) if tree else {}
        fid_pc = (tree["fidelity_per_class"].get(int(k)) if tree else None)
        geography = dict(
            top_regions=(sub.region_name.value_counts().head(5)
                         .to_dict()),
            share_gorod_okrug=float((sub.mun_type == "городской округ").mean()),
            share_mun_raion=float((sub.mun_type == "муниципальный район")
                                  .mean()))
        pop_total = int(sub.pop_2024.sum())
        pop_all = int(nodes.pop_2024.sum())
        population = dict(total=pop_total,
                          share=pop_total / max(pop_all, 1),
                          median_mo=float(sub.pop_2024.median()))
        dynamics = dyn_c.get(int(k), {}) | dyn_s.get(int(k), {})
        if not dynamics:
            dynamics = {"note": "нет помесячных данных для дрейфа/переходов"}
        if mono_flags is not None:
            mono_cat = mono_flags.iloc[idx].dropna()
            monotowns = dict(
                n=int(len(mono_cat)),
                share_in_type=float(len(mono_cat) / max(len(idx), 1)),
                lift=float((len(mono_cat) / max(len(idx), 1))
                           / max(share_mono_all, 1e-9)),
                by_category={str(int(c)): int(v) for c, v in
                             mono_cat.value_counts().items()})
        else:
            monotowns = dict(note="перечень 1398-р недоступен")
        d_own = np.linalg.norm(X_std[idx] - centroids[ki], axis=1)
        other = [j for j in range(len(classes)) if j != ki]
        d_other = (np.linalg.norm(X_std[idx][:, None, :]
                                  - centroids[other][None, :, :], axis=2)
                   .min(axis=1) if other else np.full(len(idx), np.inf))
        ratio = d_own / np.where(d_other == 0, 1e-9, d_other)
        ex_order = np.argsort(d_own)[:3]
        bd_order = np.argsort(-ratio)[:2]

        def _mo(i: int, dist: float) -> dict:
            r = sub.iloc[i]  # r["name"] скобками: Series.name — property индекса
            return dict(territory_id=int(r.territory_id), name=str(r["name"]),
                        region=str(r.region_name), dist=round(float(dist), 3))

        exemplars = [_mo(int(i), d_own[int(i)]) for i in ex_order]
        boundary = [_mo(int(i), d_own[int(i)]) for i in bd_order]
        top3 = ", ".join(c["feature_ru"] for c in core[:3])
        trend_up = next((t["feature"] for t in dynamics.get("top_trends", [])
                         if (t["slope"] or 0) > 0), None)
        name_k = names.get(int(k), f"тип K{int(k)}") if names \
            else f"тип K{int(k)}"
        note = (f"«{name_k}»: {len(idx)} МО, "
                f"{pop_total / 1e6:.1f} млн жителей "
                f"({population['share']:.0%} выборки); ядро: {top3}.")
        if trend_up:
            note += f" Растущий сегмент профиля: {trend_up}."
        else:
            note += " Устойчиво растущих сегментов в профиле нет."
        agr_k = agr_score.get(int(k), np.nan)
        if not np.isnan(agr_k) and agr_k < 0.5:
            note += (f" Внимание: три описателя типа расходятся "
                     f"(согласие {agr_k:.2f}) — портрет читать с осторожностью.")
        rows.append(dict(
            layer=layer, type_id=int(k),
            name_draft=names.get(int(k), f"тип K{int(k)}") if names
            else f"тип K{int(k)}",
            core_features=json.dumps(
                dict(top=core, tree_rule_raw=rule_raw.get("rule"),
                     tree_rule_coverage=rule_raw.get("coverage"),
                     tree_fidelity_class=fid_pc), ensure_ascii=False),
            geography=json.dumps(geography, ensure_ascii=False),
            population=json.dumps(population, ensure_ascii=False),
            n_mo=int(len(idx)),
            dynamics=json.dumps(dynamics, ensure_ascii=False),
            monotowns=json.dumps(monotowns, ensure_ascii=False),
            exemplars=json.dumps(exemplars, ensure_ascii=False),
            boundary_mos=json.dumps(boundary, ensure_ascii=False),
            practical_note=note,
            agreement_score=agr_score.get(int(k), np.nan),
            stability_note=stability_note,
        ))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- имя типа

def _name_candidates(q: dict) -> list[dict]:
    """Библиотека кандидатов (33 §6): pattern по квантилям выборки + внешний
    факт. Имена — драфты, НЕ из списка FORBIDDEN_NAMES (проверяется ниже)."""
    return [
        dict(key="prod_periphery", name="Продовольственная периферия",
             cond=lambda s: (s.share_prod >= q["share_prod_hi"])
             & (s.level_mean <= q["level_med"]) & (s.urban_share < 0.5),
             external=None),
        dict(key="service_cores", name="Сервисные городские ядра",
             cond=lambda s: (s.share_food >= q["share_food_hi"])
             & (s.level_mean >= q["level_hi"]) & (s.urban_share >= 0.9),
             external=None),
        dict(key="mid_service", name="Региональные сервисные центры",
             cond=lambda s: (s.share_food >= q["share_food_med"])
             & (s.level_mean >= q["level_med"]) & (s.urban_share >= 0.5),
             external=None),
        dict(key="resort_summer", name="Летне-курортный профиль",
             cond=lambda s: s.summer_amp_food >= q["summer_hi"],
             external="resort"),
        dict(key="north_seasonal", name="Сезонные северные территории",
             cond=lambda s: (s.dec_amp_all >= q["dec_hi"])
             & (s.season_std >= q["season_hi"]),
             external=None),
        dict(key="mp_convergent", name="Маркетплейс-конвергенты",
             cond=lambda s: s.dclr_market >= q["dclr_hi"],
             external=None),
        dict(key="monotown", name="Моногородской профиль",
             cond=lambda s: s.is_monotown.fillna(False),
             external="monotown"),
    ]


def _norm_words(name: str) -> frozenset:
    return frozenset(name.lower().replace("ё", "е").split())


def naming_protocol(labels: np.ndarray, mo_stats_df: pd.DataFrame, *,
                    layer: str, external_lists: dict[str, pd.DataFrame],
                    forbidden: list[str] | None = None,
                    min_prevalence: float = 0.6, min_lift: float = 2.0,
                    min_coverage: float = 0.30) -> dict:
    """Протокол именования (33 §6): для каждого типа — кандидаты с evidence
    pack: prevalence паттерна в типе ≥ 0.6 и lift ≥ 2; если имя требует
    внешнего факта — lift к списку ≥ 2 и покрытие списка типом ≥ 30%.
    Движок НЕ утверждает: approved=false, утверждает владелец."""
    forbidden = FORBIDDEN_NAMES if forbidden is None else forbidden
    forb_sets = [_norm_words(x) for x in forbidden]
    labels = np.asarray(labels)
    classes = _classes_of(labels)
    s = mo_stats_df.copy()
    s["type"] = labels
    s["is_monotown"] = s.territory_id.isin(
        external_lists.get("monotown", pd.DataFrame(columns=["territory_id"]))
        .territory_id)
    q = dict(share_prod_hi=s.share_prod.quantile(2 / 3),
             share_food_hi=s.share_food.quantile(2 / 3),
             share_food_med=s.share_food.quantile(0.5),
             level_med=s.level_mean.quantile(0.5),
             level_hi=s.level_mean.quantile(2 / 3),
             summer_hi=s.summer_amp_food.quantile(2 / 3),
             dec_hi=s.dec_amp_all.quantile(2 / 3),
             season_hi=s.season_std.quantile(2 / 3),
             dclr_hi=s.dclr_market.quantile(2 / 3))
    layers_out: dict[str, dict] = {}
    for k in classes:
        sub = s[s.type == k]
        cands = []
        for cand in _name_candidates(q):
            if _norm_words(cand["name"]) in forb_sets:
                continue  # имя занято конкурентом (33 §6.3)
            try:
                hit = cand["cond"](sub)
                hit_all = cand["cond"](s)
            except Exception:
                continue
            prev = float(hit.mean())
            prev_all = float(hit_all.mean())
            lift = prev / max(prev_all, 1e-9)
            ext_lift = ext_cov = None
            verdict = prev >= min_prevalence and lift >= min_lift
            reason = None
            if cand["external"]:
                ext_df = external_lists.get(cand["external"])
                if ext_df is None or not len(ext_df):
                    verdict, reason = False, "внешний список недоступен"
                else:
                    in_core = ext_df.territory_id.isin(s.territory_id)
                    ids = set(ext_df.territory_id[in_core])
                    inside = set(sub.territory_id) & ids
                    ext_cov = len(inside) / max(len(ids), 1)
                    ext_share = len(inside) / max(len(sub), 1)
                    ext_all = len(ids) / len(s)
                    ext_lift = ext_share / max(ext_all, 1e-9)
                    if not (ext_lift >= min_lift and ext_cov >= min_coverage):
                        verdict = False
                        reason = (f"внешний факт слаб: lift={ext_lift:.2f}, "
                                  f"покрытие списка={ext_cov:.0%}")
            if not verdict and reason is None:
                reason = (f"prevalence={prev:.2f} (нужно ≥{min_prevalence}), "
                          f"lift={lift:.2f} (нужно ≥{min_lift})")
            cands.append(dict(
                name=cand["name"], prevalence=round(prev, 3),
                lift=round(lift, 2),
                external_lift=None if ext_lift is None else round(ext_lift, 2),
                external_coverage=None if ext_cov is None else round(ext_cov, 3),
                verdict=bool(verdict),
                reason=None if verdict else reason))
        layers_out[str(int(k))] = dict(
            type_id=int(k), proposed_name=None, candidates=cands, note=None)
    # уникальность имён внутри слоя: одно имя — один тип (тот, где prevalence
    # выше; тай-брейк — lift); у проигравших типов кандидат снимается
    winners: dict[str, tuple[str, float, float]] = {}
    for tid, blk in layers_out.items():
        for c in blk["candidates"]:
            if not c["verdict"]:
                continue
            cur = winners.get(c["name"])
            key = (c["prevalence"], c["lift"])
            if cur is None or key > (cur[1], cur[2]):
                winners[c["name"]] = (tid, *key)
    for tid, blk in layers_out.items():
        for c in blk["candidates"]:
            if c["verdict"] and winners[c["name"]][0] != tid:
                c["verdict"] = False
                c["reason"] = (f"имя занято типом K{winners[c['name']][0]} "
                               "(там prevalence выше) — уникальность в слое")
        proposed = next((c["name"] for c in blk["candidates"]
                         if c["verdict"]), None)
        blk["proposed_name"] = proposed
        blk["note"] = (None if proposed else
                       "имя не присвоено: ни один кандидат не прошёл фильтры — "
                       f"остаётся «тип K{blk['type_id']}»")
    return dict(layer=layer, approved=False,
                rule=("prevalence ≥ 0.6 и lift ≥ 2; внешний факт: lift ≥ 2 и "
                      "покрытие ≥ 30% списка (33 §6.4); утверждает владелец"),
                types=layers_out)


# ------------------------------------------------------- внешняя валидация

def validation_monotowns(labels: np.ndarray, nodes: pd.DataFrame,
                         mono: pd.DataFrame) -> pd.DataFrame:
    """Кросс-таб тип × категория 1398-р (0 = не моногород) + lift к базе."""
    df = nodes[["territory_id"]].copy()
    df["type"] = labels
    cat = mono.drop_duplicates("territory_id").set_index("territory_id")[
        "category"]
    df["category"] = df.territory_id.map(cat).fillna(0).astype(int)
    base = df.category.value_counts(normalize=True)
    rows = []
    for (k, c), g in df.groupby(["type", "category"]):
        share = len(g) / int((df.type == k).sum())
        rows.append(dict(type_id=int(k), category=int(c), n=len(g),
                         share_in_type=share,
                         lift=float(share / base.get(c, np.nan))))
    return pd.DataFrame(rows)


def four_russias_labels(nodes: pd.DataFrame, mono: pd.DataFrame,
                        employment: pd.DataFrame) -> pd.Series:
    """«Четыре России» — воспроизводимый маппинг (34 §4): Р4 (9 республик)
    → Р1 (ГО ≥500k или Мск/СПб/Сев) → Р2 (моногород или B+C ≥ 25% занятости
    при 20k ≤ pop ≤ 250k) → Р3. Порядок приоритета существенен."""
    df = nodes[["territory_id", "ok8", "region_name", "mun_type",
                "pop_2024"]].copy()
    reg = df.region_name.str.lower()
    is_r4 = reg.apply(lambda r: any(t in r for t in _R4_SUBSTR))
    is_r1 = ((df.pop_2024 >= 500_000) & (df.mun_type == "городской округ")) \
        | reg.apply(lambda r: any(t in r for t in _R1_REGIONS))
    mono_ids = set(mono.territory_id)
    emp = employment.copy()
    emp["ok8"] = emp.oktmo.astype(np.int64).astype(str).str.zfill(8)
    # последний доступный год на МО (2024, fallback 2023→2022 — 34 §РЕШЕНИЯ)
    emp = emp.sort_values("year")
    piv = emp.pivot_table(index="ok8", columns="okved2", values="value",
                          aggfunc="last")
    bc_cols = [c for c in piv.columns
               if c.startswith("Раздел В") or c.startswith("Раздел C")]
    tot_col = [c for c in piv.columns if c.startswith("Всего")]
    bc_share = (piv[bc_cols].sum(axis=1)
                / piv[tot_col[0]].where(piv[tot_col[0]] > 0)) \
        if bc_cols and tot_col else pd.Series(dtype=float)
    df["bc_share"] = df.ok8.astype(str).map(bc_share)
    is_r2 = (df.territory_id.isin(mono_ids)
             | ((df.bc_share >= 0.25)
                & df.pop_2024.between(20_000, 250_000)))
    out = np.select([is_r4, is_r1, is_r2], ["Р4", "Р1", "Р2"], default="Р3")
    return pd.Series(out, index=nodes.territory_id, name="four_russias")


def validation_four_russias(labels: np.ndarray, nodes: pd.DataFrame,
                            fr: pd.Series) -> pd.DataFrame:
    """Кросс-таб тип × «Четыре России» (доли в типе + lift к базе)."""
    df = nodes[["territory_id"]].copy()
    df["type"] = labels
    df["fr"] = df.territory_id.map(fr)
    base = df.fr.value_counts(normalize=True)
    rows = []
    for (k, r), g in df.groupby(["type", "fr"]):
        share = len(g) / int((df.type == k).sum())
        rows.append(dict(type_id=int(k), russia=r, n=len(g),
                         share_in_type=share,
                         lift=float(share / base.get(r, np.nan))))
    return pd.DataFrame(rows)


def validation_engel(labels: np.ndarray, mo_stats_df: pd.DataFrame) -> dict:
    """Энгель-чек (34 §РЕШЕНИЯ п.3): доля продовольствия vs зарплата МО по
    типам — ожидается монотонно обратная связь. Сила — уровень МО, не регион.
    Оговорка: наши доли — в безналичных тратах, не во всех расходах (12 §3.1)."""
    s = mo_stats_df.copy()
    s["type"] = labels
    g = s.groupby("type").agg(share_prod=("share_prod", "mean"),
                              log_wage=("log_wage", "median"))
    g["wage_rub"] = np.exp(g.log_wage)
    g = g.sort_values("wage_rub")
    rho = stats.spearmanr(g.wage_rub, g.share_prod).statistic \
        if len(g) > 2 else np.nan
    monotone = bool(g.share_prod.is_monotonic_decreasing)
    n_types = len(g)
    return dict(table=g.reset_index()[["type", "wage_rub", "share_prod"]],
                spearman_wage_food=float(rho), monotone_inverse=monotone,
                note=("ожидание: share_prod ↓ с ростом зарплаты (закон Энгеля). "
                      f"Оговорка: типов всего {n_types} — Spearman на n={n_types} "
                      "статистически тривиален (для n=3 точный p=1/3 даже при "
                      "идеальном ρ); чек качественный, а не значимостный"))


def validation_resorts(nodes: pd.DataFrame, resorts: pd.DataFrame) -> dict:
    """Курортные МО (эксперимент курортного сбора, 16 МО) vs summer_amp_food:
    ожидание — летняя амплитуда выше выборочной (33/34)."""
    in_core = resorts[resorts.territory_id.isin(nodes.territory_id)]
    amp = nodes.set_index("territory_id").summer_amp_food
    r = amp.loc[in_core.territory_id]
    rest = amp.drop(index=in_core.territory_id, errors="ignore")
    return dict(n_in_core=int(len(in_core)), n_list=int(len(resorts)),
                median_amp_resorts=float(r.median()) if len(r) else np.nan,
                median_amp_rest=float(rest.median()),
                share_above_q75=float((r > rest.quantile(0.75)).mean())
                if len(r) else np.nan,
                table=pd.DataFrame(dict(
                    territory_id=in_core.territory_id.values,
                    summer_amp_food=r.values)))


def aznakay_case(nodes: pd.DataFrame, labels: np.ndarray,
                 mo_stats_df: pd.DataFrame, tid: int = 357) -> dict:
    """Кейс Азнакаевский район (tid 357, research/40): нефтяной район с
    зарплатой среднего города, но потребительским профилем периферии —
    «зарплата ≠ безналичное потребление». Одна строка фактов."""
    match = np.where(nodes.territory_id.to_numpy() == tid)[0]
    if len(match) == 0:
        i = 0
        tid = int(nodes.territory_id.iloc[0])
    else:
        i = int(match[0])
    r = nodes.iloc[i]
    k = int(labels[i])
    s = mo_stats_df.set_index("territory_id")
    same = pd.Series(labels, index=nodes.territory_id) == k
    wage_type = float(np.exp(s.loc[same[same].index, "log_wage"]).median())
    facts = dict(
        territory_id=tid, name=str(r["name"]), region=str(r.region_name),
        type_id=k,
        wage_rub=float(np.exp(r.log_wage)), wage_type_median=wage_type,
        wage_vs_type=float(np.exp(r.log_wage) / wage_type - 1),
        share_prod=float(s.loc[tid, "share_prod"]),
        share_prod_type=float(s.loc[same[same].index, "share_prod"].mean()),
        level=float(r.level_mean),
        level_type=float(s.loc[same[same].index, "level_mean"].mean()),
        pop_2024=int(r.pop_2024))
    facts["line"] = (
        f"{r['name']} ({r.region_name}): тип K{k}; зарплата "
        f"{facts['wage_rub'] / 1e3:.0f} тыс. ₽ — на {facts['wage_vs_type']:+.0%} "
        f"к медиане типа, но доля продовольствия {facts['share_prod']:.1%} "
        f"(среднее типа {facts['share_prod_type']:.1%}) — потребительский "
        "профиль периферии при зарплате выше типовой.")
    return facts


# ------------------------------------------------------------- оркестратор

def run_all(cfg, *, out_root: str | Path = "data/processed",
            outputs: str | Path = "outputs", ctx=None,
            shap_boot_replicas: int = 30, boot_B: int = 100) -> dict:
    """Полный прогон движка: макро (leiden_consensus из последнего 04) +
    подтипы (type_id_smooth, снимок 2024-12 — с пометкой пониженной
    seed-стабильности, PREREG_DEVIATIONS №8). Артефакты — в ctx.dir:
    passports_{macro,subtypes}.parquet, descriptors_{mirkin,shap}_*.parquet,
    agreement.parquet, bootstrap.parquet, validation_*.parquet,
    configs/type_names_draft.yaml, metrics.json."""
    from . import cluster as cl
    from .seeds import stage_seed

    def _log(msg: str):
        (ctx.log if ctx else log.info)(msg)

    root = Path(out_root)
    is_smoke = "smoke" in str(out_root) or bool(ctx and "smoke" in getattr(ctx, "config_name", ""))
    run04 = find_run(outputs, "gamma_star", require="labels.parquet", allow_smoke=is_smoke)
    lab04 = pd.read_parquet(run04 / "labels.parquet").sort_values("row_idx")
    nodes_all = pd.read_parquet(root / "nodes_static.parquet")
    nodes = (lab04[["territory_id", "row_idx", "leiden_consensus"]]
             .merge(nodes_all, on="territory_id")
             .sort_values("row_idx").reset_index(drop=True))
    X, feat = cl.feature_matrix(nodes, cfg)
    X_raw = nodes[feat].to_numpy(np.float64)
    means, stds = X_raw.mean(0), np.where(X_raw.std(0) == 0, 1, X_raw.std(0))
    panel = pd.read_parquet(root / "panel_monthly.parquet")
    stats_df = mo_stats(panel, nodes)
    dyn = pd.read_parquet(root / "dynamics" / "labels.parquet")
    ext = resolve_external_dir()
    mono = pd.read_csv(ext / "monotowns_1398r_matched.csv")
    resorts = pd.read_csv(ext / "resort_mo_kurortny_sbor.csv")
    _log(f"входы: {len(nodes)} МО, {len(feat)} признаков; метки 04 ← "
         f"{run04.name}; внешние ← {ext}")

    sub_last = (dyn[dyn.month == dyn.month.max()]
                .set_index("territory_id").reindex(nodes.territory_id))
    sub_lab = sub_last.type_id_smooth.copy()
    if sub_lab.isna().any():  # МО без снимка последнего месяца → мода по месяцам
        _log(f"подтипы: {int(sub_lab.isna().sum())} МО без метки "
             f"{dyn.month.max()} → модальная метка по всем месяцам")
        mode_lab = (dyn[dyn.present].groupby("territory_id")
                    .type_id_smooth.agg(lambda s: s.mode().iloc[0]))
        sub_lab = sub_lab.fillna(nodes.territory_id.map(mode_lab)).astype(int)
    layers = {
        "macro": dict(labels=nodes.leiden_consensus.to_numpy(),
                      stability_note=None),
        "subtypes": dict(labels=sub_lab.to_numpy(),
                         stability_note=(
                             "подтиповый слой: пониженная seed-стабильность "
                             "(ari_med 0.67–0.72, PREREG_DEVIATIONS №8); снимок "
                             f"{dyn.month.max()}, type_id_smooth")),
    }
    seed_i = stage_seed(cfg.seed, "interpret")
    depth = int(cfg.interpret.surrogate_max_depth)
    results: dict[str, dict] = {}
    for layer, cfg_l in layers.items():
        labels = np.asarray(cfg_l["labels"])
        _log(f"[{layer}] типов: {len(np.unique(labels))}")
        mk = mirkin_profiles(X, labels, feat, X_raw=X_raw)
        tr = surrogate_tree(X, labels, feat, max_depth=depth,
                            seed=seed_i, feature_means=means,
                            feature_stds=stds)
        _log(f"[{layer}] дерево: fidelity={tr['fidelity']:.3f}, per-class "
             f"{ {k: round(v, 3) for k, v in tr['fidelity_per_class'].items()} }")
        sh = shap_profiles(X, labels, feat, seed=seed_i)
        _log(f"[{layer}] SHAP ({sh['method']}): fidelity суррогата "
             f"{sh['fidelity']:.3f}")
        agr = agreement_table(mk.pivot(index="type_id", columns="feature",
                                       values="rel")[feat],
                              tr["class_importances"], sh["matrix"], feat)
        boot = bootstrap_stability(
            X, labels, feat, B=boot_B, seed=seed_i,
            tree_params=dict(max_depth=depth), shap_replicas=shap_boot_replicas)
        naming = naming_protocol(labels, stats_df, layer=layer,
                                 external_lists=dict(monotown=mono,
                                                     resort=resorts))
        # оверлей утверждения владельца: движок генерирует драфт, владелец
        # утверждает в configs/type_names_draft.yaml; при approved=true
        # proposed_name из конфига перекрывает сгенерированный — и сохраняется
        # в последующих дампах, паспортах и metrics.json
        names_cfg = Path("configs/type_names_draft.yaml")
        if names_cfg.exists():
            try:
                prev = yaml.safe_load(names_cfg.read_text(encoding="utf-8")) or {}
                lyr_prev = prev.get(layer) or {}
                if lyr_prev.get("approved"):
                    naming["approved"] = True
                    for kk, vv in (lyr_prev.get("types") or {}).items():
                        pn = (vv or {}).get("proposed_name")
                        if not pn:
                            continue
                        for kk2, vv2 in naming["types"].items():
                            if str(kk2) == str(kk):
                                vv2["proposed_name"] = pn
            except Exception:
                log.warning("не смог прочитать утверждения имён", exc_info=True)
        names = {int(k): v["proposed_name"] or f"тип K{k}"
                 for k, v in naming["types"].items()}
        pp = build_passports(X, X_raw, feat, labels, nodes, layer=layer,
                             mo_stats_df=stats_df, panel=panel, dyn=dyn,
                             mono=mono, mirkin_df=mk, tree=tr, shap=sh,
                             agreement=agr["summary"], names=names,
                             stability_note=cfg_l["stability_note"])
        results[layer] = dict(mirkin=mk, tree=tr, shap=sh, agreement=agr,
                              bootstrap=boot, naming=naming, passports=pp,
                              labels=labels)
    # ------------------------------------------------ внешняя валидация
    lab_macro = results["macro"]["labels"]
    v_mono = validation_monotowns(lab_macro, nodes, mono)
    emp = pd.read_csv(ext / "rosstat_pmo_employment_okved_annual.csv")
    fr = four_russias_labels(nodes, mono, emp)
    v_fr = validation_four_russias(lab_macro, nodes, fr)
    v_eng = validation_engel(lab_macro, stats_df)
    v_res = validation_resorts(nodes, resorts)
    v_az = aznakay_case(nodes, lab_macro, stats_df)
    _log(f"Энгель: spearman={v_eng['spearman_wage_food']:.2f}, "
         f"монотонно обратная: {v_eng['monotone_inverse']}")

    if ctx is not None:
        out = ctx.dir
        for layer, r in results.items():
            r["passports"].to_parquet(out / f"passports_{layer}.parquet",
                                      index=False)
            r["mirkin"].to_parquet(out / f"descriptors_mirkin_{layer}.parquet",
                                   index=False)
            r["shap"]["profiles"].to_parquet(
                out / f"descriptors_shap_{layer}.parquet", index=False)
            r["agreement"]["pairs"].assign(layer=layer).to_parquet(
                out / f"agreement_pairs_{layer}.parquet", index=False)
            r["bootstrap"].assign(layer=layer).to_parquet(
                out / f"bootstrap_{layer}.parquet", index=False)
        agr_all = pd.concat([r["agreement"]["summary"].assign(layer=l)
                             for l, r in results.items()])
        agr_all.to_parquet(out / "agreement.parquet", index=False)
        v_mono.to_parquet(out / "validation_monotowns.parquet", index=False)
        v_fr.to_parquet(out / "validation_four_russias.parquet", index=False)
        v_eng["table"].to_parquet(out / "validation_engel.parquet",
                                  index=False)
        v_res["table"].to_parquet(out / "validation_resorts.parquet",
                                  index=False)
        pd.DataFrame([{k: v for k, v in v_az.items() if k != "line"}]). \
            to_parquet(out / "validation_aznakay.parquet", index=False)
        names_path = (out / "type_names_draft.yaml") if is_smoke else Path("configs/type_names_draft.yaml")
        with open(names_path, "w", encoding="utf-8") as f:
            yaml.safe_dump({l: r["naming"] for l, r in results.items()}, f,
                           allow_unicode=True, sort_keys=False)
        metrics = dict(
            run04=run04.name, features=feat,
            layers={l: dict(
                n_types=int(len(np.unique(r["labels"]))),
                tree_fidelity=r["tree"]["fidelity"],
                tree_fidelity_per_class=r["tree"]["fidelity_per_class"],
                tree_macro_f1=r["tree"]["macro_f1"],
                shap_method=r["shap"]["method"],
                shap_fidelity=r["shap"]["fidelity"],
                agreement=r["agreement"]["summary"].to_dict("records"),
                bootstrap=r["bootstrap"].to_dict("records"),
                proposed_names={k: v["proposed_name"]
                                for k, v in r["naming"]["types"].items()},
            ) for l, r in results.items()},
            validation=dict(
                monotowns=v_mono.to_dict("records"),
                four_russias=v_fr.to_dict("records"),
                engel={k: (v.to_dict("records") if isinstance(v, pd.DataFrame)
                           else v) for k, v in v_eng.items()},
                resorts={k: (v.to_dict("records")
                             if isinstance(v, pd.DataFrame) else v)
                         for k, v in v_res.items()},
                aznakay=v_az["line"],
            ))
        ctx.write_metrics(metrics)
    return dict(results=results, validation=dict(
        monotowns=v_mono, four_russias=v_fr, engel=v_eng, resorts=v_res,
        aznakay=v_az), nodes=nodes, feature_names=feat)
