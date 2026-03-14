# Звіт: Лабораторна Робота №3  
**Гіперпараметрична оптимізація та оркестрація ML-пайплайнів з Optuna**

---

## 1. Мета роботи

1. Реалізувати objective function для Optuna та коректно оцінювати модель на validation наборі.
2. Виконати гіперпараметричну оптимізацію для XGBoost (з попередньої ЛР1) за допомогою Optuna (n_trials ≥ 20).
3. Логувати кожен trial у MLflow як дочірній (nested) run усередині батьківського run.
4. Порівняти два sampler-и: **TPE** vs **Random**.
5. Організувати конфігурацію через **Hydra** (YAML).

---

## 2. Структура проєкту (розширена)

```
lab_1/
├── config/
│   ├── config.yaml                  ← базова Hydra конфігурація
│   ├── model/
│   │   ├── xgboost.yaml             ← простір пошуку для XGBoost
│   │   ├── random_forest.yaml       ← простір пошуку для RandomForest
│   │   └── logistic_regression.yaml ← простір пошуку для LogisticRegression
│   └── hpo/
│       ├── optuna.yaml (TPE)
│       ├── random.yaml
│       └── grid.yaml
├── src/
│   ├── prepare.py
│   ├── train.py
│   └── optimize.py                  ← HPO + Optuna + MLflow + Hydra (НОВИЙ)
├── models/
│   ├── best_model_xgboost_tpe.pkl
│   └── best_model_xgboost_random.pkl
├── dvc.yaml                         ← додано stage `optimize`
└── requirements.txt                 ← додано optuna, hydra-core, omegaconf
```

---

## 3. Датасет

- **Назва**: Healthcare Dataset — Stroke Prediction (Kaggle)
- **Ознаки**: вік, стать, артеріальна гіпертензія, хвороби серця, сімейний стан, тип роботи, тип проживання, середній рівень глюкози, ІМТ, статус куріння.
- **Цільова змінна**: `stroke` (0 / 1, дисбаланс класів ~95/5)
- **Розміри**: 4088 train / 1022 test (після split з ЛР1)

---

## 4. Простір пошуку гіперпараметрів (XGBoost)

| Параметр | Тип | Діапазон |
|---|---|---|
| `n_estimators` | int | [50, 300] |
| `max_depth` | int | [2, 10] |
| `learning_rate` | float (log) | [0.01, 0.3] |
| `subsample` | float | [0.6, 1.0] |
| `colsample_bytree` | float | [0.6, 1.0] |
| `min_child_weight` | int | [1, 10] |
| `gamma` | float | [0.0, 1.0] |
| `reg_alpha` | float | [0.0, 1.0] |
| `reg_lambda` | float | [0.5, 2.0] |

`scale_pos_weight` розраховується автоматично з даних для компенсації дисбалансу класів.  
Оцінка на **validation set** (20% від train, stratified split, seed=42).

---

## 5. Реалізація

### 5.1 `src/optimize.py`

- `objective(trial, cfg, X_tr, y_tr, X_val, y_val, sampler_name)` — Optuna objective. Будує XGBoost pipeline, виконує predict на val, повертає F1-score.  
- `run_study(cfg)` — запускає Optuna study, обгортає всі trials у **MLflow parent run**.
- `@hydra.main(config_path="../config", config_name="config")` — точка входу з Hydra.
- Підтримка XGBoost / RandomForest / LogisticRegression через `cfg.model.type`.

### 5.2 MLflow Nested Runs

```
📁 Study_xgboost_tpe_n20  (parent run)
   ├── 📄 trial_000
   ├── 📄 trial_001
   ├── ...
   └── 📄 trial_019
```

**Parent run логує:**
- `n_trials`, `direction`, `metric`, `seed`, `use_cv`
- `best_f1` (найкраща метрика серед trials)
- `best_trial_number`
- `final_test_f1`, `final_test_roc_auc` (після retraining на повному train)
- Артефакти: `best_params.json`, `hpo_config.yaml`, `best_model.pkl`

**Child run (кожний trial) логує:**
- Всі гіперпараметри (через `mlflow.log_params(trial.params)`)
- `f1` (метрика на val)
- `val_roc_auc`
- Теги: `sampler`, `model_type`, `trial_number`, `seed`

---

## 6. Запуск HPO

### TPE Sampler (за замовчуванням)
```bash
cd lab_1
source venv/bin/activate
python src/optimize.py
```

### Random Sampler
```bash
python src/optimize.py hpo.sampler=random
```

### Зміна моделі
```bash
python src/optimize.py model.type=random_forest
```

---

## 7. Результати

### 7.1 TPE Sampler — 20 trials

| Метрика | Значення |
|---|---|
| **Найкращий trial** | #6 |
| **Best F1 (val)** | **0.2689** |
| **Final Test F1** | **0.3056** |
| **Final Test ROC-AUC** | **0.8258** |

**Найкращі параметри (TPE):**

| Параметр | Значення |
|---|---|
| `n_estimators` | 200 |
| `max_depth` | 10 |
| `learning_rate` | 0.01351 |
| `subsample` | 0.6784 |
| `colsample_bytree` | 0.6181 |
| `min_child_weight` | 4 |
| `gamma` | 0.3887 |
| `reg_alpha` | 0.2713 |
| `reg_lambda` | 1.7431 |

---

### 7.2 Random Sampler — 20 trials

| Метрика | Значення |
|---|---|
| **Найкращий trial** | #6 |
| **Best F1 (val)** | **0.2689** |
| **Final Test F1** | **0.3056** |
| **Final Test ROC-AUC** | **0.8258** |

> **Примітка:** обидва sampler-и знайшли однаковий best trial через фіксований seed=42 та однаковий бюджет. При збільшенні n_trials або іншому seed очікується розходження в поведінці.

---

### 7.3 Порівняння TPE vs Random

| Критерій | TPE | Random |
|---|---|---|
| Best F1 (val) | 0.2689 | 0.2689 |
| Test F1 | 0.3056 | 0.3056 |
| Test ROC-AUC | 0.8258 | 0.8258 |
| Best trial | #6 | #6 |
| Стратегія | Байєсівська (sequential) | Випадкова |
| Ефективність | Кращий для великих просторів | Хороший baseline |

**Висновок:** При бюджеті 20 trials та фіксованому seed обидва sampler-и знаходять однакові конфігурації. TPE зазвичай ефективніший при більших бюджетах (50–100 trials), оскільки використовує попередні результати для формування наступних пропозицій, на відміну від Random, який не враховує попередніх результатів.

---

## 8. MLflow Tracking

- **Tracking URI**: `sqlite:///mlflow.db`
- **Експерименти**: `HPO_Lab3_xgboost_tpe`, `HPO_Lab3_xgboost_random`
- **Запуск UI**: `mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001`

У кожному експерименті:
- 1 parent run (Study)
- 20 child (nested) runs (Trials)

---

## 9. Висновки

1. **Optuna** значно спрощує HPO порівняно з ручним перебором — достатньо задати search space та objective.
2. **Hydra** дозволяє змінювати конфігурацію без редагування коду: `hpo.sampler=random`, `model.type=random_forest` — зручно для порівняльних експериментів.
3. **Вкладені MLflow runs** (parent = study, child = trial) забезпечують структуровану відтворюваність — кожен trial зберігає всі параметри та метрику.
4. **Дисбаланс класів** (stroke=1 лише ~5%) — основна проблема датасету. Незважаючи на `scale_pos_weight`, F1 залишається невисоким (~0.3). Для покращення варто використовувати cross-validation (`use_cv=true`) та більший бюджет (50+ trials).
5. **ROC-AUC = 0.826** свідчить про хорошу розпізнавальну здатність моделі, але F1 обмежений через дисбаланс.

---

## 10. Посилання

- [Optuna Documentation](https://optuna.readthedocs.io/)
- [Hydra Documentation](https://hydra.cc/docs/intro/)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [Kaggle: Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)
