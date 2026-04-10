import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
from itertools import product as itertools_product
from catboost import CatBoostRanker, Pool
from sqlalchemy.ext.asyncio import AsyncSession
import lightgbm as lgb

from .feature_extractor import feature_extractor
from .training_data_generator import training_data_generator
from ..db import queries

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MLPRankerNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.GELU(), nn.BatchNorm1d(h), nn.Dropout(0.15)])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PairwiseRankingLoss(nn.Module):
    def forward(self, scores, labels):
        pos_mask = labels > 0
        neg_mask = labels == 0
        if not pos_mask.any() or not neg_mask.any():
            return torch.tensor(0.0, device=scores.device, requires_grad=True)
        pos_scores = scores[pos_mask]
        neg_scores = scores[neg_mask]
        n = min(len(pos_scores), len(neg_scores))
        return -torch.log(torch.sigmoid(pos_scores[:n] - neg_scores[:n]) + 1e-8).mean()


def compute_ndcg_at_k(y_true, y_pred, groups, k=10):
    unique_groups = sorted(set(groups))
    ndcgs = []
    for g in unique_groups:
        mask = [i for i, gi in enumerate(groups) if gi == g]
        if len(mask) < 2:
            continue
        true = np.array([y_true[i] for i in mask])
        pred = np.array([y_pred[i] for i in mask])
        order = np.argsort(-pred)[:k]
        dcg = sum((2 ** true[order[i]] - 1) / np.log2(i + 2) for i in range(min(k, len(order))))
        ideal_order = np.argsort(-true)[:k]
        idcg = sum((2 ** true[ideal_order[i]] - 1) / np.log2(i + 2) for i in range(min(k, len(ideal_order))))
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return np.mean(ndcgs) if ndcgs else 0.0


class EnsembleRankerService:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.catboost_model: Optional[CatBoostRanker] = None
        self.mlp_model: Optional[MLPRankerNet] = None
        self.mlp_mean: Optional[np.ndarray] = None
        self.mlp_std: Optional[np.ndarray] = None
        self.lgbm_model: Optional[lgb.Booster] = None

        self.weights = {"catboost": 0.4, "mlp": 0.35, "lightgbm": 0.25}
        self.model_version: Optional[str] = None
        self.model_metadata: Optional[Dict] = None

        self._load_models()

    @property
    def is_ready(self) -> bool:
        return any([self.catboost_model, self.mlp_model, self.lgbm_model])

    def _load_models(self):
        catboost_files = list(self.models_dir.glob("catboost_ranker_*.cbm"))
        if catboost_files:
            latest = max(catboost_files, key=lambda p: p.stat().st_mtime)
            try:
                self.catboost_model = CatBoostRanker()
                self.catboost_model.load_model(str(latest))
                print(f"✓ CatBoost loaded: {latest.name}")
            except Exception as e:
                print(f"CatBoost load error: {e}")
                self.catboost_model = None

        mlp_files = list(self.models_dir.glob("mlp_ranker_*.pt"))
        if mlp_files:
            latest = max(mlp_files, key=lambda p: p.stat().st_mtime)
            try:
                state = torch.load(str(latest), map_location=DEVICE, weights_only=True)
                input_dim = len(feature_extractor.feature_names)
                self.mlp_model = MLPRankerNet(input_dim).to(DEVICE)
                self.mlp_model.load_state_dict(state)
                self.mlp_model.eval()

                scaler_path = latest.parent / latest.name.replace('.pt', '_scaler.pkl')
                if scaler_path.exists():
                    with open(scaler_path, 'rb') as f:
                        scaler_data = pickle.load(f)
                        self.mlp_mean = scaler_data['mean']
                        self.mlp_std = scaler_data['std']
                print(f"✓ MLP loaded: {latest.name}")
            except Exception as e:
                print(f"MLP load error: {e}")
                self.mlp_model = None

        lgbm_files = list(self.models_dir.glob("lgbm_ranker_*.txt"))
        if lgbm_files:
            latest = max(lgbm_files, key=lambda p: p.stat().st_mtime)
            try:
                self.lgbm_model = lgb.Booster(model_file=str(latest))
                print(f"✓ LightGBM loaded: {latest.name}")
            except Exception as e:
                print(f"LightGBM load error: {e}")
                self.lgbm_model = None

        weights_path = self.models_dir / "ensemble_weights.json"
        if weights_path.exists():
            with open(weights_path) as f:
                self.weights = json.load(f)
            print(f"✓ Ensemble weights: {self.weights}")

        metadata_files = list(self.models_dir.glob("ensemble_*_metadata.json"))
        if metadata_files:
            latest = max(metadata_files, key=lambda p: p.stat().st_mtime)
            with open(latest) as f:
                self.model_metadata = json.load(f)
            self.model_version = self.model_metadata.get("version")

    async def train_model(
        self,
        session: AsyncSession,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        min_feedback_count: int = 5,
        negative_sampling_ratio: int = 3,
    ) -> Dict:
        print("\n" + "=" * 80)
        print("ОБУЧЕНИЕ ENSEMBLE RANKER")
        print("=" * 80)

        X_all, y_all, groups = await training_data_generator.generate_training_data(
            session=session,
            min_feedback_count=min_feedback_count,
            negative_sampling_ratio=negative_sampling_ratio,
        )

        if len(X_all) < 100:
            raise ValueError(
                f"Недостаточно данных: {len(X_all)} примеров. Нужно минимум 100."
            )

        from sklearn.model_selection import train_test_split
        unique_groups = list(set(groups))
        train_groups, val_groups = train_test_split(unique_groups, test_size=0.2, random_state=42)

        train_mask = [g in set(train_groups) for g in groups]
        val_mask = [g in set(val_groups) for g in groups]

        X_train = X_all[train_mask].reset_index(drop=True)
        y_train = y_all[train_mask].reset_index(drop=True)
        g_train = [groups[i] for i, m in enumerate(train_mask) if m]

        X_val = X_all[val_mask].reset_index(drop=True)
        y_val = y_all[val_mask].reset_index(drop=True)
        g_val = [groups[i] for i, m in enumerate(val_mask) if m]

        train_order = np.argsort(g_train)
        X_train = X_train.iloc[train_order].reset_index(drop=True)
        y_train = y_train.iloc[train_order].reset_index(drop=True)
        g_train = [g_train[i] for i in train_order]

        val_order = np.argsort(g_val)
        X_val = X_val.iloc[val_order].reset_index(drop=True)
        y_val = y_val.iloc[val_order].reset_index(drop=True)
        g_val = [g_val[i] for i in val_order]

        print(f"\nTrain: {len(X_train)} примеров, {len(train_groups)} query")
        print(f"Val:   {len(X_val)} примеров, {len(val_groups)} query")

        metrics = {}

        print("\n" + "-" * 80)
        print("[1/3] CatBoost (YetiRank)")
        print("-" * 80)
        try:
            self.catboost_model = CatBoostRanker(
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                loss_function="YetiRank",
                custom_metric=["NDCG:top=10"],
                random_seed=42,
                verbose=50,
                use_best_model=True,
                eval_metric="NDCG:top=10",
            )
            train_pool = Pool(data=X_train, label=y_train, group_id=g_train)
            val_pool = Pool(data=X_val, label=y_val, group_id=g_val)
            self.catboost_model.fit(train_pool, eval_set=val_pool, plot=False)

            cb_val_pred = self.catboost_model.predict(val_pool)
            cb_ndcg = compute_ndcg_at_k(y_val.values, cb_val_pred, g_val, k=10)
            metrics["catboost"] = {"ndcg@10": float(cb_ndcg)}
            print(f"  CatBoost val NDCG@10: {cb_ndcg:.4f}")
        except Exception as e:
            print(f"  CatBoost error: {e}")
            self.catboost_model = None
            cb_val_pred = None

        print("\n" + "-" * 80)
        print("[2/3] MLP Ranker (BPR)")
        print("-" * 80)
        try:
            X_train_np = X_train.values.astype(np.float32)
            X_val_np = X_val.values.astype(np.float32)

            self.mlp_mean = X_train_np.mean(axis=0)
            self.mlp_std = X_train_np.std(axis=0) + 1e-8
            X_train_normed = (X_train_np - self.mlp_mean) / self.mlp_std
            X_val_normed = (X_val_np - self.mlp_mean) / self.mlp_std

            input_dim = X_train_np.shape[1]
            self.mlp_model = MLPRankerNet(input_dim).to(DEVICE)

            loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_train_normed, dtype=torch.float32),
                    torch.tensor(y_train.values, dtype=torch.float32),
                ),
                batch_size=512, shuffle=True,
            )

            optimizer = optim.Adam(self.mlp_model.parameters(), lr=1e-3, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=80)
            criterion = PairwiseRankingLoss()

            self.mlp_model.train()
            for epoch in range(80):
                for xb, yb in loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    optimizer.zero_grad()
                    loss = criterion(self.mlp_model(xb), yb)
                    if loss.requires_grad:
                        loss.backward()
                        optimizer.step()
                scheduler.step()

            self.mlp_model.eval()
            with torch.no_grad():
                mlp_val_pred = self.mlp_model(
                    torch.tensor(X_val_normed, dtype=torch.float32).to(DEVICE)
                ).cpu().numpy()

            mlp_ndcg = compute_ndcg_at_k(y_val.values, mlp_val_pred, g_val, k=10)
            metrics["mlp"] = {"ndcg@10": float(mlp_ndcg)}
            print(f"  MLP val NDCG@10: {mlp_ndcg:.4f}")
        except Exception as e:
            print(f"  MLP error: {e}")
            import traceback
            traceback.print_exc()
            self.mlp_model = None
            mlp_val_pred = None

        print("\n" + "-" * 80)
        print("[3/3] LightGBM (LambdaRank)")
        print("-" * 80)
        try:
            train_group_sizes = pd.Series(g_train).groupby(g_train).size().values
            val_group_sizes = pd.Series(g_val).groupby(g_val).size().values

            lgb_train = lgb.Dataset(X_train, label=y_train, group=train_group_sizes)
            lgb_val = lgb.Dataset(X_val, label=y_val, group=val_group_sizes, reference=lgb_train)

            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "ndcg_eval_at": [5, 10],
                "num_leaves": 31,
                "learning_rate": learning_rate,
                "verbose": -1,
                "seed": 42,
            }

            self.lgbm_model = lgb.train(
                params, lgb_train,
                num_boost_round=iterations,
                valid_sets=[lgb_val],
                callbacks=[lgb.log_evaluation(50)],
            )

            lgb_val_pred = self.lgbm_model.predict(X_val)
            lgb_ndcg = compute_ndcg_at_k(y_val.values, lgb_val_pred, g_val, k=10)
            metrics["lightgbm"] = {"ndcg@10": float(lgb_ndcg)}
            print(f"  LightGBM val NDCG@10: {lgb_ndcg:.4f}")
        except Exception as e:
            print(f"  LightGBM error: {e}")
            import traceback
            traceback.print_exc()
            self.lgbm_model = None
            lgb_val_pred = None

        print("\n" + "-" * 80)
        print("ОПТИМИЗАЦИЯ ВЕСОВ АНСАМБЛЯ")
        print("-" * 80)

        available_preds = {}
        if cb_val_pred is not None:
            available_preds["catboost"] = self._normalize_scores(cb_val_pred)
        if mlp_val_pred is not None:
            available_preds["mlp"] = self._normalize_scores(mlp_val_pred)
        if lgb_val_pred is not None:
            available_preds["lightgbm"] = self._normalize_scores(lgb_val_pred)

        if len(available_preds) >= 2:
            best_ndcg = 0.0
            best_weights = {}
            names = list(available_preds.keys())
            preds_list = [available_preds[n] for n in names]

            steps = np.arange(0.0, 1.05, 0.1)
            for combo in itertools_product(steps, repeat=len(names)):
                if abs(sum(combo) - 1.0) > 0.01:
                    continue
                ensemble_pred = sum(w * p for w, p in zip(combo, preds_list))
                ndcg = compute_ndcg_at_k(y_val.values, ensemble_pred, g_val, k=10)
                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_weights = {n: round(float(w), 2) for n, w in zip(names, combo)}

            self.weights = best_weights
            metrics["ensemble"] = {"ndcg@10": float(best_ndcg), "weights": best_weights}
            print(f"  Best weights: {best_weights}")
            print(f"  Ensemble NDCG@10: {best_ndcg:.4f}")
        elif len(available_preds) == 1:
            name = list(available_preds.keys())[0]
            self.weights = {name: 1.0}
            metrics["ensemble"] = metrics.get(name, {})
            print(f"  Only {name} available, using single model")

        print("\n" + "-" * 80)
        print("СОХРАНЕНИЕ МОДЕЛЕЙ")
        print("-" * 80)

        self.model_version = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.catboost_model:
            path = self.models_dir / f"catboost_ranker_{self.model_version}.cbm"
            self.catboost_model.save_model(str(path))
            print(f"  ✓ CatBoost: {path.name}")

        if self.mlp_model:
            path = self.models_dir / f"mlp_ranker_{self.model_version}.pt"
            torch.save(self.mlp_model.state_dict(), str(path))
            scaler_path = self.models_dir / f"mlp_ranker_{self.model_version}_scaler.pkl"
            with open(scaler_path, 'wb') as f:
                pickle.dump({'mean': self.mlp_mean, 'std': self.mlp_std}, f)
            print(f"  ✓ MLP: {path.name}")

        if self.lgbm_model:
            path = self.models_dir / f"lgbm_ranker_{self.model_version}.txt"
            self.lgbm_model.save_model(str(path))
            print(f"  ✓ LightGBM: {path.name}")

        weights_path = self.models_dir / "ensemble_weights.json"
        with open(weights_path, 'w') as f:
            json.dump(self.weights, f, indent=2)

        self.model_metadata = {
            "version": self.model_version,
            "trained_at": datetime.now().isoformat(),
            "train_samples": len(X_train),
            "val_samples": len(X_val),
            "metrics": metrics,
            "weights": self.weights,
            "models": {
                "catboost": self.catboost_model is not None,
                "mlp": self.mlp_model is not None,
                "lightgbm": self.lgbm_model is not None,
            },
        }

        metadata_path = self.models_dir / f"ensemble_{self.model_version}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)

        print(f"\n✓ Ensemble metadata: {metadata_path.name}")
        print("\n" + "=" * 80)
        print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
        print("=" * 80)

        return self.model_metadata

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        min_s, max_s = scores.min(), scores.max()
        if max_s - min_s < 1e-8:
            return np.full_like(scores, 0.5)
        return (scores - min_s) / (max_s - min_s)

    async def rank_candidates(
        self,
        main_product: Dict,
        candidates: List[Dict],
        session: AsyncSession,
        cart_products: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        if not self.is_ready or not candidates:
            return candidates

        X = []
        valid_candidates = []

        main_id = main_product["id"]
        main_embedding = await queries.get_product_embedding(session, main_id)
        candidate_ids = [c["id"] for c in candidates]
        embeddings_map = await queries.get_embeddings_map(session, candidate_ids)
        pair_stats = await queries.get_pair_feedback_stats(session, main_id, candidate_ids)
        copurchase_stats = await queries.get_copurchase_stats(session, main_id, candidate_ids)

        for candidate in candidates:
            cand_id = candidate["id"]
            features = await feature_extractor.extract_features(
                main_product=main_product,
                candidate_product=candidate,
                main_embedding=main_embedding,
                candidate_embedding=embeddings_map.get(cand_id),
                pair_feedback=pair_stats.get(cand_id, {"positive": 0, "negative": 0}),
                scenario_feedback={"positive": 0, "negative": 0},
                copurchase_count=copurchase_stats.get(cand_id, 0),
                cart_products=cart_products,
                session=session,
            )
            if features:
                X.append(list(features.values()))
                valid_candidates.append(candidate)

        if not X:
            return candidates

        X_df = pd.DataFrame(X, columns=feature_extractor.feature_names)
        X_np = X_df.values.astype(np.float32)

        all_scores = {}

        if self.catboost_model:
            raw = self.catboost_model.predict(X_df)
            all_scores["catboost"] = self._normalize_scores(raw)

        if self.mlp_model and self.mlp_mean is not None:
            X_normed = (X_np - self.mlp_mean) / self.mlp_std
            self.mlp_model.eval()
            with torch.no_grad():
                raw = self.mlp_model(torch.tensor(X_normed, dtype=torch.float32).to(DEVICE)).cpu().numpy()
            all_scores["mlp"] = self._normalize_scores(raw)

        if self.lgbm_model:
            raw = self.lgbm_model.predict(X_df)
            all_scores["lightgbm"] = self._normalize_scores(raw)

        if not all_scores:
            return candidates

        total_weight = sum(self.weights.get(name, 0) for name in all_scores)
        if total_weight < 1e-8:
            total_weight = 1.0

        ensemble_scores = np.zeros(len(valid_candidates))
        for name, scores in all_scores.items():
            w = self.weights.get(name, 0) / total_weight
            ensemble_scores += w * scores

        ensemble_scores = 0.5 + ensemble_scores * 0.5

        for candidate, score in zip(valid_candidates, ensemble_scores):
            candidate["ml_score"] = float(score)

        valid_candidates.sort(key=lambda x: x["ml_score"], reverse=True)
        return valid_candidates

    def get_model_info(self) -> Dict:
        if not self.is_ready:
            return {
                "status": "no_model",
                "message": "Ни одна модель не обучена. Используется формульный скоринг.",
            }

        return {
            "status": "ready",
            "type": "ensemble",
            "version": self.model_version,
            "weights": self.weights,
            "models": {
                "catboost": {"status": "ready" if self.catboost_model else "not_trained"},
                "mlp": {"status": "ready" if self.mlp_model else "not_trained"},
                "lightgbm": {"status": "ready" if self.lgbm_model else "not_trained"},
            },
            "metadata": self.model_metadata,
            "feature_count": len(feature_extractor.feature_names),
            "features": feature_extractor.feature_names,
        }


ensemble_ranker = EnsembleRankerService()