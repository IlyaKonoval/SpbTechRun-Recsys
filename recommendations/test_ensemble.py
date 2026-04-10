import asyncio
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "app"))

from app.db.database import async_session
from app.ml.ensemble_ranker import ensemble_ranker


async def main():
    print("=" * 80)
    print("ОБУЧЕНИЕ ENSEMBLE RANKER")
    print("=" * 80)

    async with async_session() as session:
        metadata = await ensemble_ranker.train_model(
            session=session,
            iterations=500,
            learning_rate=0.05,
            depth=6,
            min_feedback_count=1,
        )

    print("\nИТОГО:")
    for model, info in metadata.get("metrics", {}).items():
        if isinstance(info, dict) and "ndcg@10" in info:
            print(f"  {model}: NDCG@10 = {info['ndcg@10']:.4f}")
        if isinstance(info, dict) and "weights" in info:
            print(f"  weights: {info['weights']}")

    info = ensemble_ranker.get_model_info()
    print(f"\nМодели: {info['weights']}")


if __name__ == "__main__":
    asyncio.run(main())
