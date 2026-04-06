from pathlib import Path

import numpy as np
import polars as pl

from tactical.models import (
    ChangepointConfig,
    GMMConfig,
    GMMModel,
    HMMConfig,
    HMMModel,
    PreprocessingPipeline,
    detect_changepoints,
    select_gmm_k,
)


def test_full_model_pipeline(tmp_path: Path) -> None:
    """Test the full model pipeline end-to-end."""
    # 1. Create a synthetic polars DataFrame mimicking the feature pipeline output
    np.random.seed(42)
    n_samples = 200

    # Create 3 synthetic clusters
    cluster_1 = np.random.normal(loc=-5.0, scale=1.0, size=(70, 20))
    cluster_2 = np.random.normal(loc=0.0, scale=1.0, size=(65, 20))
    cluster_3 = np.random.normal(loc=5.0, scale=1.0, size=(65, 20))

    features = np.vstack([cluster_1, cluster_2, cluster_3])

    match_ids = ["match_1"] * 100 + ["match_2"] * 50 + ["match_3"] * 50
    timestamps = np.arange(n_samples, dtype=float) * 0.1

    data = {
        "match_id": match_ids,
        "timestamp": timestamps,
    }

    feature_cols = []
    for i in range(10):
        data[f"t1_feature_{i}"] = features[:, i]
        data[f"t2_feature_{i}"] = features[:, i + 10]
        feature_cols.extend([f"t1_feature_{i}", f"t2_feature_{i}"])

    df = pl.DataFrame(data)

    # 2. Create a PreprocessingPipeline
    pipeline = PreprocessingPipeline(pca_variance_threshold=0.95)

    # 3. Fit-transform the DataFrame
    x = pipeline.fit_transform(df)

    # 4. Verify shape and no nulls
    assert x.shape[0] == n_samples
    assert x.shape[1] > 0
    assert not np.isnan(x).any()

    # 5. Run select_gmm_k
    selection_config = GMMConfig(k_min=2, k_max=5, random_state=42)
    selection_result = select_gmm_k(x, config=selection_config)

    # 6. Verify ModelSelectionResult has scores
    assert len(selection_result.k_values) == 4
    assert list(selection_result.k_values) == [2, 3, 4, 5]

    # 7. Verify best_k_bic is 3
    assert selection_result.best_k_bic == 3

    # 8. Create GMMModel
    gmm_config = GMMConfig(k_max=3, random_state=42)
    gmm = GMMModel(config=gmm_config)
    gmm.fit(x)

    labels = gmm.predict(x)
    probs = gmm.predict_proba(x)

    # 9. Verify labels
    assert labels.shape == (n_samples,)
    assert set(labels).issubset({0, 1, 2})

    # 10. Verify predict_proba rows sum to 1
    assert np.allclose(probs.sum(axis=1), 1.0)

    # 11. Create HMMModel
    hmm_config = HMMConfig(n_states=3, random_state=42)
    hmm = HMMModel(config=hmm_config)

    # 12. Compute sequence lengths
    # All rows retained in this simple example
    seq_lengths = [100, 50, 50]

    # 13. Fit HMM with GMM initialization
    hmm.fit_from_gmm(x, sequence_lengths=seq_lengths, gmm_model=gmm)

    # 14. Predict states. Verify same shape as GMM output.
    hmm_labels = hmm.predict(x, sequence_lengths=seq_lengths)
    assert hmm_labels.shape == (n_samples,)
    assert set(hmm_labels).issubset({0, 1, 2})

    # 15. Verify transition matrix rows sum to 1
    trans_mat = hmm.transition_matrix
    assert trans_mat is not None
    assert trans_mat.shape == (3, 3)
    assert np.allclose(trans_mat.sum(axis=1), 1.0)

    # 16. Run detect_changepoints
    match_1_probs = probs[:100]
    match_1_times = timestamps[:100]
    cp_config = ChangepointConfig()

    cp_result = detect_changepoints(
        signal=match_1_probs,
        match_minutes=match_1_times,
        match_id="match_1",
        config=cp_config,
    )

    # 17. Verify ChangepointResult
    assert cp_result.match_id == "match_1"
    assert isinstance(cp_result.changepoints, tuple)
    assert len(cp_result.changepoints) > 0

    # 18. Verify all models can be saved and loaded from tmp_path
    gmm_path = tmp_path / "gmm.pkl"
    gmm.save(gmm_path)

    loaded_gmm = GMMModel.load(gmm_path)
    loaded_labels = loaded_gmm.predict(x)
    assert np.array_equal(labels, loaded_labels)

    hmm_path = tmp_path / "hmm.pkl"
    hmm.save(hmm_path)

    loaded_hmm = HMMModel.load(hmm_path)
    loaded_hmm_labels = loaded_hmm.predict(x, sequence_lengths=seq_lengths)
    assert np.array_equal(hmm_labels, loaded_hmm_labels)

    pipeline_path = tmp_path / "pipeline.pkl"
    pipeline.save(pipeline_path)

    loaded_pipeline = PreprocessingPipeline.load(pipeline_path)
    x_loaded = loaded_pipeline.transform(df)
    assert np.allclose(x, x_loaded)
