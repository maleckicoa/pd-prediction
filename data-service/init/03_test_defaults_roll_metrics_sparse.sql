-- Sparse rolling KPIs: same windowed metrics as test_defaults_roll_metrics, one row per id divisible by 30.
DROP VIEW IF EXISTS test_defaults_roll_metrics_sparse;

CREATE VIEW test_defaults_roll_metrics_sparse AS
SELECT
    id,
    model_name,
    predicted_at_utc,
    rolling_precision,
    rolling_recall
FROM test_defaults_roll_metrics
WHERE MOD(id, 30) = 0;
