DROP VIEW IF EXISTS test_feat_roll_psi;

CREATE VIEW test_feat_roll_psi AS
WITH joined AS (
    SELECT
        r.loan_event_idx,
        r.loan_uuid,
        r.observed_at_utc,
        r.feature_name,
        r.feature_type,
        r.bin_index,
        r.category_value,
        r.rolling_window_size,
        r.observed_pct,
        f.observation_pct AS expected_pct
    FROM test_feat_roll_hit r
    INNER JOIN train_feat_dist f
        ON r.feature_name = f.feature_name
        AND r.feature_type = f.feature_type
        AND r.bin_index IS NOT DISTINCT FROM f.bin_index
        AND r.category_value IS NOT DISTINCT FROM f.category_value
),
bucket_psi AS (
    SELECT
        j.loan_event_idx,
        j.loan_uuid,
        j.observed_at_utc,
        j.feature_name,
        j.rolling_window_size,
        (
            (GREATEST(j.observed_pct, 1e-6) - GREATEST(j.expected_pct, 1e-6))
            * LN(GREATEST(j.observed_pct, 1e-6) / GREATEST(j.expected_pct, 1e-6))
        ) AS bucket_psi
    FROM joined j
)
SELECT
    loan_event_idx,
    loan_uuid,
    observed_at_utc,
    feature_name,
    MAX(rolling_window_size) AS rolling_window_size,
    SUM(bucket_psi) AS rolling_psi
FROM bucket_psi
GROUP BY
    loan_event_idx,
    loan_uuid,
    observed_at_utc,
    feature_name;
