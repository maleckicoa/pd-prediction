DROP VIEW IF EXISTS test_feat_roll_hit;

CREATE VIEW test_feat_roll_hit AS
WITH rolling AS (
    SELECT
        e.loan_event_idx,
        e.loan_uuid,
        e.observed_at_utc,
        e.feature_name,
        e.feature_type,
        e.bin_index,
        e.category_value,
        e.hit,
        (SUM(e.hit) OVER (
            PARTITION BY e.feature_name, e.feature_type, e.bin_index, e.category_value
            ORDER BY e.loan_event_idx ASC, e.id ASC
            ROWS BETWEEN 4999 PRECEDING AND CURRENT ROW
        ))::DOUBLE PRECISION AS rolling_hit_count,
        COUNT(*) OVER (
            PARTITION BY e.feature_name, e.feature_type, e.bin_index, e.category_value
            ORDER BY e.loan_event_idx ASC, e.id ASC
            ROWS BETWEEN 4999 PRECEDING AND CURRENT ROW
        ) AS rolling_window_size
    FROM test_feat_hit e
)
SELECT
    loan_event_idx,
    loan_uuid,
    observed_at_utc,
    feature_name,
    feature_type,
    bin_index,
    category_value,
    hit,
    rolling_hit_count,
    rolling_window_size,
    CASE
        WHEN rolling_window_size > 0 THEN rolling_hit_count / rolling_window_size
        ELSE 0
    END AS observed_pct
FROM rolling;
