-- Rolling metrics view (window size = 5000).
-- Window frame naturally shrinks for early rows where fewer than 5000
-- observations exist in a model partition.
DROP VIEW IF EXISTS test_defaults_roll_metrics;

CREATE VIEW test_defaults_roll_metrics AS
WITH ordered AS (
    SELECT
        d.id,
        d.uuid,
        d.pd,
        d.threshold_applied,
        d.loan_default,
        d.model_name,
        d.predicted_at_utc,
        CASE WHEN d.loan_default IN (0, 1) THEN 1 ELSE 0 END AS label_known,
        CASE WHEN d.pd > d.threshold_applied THEN 1 ELSE 0 END AS pred_default
    FROM test_defaults d
),
metrics AS (
    SELECT
        o.*,
        SUM(CASE WHEN o.label_known = 1 THEN 1 ELSE 0 END)
            OVER (
                PARTITION BY o.model_name
                ORDER BY o.predicted_at_utc ASC, o.id ASC
                ROWS BETWEEN 4999 PRECEDING AND CURRENT ROW
            ) AS rolling_known_rows,
        SUM(CASE WHEN o.pred_default = 1 THEN 1 ELSE 0 END)
            OVER (
                PARTITION BY o.model_name
                ORDER BY o.predicted_at_utc ASC, o.id ASC
                ROWS BETWEEN 4999 PRECEDING AND CURRENT ROW
            ) AS rolling_pred_positive,
        SUM(CASE WHEN o.loan_default = 1 THEN 1 ELSE 0 END)
            OVER (
                PARTITION BY o.model_name
                ORDER BY o.predicted_at_utc ASC, o.id ASC
                ROWS BETWEEN 4999 PRECEDING AND CURRENT ROW
            ) AS rolling_actual_positive,
        SUM(CASE WHEN o.pred_default = 1 AND o.loan_default = 1 THEN 1 ELSE 0 END)
            OVER (
                PARTITION BY o.model_name
                ORDER BY o.predicted_at_utc ASC, o.id ASC
                ROWS BETWEEN 4999 PRECEDING AND CURRENT ROW
            ) AS rolling_true_positive
    FROM ordered o
)
SELECT
    m.id,
    m.uuid,
    m.pd,
    m.threshold_applied,
    m.loan_default,
    m.model_name,
    m.predicted_at_utc,
    m.rolling_known_rows,
    m.rolling_pred_positive,
    m.rolling_actual_positive,
    m.rolling_true_positive,
    -- Use zero fallback until metric denominators become positive.
    CASE
        WHEN m.rolling_pred_positive > 0
            THEN m.rolling_true_positive::DOUBLE PRECISION / m.rolling_pred_positive
        ELSE 0
    END AS rolling_precision,
    CASE
        WHEN m.rolling_actual_positive > 0
            THEN m.rolling_true_positive::DOUBLE PRECISION / m.rolling_actual_positive
        ELSE 0
    END AS rolling_recall
FROM metrics m;
