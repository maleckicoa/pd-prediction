CREATE SEQUENCE IF NOT EXISTS test_feat_hit_loan_event_idx_seq START 1;

CREATE TABLE IF NOT EXISTS test_feat_hit (
    id BIGSERIAL PRIMARY KEY,
    loan_event_idx BIGINT NOT NULL,
    loan_uuid UUID NOT NULL,
    observed_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    feature_name TEXT NOT NULL,
    feature_type TEXT NOT NULL CHECK (feature_type IN ('numerical', 'categorical')),
    bin_index INTEGER,
    category_value TEXT,
    hit SMALLINT NOT NULL CHECK (hit IN (0, 1))
);

CREATE INDEX IF NOT EXISTS test_feat_hit_event_idx_idx
    ON test_feat_hit (loan_event_idx ASC);

CREATE INDEX IF NOT EXISTS test_feat_hit_feature_bucket_idx
    ON test_feat_hit (
        feature_name ASC,
        feature_type ASC,
        bin_index ASC,
        category_value ASC
    );
