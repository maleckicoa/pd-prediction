CREATE TABLE IF NOT EXISTS train_feat_dist (
    id BIGSERIAL PRIMARY KEY,
    feature_name TEXT NOT NULL,
    feature_type TEXT NOT NULL CHECK (feature_type IN ('numerical', 'categorical')),
    bin_index INTEGER,
    category_value TEXT,
    bin_left DOUBLE PRECISION,
    bin_right DOUBLE PRECISION,
    observation_pct DOUBLE PRECISION NOT NULL CHECK (observation_pct >= 0 AND observation_pct <= 1),
    created_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CHECK (
        (
            feature_type = 'numerical'
            AND bin_index IS NOT NULL
            AND (
                (
                    bin_index = 0
                    AND category_value = 'MISSING'
                    AND bin_left IS NULL
                    AND bin_right IS NULL
                )
                OR (
                    bin_index > 0
                    AND category_value IS NULL
                    AND bin_left IS NOT NULL
                    AND (
                        bin_right IS NOT NULL
                        OR (bin_right IS NULL AND bin_left IS NOT NULL)
                    )
                )
            )
        )
        OR (
            feature_type = 'categorical'
            AND bin_index IS NULL
            AND category_value IS NOT NULL
            AND bin_left IS NULL
            AND bin_right IS NULL
        )
    )
);

CREATE INDEX IF NOT EXISTS train_feat_dist_feature_name_idx
    ON train_feat_dist (feature_name);
