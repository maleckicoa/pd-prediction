CREATE TABLE train_loans (
    uuid UUID NOT NULL,
    "default" INTEGER,
    account_amount_added_12_24m INTEGER,
    account_days_in_dc_12_24m INTEGER,
    account_days_in_rem_12_24m INTEGER,
    account_days_in_term_12_24m INTEGER,
    account_incoming_debt_vs_paid_0_24m DOUBLE PRECISION,
    account_status INTEGER,
    account_worst_status_0_3m INTEGER,
    account_worst_status_12_24m INTEGER,
    account_worst_status_3_6m INTEGER,
    account_worst_status_6_12m INTEGER,
    age INTEGER,
    avg_payment_span_0_12m DOUBLE PRECISION,
    avg_payment_span_0_3m DOUBLE PRECISION,
    merchant_category TEXT,
    merchant_group TEXT,
    has_paid BOOLEAN,
    max_paid_inv_0_12m INTEGER,
    max_paid_inv_0_24m INTEGER,
    name_in_email TEXT,
    num_active_div_by_paid_inv_0_12m DOUBLE PRECISION,
    num_active_inv INTEGER,
    num_arch_dc_0_12m INTEGER,
    num_arch_dc_12_24m INTEGER,
    num_arch_ok_0_12m INTEGER,
    num_arch_ok_12_24m INTEGER,
    num_arch_rem_0_12m INTEGER,
    num_arch_written_off_0_12m INTEGER,
    num_arch_written_off_12_24m INTEGER,
    num_unpaid_bills INTEGER,
    status_last_archived_0_24m INTEGER,
    status_2nd_last_archived_0_24m INTEGER,
    status_3rd_last_archived_0_24m INTEGER,
    status_max_archived_0_6_months INTEGER,
    status_max_archived_0_12_months INTEGER,
    status_max_archived_0_24_months INTEGER,
    recovery_debt INTEGER,
    sum_capital_paid_account_0_12m INTEGER,
    sum_capital_paid_account_12_24m INTEGER,
    sum_paid_inv_0_12m INTEGER,
    time_hours DOUBLE PRECISION,
    worst_status_active_inv INTEGER
);

CREATE INDEX train_loans_uuid_idx ON train_loans (uuid);

CREATE TABLE test_loans (
    uuid UUID NOT NULL,
    "default" INTEGER,
    account_amount_added_12_24m INTEGER,
    account_days_in_dc_12_24m INTEGER,
    account_days_in_rem_12_24m INTEGER,
    account_days_in_term_12_24m INTEGER,
    account_incoming_debt_vs_paid_0_24m DOUBLE PRECISION,
    account_status INTEGER,
    account_worst_status_0_3m INTEGER,
    account_worst_status_12_24m INTEGER,
    account_worst_status_3_6m INTEGER,
    account_worst_status_6_12m INTEGER,
    age INTEGER,
    avg_payment_span_0_12m DOUBLE PRECISION,
    avg_payment_span_0_3m DOUBLE PRECISION,
    merchant_category TEXT,
    merchant_group TEXT,
    has_paid BOOLEAN,
    max_paid_inv_0_12m INTEGER,
    max_paid_inv_0_24m INTEGER,
    name_in_email TEXT,
    num_active_div_by_paid_inv_0_12m DOUBLE PRECISION,
    num_active_inv INTEGER,
    num_arch_dc_0_12m INTEGER,
    num_arch_dc_12_24m INTEGER,
    num_arch_ok_0_12m INTEGER,
    num_arch_ok_12_24m INTEGER,
    num_arch_rem_0_12m INTEGER,
    num_arch_written_off_0_12m INTEGER,
    num_arch_written_off_12_24m INTEGER,
    num_unpaid_bills INTEGER,
    status_last_archived_0_24m INTEGER,
    status_2nd_last_archived_0_24m INTEGER,
    status_3rd_last_archived_0_24m INTEGER,
    status_max_archived_0_6_months INTEGER,
    status_max_archived_0_12_months INTEGER,
    status_max_archived_0_24_months INTEGER,
    recovery_debt INTEGER,
    sum_capital_paid_account_0_12m INTEGER,
    sum_capital_paid_account_12_24m INTEGER,
    sum_paid_inv_0_12m INTEGER,
    time_hours DOUBLE PRECISION,
    worst_status_active_inv INTEGER
);

CREATE INDEX test_loans_uuid_idx ON test_loans (uuid);

CREATE TABLE loan_ids (
    id BIGSERIAL PRIMARY KEY,
    uuid UUID NOT NULL UNIQUE
);

CREATE TABLE test_defaults (
    id BIGINT NOT NULL REFERENCES loan_ids(id),
    uuid UUID NOT NULL,
    pd DOUBLE PRECISION NOT NULL,
    threshold_applied DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    loan_default INTEGER,
    model_name TEXT NOT NULL DEFAULT 'unknown',
    predicted_at_utc TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX test_defaults_uuid_idx ON test_defaults (uuid);
CREATE INDEX test_defaults_id_idx ON test_defaults (id);
CREATE INDEX test_defaults_predicted_at_idx ON test_defaults (predicted_at_utc DESC);

