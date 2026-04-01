import joblib
import uvicorn
import pandas as pd
import json

from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel
from typing import List, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from preprocessing import QuantileBinner


# load model bundle
bundle = joblib.load("model_bundle.pkl")
mod = bundle["model"]
features = bundle["features"]

app = FastAPI()


class LoanData(BaseModel):
    uuid: Optional[str] = None
    default: Optional[int] = None
    account_amount_added_12_24m: Optional[int] = None
    account_days_in_dc_12_24m: Optional[int] = None
    account_days_in_rem_12_24m: Optional[int] = None
    account_days_in_term_12_24m: Optional[int] = None
    account_incoming_debt_vs_paid_0_24m: Optional[float] = None
    account_status: Optional[int] = None
    account_worst_status_0_3m: Optional[int] = None
    account_worst_status_12_24m: Optional[int] = None
    account_worst_status_3_6m: Optional[int] = None
    account_worst_status_6_12m: Optional[int] = None
    age: Optional[int] = None
    avg_payment_span_0_12m: Optional[float] = None
    avg_payment_span_0_3m: Optional[float] = None
    merchant_category: Optional[str] = None
    merchant_group: Optional[str] = None
    has_paid: Optional[bool] = None
    max_paid_inv_0_12m: Optional[int] = None
    max_paid_inv_0_24m: Optional[int] = None
    name_in_email: Optional[str] = None
    num_active_div_by_paid_inv_0_12m: Optional[float] = None
    num_active_inv: Optional[int] = None
    num_arch_dc_0_12m: Optional[int] = None
    num_arch_dc_12_24m: Optional[int] = None
    num_arch_ok_0_12m: Optional[int] = None
    num_arch_ok_12_24m: Optional[int] = None
    num_arch_rem_0_12m: Optional[int] = None
    num_arch_written_off_0_12m: Optional[int] = None
    num_arch_written_off_12_24m: Optional[int] = None
    num_unpaid_bills: Optional[int] = None
    status_last_archived_0_24m: Optional[int] = None
    status_2nd_last_archived_0_24m: Optional[int] = None
    status_3rd_last_archived_0_24m: Optional[int] = None
    status_max_archived_0_6_months: Optional[int] = None
    status_max_archived_0_12_months: Optional[int] = None
    status_max_archived_0_24_months: Optional[int] = None
    recovery_debt: Optional[int] = None
    sum_capital_paid_account_0_12m: Optional[int] = None
    sum_capital_paid_account_12_24m: Optional[int] = None
    sum_paid_inv_0_12m: Optional[int] = None
    time_hours: Optional[float] = None
    worst_status_active_inv: Optional[int] = None


# =========================
# HELPER FUNCTION
# =========================
def prepare_dataframe(data_list):
    if not data_list:
        raise ValueError("Empty input data")

    df = pd.DataFrame(data_list)

    # drop target
    df = df.drop(columns=["default"], errors="ignore")

    # fix types
    if "has_paid" in df.columns:
        df["has_paid"] = df["has_paid"].astype(float)

    # enforce schema
    df = df.reindex(columns=features)

    return df


# =========================
# MAIN ENDPOINT
# =========================
@app.post("/")
async def loans_request(data: List[LoanData]):
    try:
        df = prepare_dataframe([item.model_dump() for item in data])

        preds = mod.predict_proba(df)[:, 1]

        return [
            {"uuid": u, "pd": float(p)}
            for u, p in zip(df["uuid"], preds)
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# FILE ENDPOINT
# =========================
@app.post("/loans_file/")
async def loans_file(file: UploadFile):
    try:
        json_data = await file.read()
        loan_data = json.loads(json_data)

        if isinstance(loan_data, dict):
            loan_data = [loan_data]

        df = prepare_dataframe(loan_data)

        preds = mod.predict_proba(df)[:, 1]

        return [
            {"uuid": u, "pd": float(p)}
            for u, p in zip(df["uuid"], preds)
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)