
###################### PD Predict - MLflow targets
mlflow-up:
	docker compose up -d mlflow

mlflow-up-build:
	docker compose up -d --build mlflow

mlflow-down:
	docker compose stop mlflow

mlflow-delete:
	docker compose rm -fsv mlflow


###################### PD Predict - Data-service targets
data-service-up-build:
	docker compose up -d --build postgres

data-service-up:
	docker compose up -d postgres

data-service-down:
	docker compose stop postgres


###################### PD Predict - Train-service targets
train-service-build:
	docker compose --profile train build --no-cache train

train-service-lr1:
	docker compose --profile train run --rm train /app/train-service/src/lr1/lr1_model_train.py

train-service-lr2:
	docker compose --profile train run --rm train /app/train-service/src/lr2/lr2_model_train.py

train-service-nn1:
	docker compose --profile train run --rm train /app/train-service/src/nn1/nn1_model_train.py

train-service-xgb1:
	docker compose --profile train run --rm train /app/train-service/src/xgb1/xgb1_model_train.py

train-service-xgb2:
	docker compose --profile train run --rm train /app/train-service/src/xgb2/xgb2_model_train.py


###################### PD Predict - -service targets
fetch-service-up:
	docker compose --profile fetch up -d fetch

fetch-service-up-build:
	docker compose --profile fetch up -d --build fetch

fetch-service-down:
	docker compose --profile fetch stop fetch


###################### PD Predict - Predict-service targets
predict-service-up-build:
	docker compose --profile predict up -d --build predict

predict-service-up:
	docker compose --profile predict up -d predict

predict-service-down:
	docker compose --profile predict stop predict

predict-service-xgb1-test-csv:
	docker compose --profile predict exec -T predict python /app/predict-service/src/xgb1/xgb1_predict_test_loans_to_csv.py

