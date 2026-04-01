FROM python:3.11
WORKDIR /usr/src/app
COPY PD_model_train.py main.py preprocessing.py model_bundle.pkl dataset.csv requirements.txt ./
COPY loan1.json loan5.json loan20.json curl_file.txt curl_request.txt ./
RUN pip install -r requirements.txt
CMD ["/bin/sh","-c","python main.py"]