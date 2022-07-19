# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* drought_detection/*.py

black:
	@black scripts/* drought_detection/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr drought_detection-*.dist-info
	@rm -fr drought_detection.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''

# ----------------------------------
#      UPLOAD PACKAGE TO PYPI
# ----------------------------------
PYPI_USERNAME=<AUTHOR>
build:
	@python setup.py sdist bdist_wheel

pypi_test:
	@twine upload -r testpypi dist/* -u $(PYPI_USERNAME)

pypi:
	@twine upload dist/* -u $(PYPI_USERNAME)


# ================================================================================

# ----------------------------------
#      GCP SETUP
# ----------------------------------
# project id
PROJECT_ID=drought-detection

# bucket name
BUCKET_NAME=wagon-data-batch913-drought_detection

# choose your region from https://cloud.google.com/storage/docs/locations#available_locations
REGION=europe-west4



JOB_NAME=tmp_pat_hel_model_save_$(shell date +'%Y%m%d_%H%M%S')

PACKAGE_NAME=drought_detection
FILENAME=trainer

PYTHON_VERSION=3.7
FRAMEWORK=scikit-learn
RUNTIME_VERSION=2.8


set_project:
	@gcloud config set project ${PROJECT_ID}

create_bucket:
	@gsutil mb -l ${REGION} -p ${PROJECT_ID} gs://${BUCKET_NAME}



# ----------------------------------
#      UPLOAD DATA TO GCP
# ----------------------------------
# path to the file to upload to GCP
LOCAL_PATH="./drought_detection/data/tmp.csv"


# bucket directory in which to store the uploaded file
BUCKET_FOLDER=data

# name for the uploaded file inside of the bucket (we choose not to rename the file that we upload)
BUCKET_FILE_NAME=$(shell basename ${LOCAL_PATH})
BUCKET_TRAINING_FOLDER=SavedModel

# "shortcut" to upload file to GCP using 'make' in terminal (in terminal: make upload_data)
upload_data:
    # @gsutil cp train_1k.csv gs://wagon-ml-my-bucket-name/data/train_1k.csv
	@gsutil cp ${LOCAL_PATH} gs://${BUCKET_NAME}/${BUCKET_FOLDER}

#opens webpage
streamlit:
	-@streamlit run app.py

gcp_submit_training:
	gcloud ai-platform jobs submit training ${JOB_NAME} \
		--job-dir gs://${BUCKET_NAME}/${BUCKET_TRAINING_FOLDER} \
		--package-path ${PACKAGE_NAME} \
		--module-name ${PACKAGE_NAME}.${FILENAME} \
		--python-version=${PYTHON_VERSION} \
		--runtime-version=${RUNTIME_VERSION} \
		--region ${REGION} \
		--scale-tier BASIC_TPU \
		--stream-logs
