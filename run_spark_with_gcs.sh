#!/usr/bin/env bash

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-digital-layout-475700-c2}"
CONNECTOR_JAR="${CONNECTOR_JAR:-$PWD/third_party/gcs-connector-hadoop3-2.2.30-shaded.jar}"
ADC_JSON="${GOOGLE_APPLICATION_CREDENTIALS:-$PWD/.gcloud-config/application_default_credentials.json}"

if [[ ! -f "$CONNECTOR_JAR" ]]; then
  echo "Missing GCS connector jar: $CONNECTOR_JAR" >&2
  echo "Download gcs-connector-hadoop3-2.2.30-shaded.jar into third_party/ first." >&2
  exit 1
fi

if [[ ! -f "$ADC_JSON" ]]; then
  echo "Missing Application Default Credentials file: $ADC_JSON" >&2
  echo "Run 'gcloud auth application-default login' first." >&2
  exit 1
fi

exec env GOOGLE_APPLICATION_CREDENTIALS="$ADC_JSON" spark-submit \
  --jars "$CONNECTOR_JAR" \
  --conf "spark.hadoop.fs.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem" \
  --conf "spark.hadoop.fs.AbstractFileSystem.gs.impl=com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS" \
  --conf "spark.hadoop.fs.gs.project.id=$PROJECT_ID" \
  --conf "spark.hadoop.fs.gs.auth.type=APPLICATION_DEFAULT" \
  "$@"
