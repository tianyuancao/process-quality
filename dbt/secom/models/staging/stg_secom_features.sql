{{ config(materialized='table') }}

{#
  Feature-layer view of the raw sensors.

  Deliberately does NOT impute. Nulls are preserved so that imputation can
  happen inside the scikit-learn Pipeline, fitted on training folds only.
  stg_secom (with median imputation) exists for warehouse-side consumers
  such as dashboards, where cross-fold leakage is not a concern.
#}

{%- set rows = run_query("select column_name from " ~ ref('kept_sensors')) -%}
{%- if execute -%}
  {%- set kept = rows.columns[0].values() -%}
{%- else -%}
  {%- set kept = [] -%}
{%- endif -%}

select
    WAFER_ID     as wafer_id,
    MEASURED_AT  as measured_at,
    IS_FAIL      as is_fail
    {%- for c in kept %},
    {{ c }} as {{ c | lower }}
    {%- endfor %}

from {{ source('raw', 'secom_raw') }}
order by measured_at