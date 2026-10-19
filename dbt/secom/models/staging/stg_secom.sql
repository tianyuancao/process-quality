{%- set kept = dbt_utils.get_column_values(ref('kept_sensors'), 'column_name') -%}

select
    WAFER_ID as wafer_id,
    MEASURED_AT as measured_at, 
    IS_FAIL as is_fail
    {%- for c in kept %},
    coalesce({{ c }}, median({{ c }}) over ()) as {{ c | lower }}
    {%- endfor %}
from {{ source('raw', 'secom_raw') }}