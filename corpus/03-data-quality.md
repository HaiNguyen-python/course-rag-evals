# Data Quality and Testing

Bad data is often more harmful than missing data, because people trust it. Data quality checks catch problems before they reach dashboards or models.

## Dimensions of data quality

Common dimensions are completeness (are values missing?), uniqueness (are there duplicate keys?), validity (are values in the allowed range or format?), consistency (do related tables agree?) and freshness (did the data arrive on time?).

## Tests in practice

Tools such as dbt and Great Expectations let teams write tests as code. Typical tests check that a primary key is unique and not null, that foreign keys exist in the parent table, and that a column only contains accepted values. Tests run on every pipeline run, and a failing test should stop bad data from flowing to the next layer.

## Data contracts

A data contract is an agreement between the producer and consumers of a dataset about its schema, meaning and quality. Contracts make breaking changes visible early, because a producer who wants to rename a column must update the contract first.

## Monitoring

Beyond tests, teams monitor row counts, null rates and freshness over time. A sudden drop in daily row count is often the first sign that an upstream system has changed. Alerts should go to the team that owns the pipeline, with a short runbook that explains what to check first.
