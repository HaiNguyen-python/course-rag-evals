# Medallion Architecture

The medallion architecture organises a data lake into three layers: bronze, silver and gold. Each layer has a clear contract, which makes pipelines easier to debug and to rebuild.

## Bronze layer

The bronze layer stores raw data exactly as it arrived from the source system. Nothing is cleaned or removed. Files are usually written in an open columnar format such as Parquet and partitioned by load date. Keeping the raw data means you can always reprocess history when a bug is found in a later step.

## Silver layer

The silver layer holds cleaned and conformed data. Typical steps are casting types, removing duplicate records, standardising identifiers and joining reference data. Rows that fail validation should be moved to a quarantine table instead of being silently dropped, so that data quality problems stay visible.

## Gold layer

The gold layer contains business-ready tables, often modelled as a star schema with fact and dimension tables. Analysts, dashboards and machine learning features read from gold. Gold tables should be documented and tested because many people depend on them.

## Why use layers

Layers separate concerns: ingestion problems are fixed in bronze, cleaning rules live in silver, and business logic lives in gold. Because each step can be re-run from the layer before it, the whole pipeline becomes reproducible. A common rule is that a pipeline should be idempotent: running it twice for the same date gives the same result.
