# Orchestration with Apache Airflow

An orchestrator schedules pipeline tasks, runs them in the right order, retries failures and records what happened. Apache Airflow is one of the most widely used open-source orchestrators.

## DAGs and tasks

In Airflow, a pipeline is defined as a DAG, a directed acyclic graph of tasks written in Python. Each task does one unit of work, such as loading a file or running a transformation. Dependencies between tasks decide the execution order.

## Scheduling and backfills

A DAG has a schedule, for example every day at 06:00. Each run covers a data interval. A backfill re-runs the DAG for past intervals, which is only safe when tasks are idempotent and write results for their own interval.

## Retries and alerts

Tasks can be retried automatically with a delay, which handles temporary failures such as a busy API. When retries are exhausted, Airflow marks the task as failed and can send an alert. Good practice is to keep business logic outside the DAG file, in a normal Python package that can be unit tested, and let the DAG only call it.

## Sensors and external triggers

Sensors wait for a condition, such as a file appearing in object storage. Airflow also supports data-aware scheduling, where a DAG starts when an upstream dataset has been updated, instead of at a fixed time.
