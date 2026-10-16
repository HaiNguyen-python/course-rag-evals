# Batch and Streaming Processing

Data pipelines either process data in batches on a schedule or continuously as a stream. The right choice depends on how fresh the data must be and how much complexity the team can support.

## Batch processing

A batch job processes a bounded set of data, for example all orders from yesterday. Batch jobs are simpler to build, test and re-run. They are a good fit for daily reports, model training and most analytics. Their main limitation is latency: results are only as fresh as the last run.

## Stream processing

A streaming job processes events continuously as they arrive, usually from a message broker such as Apache Kafka. Streaming is needed when decisions must be made within seconds or minutes, such as fraud detection or live transport monitoring.

## Event time and watermarks

Streaming systems distinguish event time, when something happened, from processing time, when the system saw it. Events can arrive late because of network delays. A watermark tells the engine how long to wait for late events before a time window is considered complete. In Apache Spark, data that arrives within the watermark delay is guaranteed to be counted, and the watermark also allows old state to be cleaned up.

## Delivery guarantees

At-most-once delivery can lose messages. At-least-once delivery never loses messages but can create duplicates. Exactly-once results are usually achieved by combining replayable sources, checkpoints that store offsets, and idempotent sinks that can safely receive the same write twice.
