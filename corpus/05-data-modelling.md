# Dimensional Data Modelling

Dimensional modelling organises analytical data into facts and dimensions. It was popularised by Ralph Kimball and is still the most common way to model a data warehouse.

## Facts and dimensions

A fact table stores measurements of a business process, such as sales amounts or student activity counts. Each row has a grain, for example one row per student per day. Dimension tables describe the context of facts: who, what, where and when. Examples are dim_student, dim_course and dim_date.

## Star schema

In a star schema, the fact table sits in the centre and joins directly to each dimension. Star schemas are easy for analysts to query and work well with BI tools.

## Slowly changing dimensions

Attributes in dimensions change over time, for example a student moves to another city. A type 1 slowly changing dimension overwrites the old value. A type 2 slowly changing dimension keeps history by adding a new row with valid_from and valid_to dates, so that facts can be joined to the attributes that were true at that time.

## Choosing the grain

The grain is the most important design decision. A finer grain gives more flexibility but larger tables. A good rule is to choose the finest grain that the source data supports and that the business may need, and to aggregate later in reporting tables.
