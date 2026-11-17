# FootIQ
### Distributed Football Analytics and Tactical Intelligence Platform

FootIQ is an academic cluster-computing project that uses Apache Spark and Apache Kafka to process and analyse football match-event data from Wyscout.

## Objectives
- Process football event data using Apache Spark.
- Implement Bronze, Silver, and Gold data-processing layers.
- Analyse player passing networks using graph analytics.
- Build a shot-outcome classification model using Spark MLlib.
- Apply K-means clustering to player statistics.
- Demonstrate event streaming using Kafka and Spark Structured Streaming.
- Benchmark Spark partition configurations.
- Visualise analytical results through an interactive dashboard.

## Technology Stack
- Python, Scala, SQL
- Apache Spark and PySpark
- Apache Spark GraphX and MLlib
- Apache Kafka
- Streamlit and Plotly
- Maven

## Dataset
Wyscout Soccer Match Event Dataset.

- [Dataset collection](https://figshare.com/collections/Soccer_match_event_dataset/4415000)
- [Research paper](https://doi.org/10.1038/s41597-019-0247-7)

## Architecture
1. Bronze – raw data ingestion.
2. Silver – data cleaning and validation.
3. Gold – analytical datasets.
4. Analytics – graph analysis, shot prediction, and player clustering.
5. Streaming – Kafka event replay and Spark Structured Streaming.
6. Dashboard – interactive exploration of results.

## Results

| Dataset / Output | Records |
|---|---:|
| Silver events | 3,251,294 |
| Match summaries | 1,941 |
| Team statistics | 142 |
| Player statistics | 3,610 |
| Shot analytics | 43,078 |
| Graph vertices | 3,598 |
| Graph pass edges | 60,738 |

Random Forest achieved the highest test F1-score (approximately 0.352) among the three evaluated shot-outcome models.

## Dashboard

Run the dashboard from the project directory:

```bash
streamlit run dashboard.py
```

The dashboard requires the generated Gold datasets, which are not included in this repository.

## Repository Structure
- `src/` – Scala and Spark source code.
- `scripts/` – data preparation utilities.
- `dashboard.py` – Streamlit dashboard.
- `results/` – model comparison and benchmark outputs.
- `pom.xml` – Maven configuration.

## Execution Environment
Developed and tested locally on macOS. The project demonstrates Spark and Kafka workflows in local mode; it does not claim a production multi-node cluster deployment.

## Limitations
- Generated datasets and trained models are excluded from GitHub.
- Some analytics and streaming experiments were executed interactively and are not yet packaged as standalone source scripts.
- Streaming uses historical event replay rather than a live football feed.

## Attribution
Dataset: Wyscout Soccer Match Event Dataset, distributed under CC BY 4.0. Please refer to the original dataset collection and research paper for attribution details.


