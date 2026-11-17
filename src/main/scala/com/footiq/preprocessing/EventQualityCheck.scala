package com.footiq.preprocessing

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object EventQualityCheck {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("FootIQ Data Quality Check")
      .master("local[*]")
      .getOrCreate()

    try {

      // Read the existing Bronze dataset
      val eventsDF = spark.read
        .parquet("data/bronze/events")

      println("================================")
      println("    FOOTIQ DATA QUALITY CHECK")
      println("================================")

      val qualitySummary = eventsDF.agg(

        count(lit(1)).as("total_events"),

        sum(when(
          col("playerId").isNull || col("playerId") === 0L,
          1L
        ).otherwise(0L)).as("missing_or_unknown_player"),

        sum(when(
          col("subEventName").isNull ||
          trim(col("subEventName")) === "",
          1L
        ).otherwise(0L)).as("missing_subevent_name"),

        sum(when(
          col("tags").isNull || size(col("tags")) === 0,
          1L
        ).otherwise(0L)).as("empty_tags"),

        sum(when(
          col("positions").isNull || size(col("positions")) =!= 2,
          1L
        ).otherwise(0L)).as("incomplete_positions"),

        sum(when(
          expr("""
            exists(
              positions,
              p -> p.x < 0 OR p.x > 100
                OR p.y < 0 OR p.y > 100
            )
          """),
          1L
        ).otherwise(0L)).as("events_with_invalid_coordinates")

      )

      qualitySummary.show(false)

    } finally {
      spark.stop()
    }
  }
}
