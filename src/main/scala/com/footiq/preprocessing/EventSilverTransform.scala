package com.footiq.preprocessing

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

object EventSilverTransform {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("FootIQ Silver Transformation")
      .master("local[*]")
      .getOrCreate()

    try {

      val bronzePath = "data/bronze/events"
      val silverPath = "data/silver/events"

      // Read Bronze dataset
      val bronzeDF = spark.read.parquet(bronzePath)

      // Standardize column names
      val silverDF = bronzeDF.select(
        col("id").as("event_id"),
        col("eventId").as("event_type_id"),
        col("eventName").as("event_name"),
        col("subEventId").as("sub_event_id"),
        col("subEventName").as("sub_event_name"),
        col("matchId").as("match_id"),
        col("playerId").as("player_id"),
        col("teamId").as("team_id"),
        col("matchPeriod").as("period"),
        col("eventSec").as("event_time_sec"),
        col("positions"),
        col("tags")
      )

      // Standardize text fields
      val standardizedDF = silverDF
        .withColumn("event_name", trim(col("event_name")))
        .withColumn(
          "sub_event_name",
          when(
            col("sub_event_name").isNull ||
            (trim(col("sub_event_name")) === ""),
            lit(null).cast("string")
          ).otherwise(trim(col("sub_event_name")))
        )

      // Extract position information
      val positionedDF = standardizedDF
        .withColumn(
          "position_count",
          coalesce(size(col("positions")), lit(0))
        )
        .withColumn(
          "start_x",
          col("positions").getItem(0).getField("x")
        )
        .withColumn(
          "start_y",
          col("positions").getItem(0).getField("y")
        )
        .withColumn(
          "end_x",
          col("positions").getItem(1).getField("x")
        )
        .withColumn(
          "end_y",
          col("positions").getItem(1).getField("y")
        )

      // Add data-quality flags
      val finalDF = positionedDF
        .withColumn(
          "has_player",
          col("player_id").isNotNull &&
          (col("player_id") =!= lit(0L))
        )
        .withColumn(
          "has_complete_positions",
          col("position_count") === 2
        )
        .withColumn(
          "has_valid_coordinates",
          when(
            col("position_count") === 0,
            lit(false)
          ).otherwise(
            not(expr("""
              exists(
                positions,
                p -> p.x < 0 OR p.x > 100
                  OR p.y < 0 OR p.y > 100
              )
            """))
          )
        )

      println("================================")
      println("    FOOTIQ SILVER TRANSFORMATION")
      println("================================")

      println("\nSilver Dataset Schema:")
      finalDF.printSchema()

      println("\nSample Silver Records:")
      finalDF.show(5, false)

      // Save the transformed data
      finalDF.write
        .mode("overwrite")
        .parquet(silverPath)

      println("\nSilver data successfully saved!")

      // Verify the saved data
      val savedDF = spark.read.parquet(silverPath)

      println(
        "\nTotal Silver Records: " + savedDF.count()
      )

    } finally {
      spark.stop()
    }
  }
}
