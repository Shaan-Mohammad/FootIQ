package com.footiq

import org.apache.spark.sql.SparkSession

object FootIQApp {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("FootIQ")
      .master("local[*]")
      .getOrCreate()

    try {

      val eventsPath = "data/raw/events_jsonl/*.jsonl"
      val bronzePath = "data/bronze/events"

      // Read normalized JSON Lines files
      val eventsDF = spark.read
        .option("multiLine", "false")
        .json(eventsPath)

      println("================================")
      println("       FOOTIQ DATA INGESTION")
      println("================================")

      println("\nDataset Schema:")
      eventsDF.printSchema()

      println("\nFirst 5 Football Events:")
      eventsDF.show(5, false)

      // Write data into Bronze layer
      eventsDF.write
        .mode("overwrite")
        .parquet(bronzePath)

      println("\nBronze data successfully saved!")

      // Verify the saved Parquet data
      val bronzeDF = spark.read.parquet(bronzePath)

      println("\nTotal Bronze Records: " + bronzeDF.count())

    } finally {
      spark.stop()
    }
  }
}
