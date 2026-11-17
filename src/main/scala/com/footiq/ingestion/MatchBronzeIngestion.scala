package com.footiq.ingestion

import org.apache.spark.sql.SparkSession

object MatchBronzeIngestion {

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("FootIQ Match Bronze Ingestion")
      .master("local[*]")
      .getOrCreate()

    try {

      val matchesPath = "data/raw/matches/*.json"
      val bronzePath = "data/bronze/matches"

      // Read original match JSON files
      val matchesDF = spark.read
        .option("multiLine", "true")
        .json(matchesPath)

      println("Match Dataset Schema:")
      matchesDF.printSchema()

      println("Sample Match Records:")
      matchesDF.show(5, false)

      // Count source records
      val totalMatches = matchesDF.count()
      println(s"Total Match Records: $totalMatches")

      // Save matches in Parquet format
      matchesDF.write
        .mode("overwrite")
        .parquet(bronzePath)

      // Verify saved records
      val savedDF = spark.read.parquet(bronzePath)

      println("Total Bronze Match Records: " + savedDF.count())
      println("Match Bronze Ingestion Completed!")

    } finally {
      spark.stop()
    }
  }
}
