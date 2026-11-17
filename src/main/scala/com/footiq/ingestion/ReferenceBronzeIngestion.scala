package com.footiq.ingestion

import org.apache.spark.sql.{DataFrame, SparkSession}

object ReferenceBronzeIngestion {

  def saveDataset(
      name: String,
      df: DataFrame,
      outputPath: String
  ): Unit = {

    println(s"\n========== $name ==========")

    df.printSchema()
    df.show(3, false)

    val sourceCount = df.count()
    println(s"Source Records: $sourceCount")

    df.write
      .mode("overwrite")
      .parquet(outputPath)

    val savedDF = sparkSession.read.parquet(outputPath)
    val savedCount = savedDF.count()

    println(s"Saved Records: $savedCount")
  }

  private var sparkSession: SparkSession = _

  def main(args: Array[String]): Unit = {

    sparkSession = SparkSession.builder()
      .appName("FootIQ Reference Bronze Ingestion")
      .master("local[*]")
      .getOrCreate()

    try {

      val referencePath = "data/reference/"
      val bronzePath = "data/bronze/reference/"

      // Read JSON reference datasets
      val competitionsDF = sparkSession.read
        .option("multiLine", "true")
        .json(referencePath + "competitions.json")

      val playersDF = sparkSession.read
        .option("multiLine", "true")
        .json(referencePath + "players.json")

      val teamsDF = sparkSession.read
        .option("multiLine", "true")
        .json(referencePath + "teams.json")

      // Read CSV reference datasets
      val eventNamesDF = sparkSession.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(referencePath + "eventid2name.csv")

      val tagNamesDF = sparkSession.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(referencePath + "tags2name.csv")

      // Save all five datasets as Parquet
      saveDataset(
        "Competitions",
        competitionsDF,
        bronzePath + "competitions"
      )

      saveDataset(
        "Players",
        playersDF,
        bronzePath + "players"
      )

      saveDataset(
        "Teams",
        teamsDF,
        bronzePath + "teams"
      )

      saveDataset(
        "Event Names",
        eventNamesDF,
        bronzePath + "event_names"
      )

      saveDataset(
        "Tag Names",
        tagNamesDF,
        bronzePath + "tag_names"
      )

      println("\nAll reference datasets ingested successfully!")

    } finally {
      sparkSession.stop()
    }
  }
}
