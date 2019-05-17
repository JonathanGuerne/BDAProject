import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._


object DataPreparation extends App {
  val spark = SparkSession.builder.config("spark.master", "local[2]").getOrCreate()

  import spark.implicits._

  // TODO pedro word 2 vect, (column vector concatenation ?)
  val features = spark.read.parquet("features_extract.parquet").orderBy($"startYear")
  features.createOrReplaceTempView("features")

  features.show(50, false)

  // TODO pedro produce an X and Y dataset, output to csv
}