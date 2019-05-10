import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._


object BDA_Project extends App {
  val spark = SparkSession.builder.config("spark.master", "local[2]").getOrCreate()

  val df = spark.read.option("sep", "\t").option("header", "true").csv("dataset/title.basics.tsv")
  df.createOrReplaceTempView("titles")

  println(df.columns.size)
  println(df.count())

  val df_new = df.filter("titleType == \"movie\"")
  val df3 = df_new.drop("titleType").drop("originalTitle").drop("endYear")

  println(df3.count())
  println(df3.columns.size)

}