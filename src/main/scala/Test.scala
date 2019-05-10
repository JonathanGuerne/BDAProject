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

  // TODO 1 THIBAUT Look for the rating of each movie. Create a new dataframe with the name of the movie and it's rating

  // TODO 2 JOHN Look for each film's director(s) information is store in title.crew. try to append directos name to dataframe

  // TODO 3 PEDRO split genres into n columns, n beeing the maximum number of genre a movie can have

}