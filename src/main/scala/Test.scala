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
  import spark.implicits._

  val df = spark.read.option("sep", "\t").option("header", "true").option("nullValue", "\\N").csv("dataset/title.basics.tsv")
  df.createOrReplaceTempView("titles")

  println(df.columns.size)
  println(df.count())

  val df_new = df.filter("titleType == \"movie\"")
  val df3 = df_new.drop("titleType").drop("originalTitle").drop("endYear").na.drop()

  println(df3.count())
  println(df3.columns.size)

  // TODO 1 THIBAUT Look for the rating of each movie. Create a new dataframe with the name of the movie and it's rating

  // TODO 2 JOHN Look for each film's director(s) information is store in title.crew. try to append directos name to dataframe


  val df_crew = spark.read.option("sep", "\t").option("header", "true").csv("dataset/title.crew.tsv")
  df_crew.createOrReplaceTempView("crew")

  println(df_crew.count())

  // inner join to remove unwanted ids
  val all_films_ids = df3.collect().map(r => r.get(0))
  val df_crew_movie = df_crew.filter($"tconst".isin(all_films_ids: _*))

  println(df_crew_movie.count())

  //println(df_crew.first())
  val nb_directors = df_crew_movie.collect().map(r => (r.get(1).toString.split(",").size))


  1 to 10 foreach { i =>
    println("#directors : " + i + " => " +nb_directors.foldLeft(0)((over_4, nb_dir) => if (nb_dir > i) over_4 + 1 else over_4))
  }

//  df_crew.map(row: Row => Int){
//    row.get(1).toString.split(",").size
//  })


  // for each movie, get list of all directors



  // TODO 3 PEDRO split genres into n columns, n beeing the maximum number of genre a movie can have

}