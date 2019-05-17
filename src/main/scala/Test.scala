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

  println(df.count())

  val df_new = df.filter("titleType == \"movie\"")
  val df3 = df_new.drop("titleType").drop("originalTitle").drop("endYear").na.drop()

  println(df3.count())



  // TODO 1 THIBAUT Look for the rating of each movie. Create a new dataframe with the name of the movie and it's rating

  val all_films_ids = df3.collect().map(r => r.get(0))
  val df_ratings = spark.read.option("sep", "\t").option("header","true").csv("dataset/title.ratings.tsv")
  val df_label = df_ratings.drop("numVotes")

  val df_global=df3.join(df_ratings, df_ratings("tconst")===df3("tconst")).drop(df_ratings("tconst")).na.drop()
  df_global.printSchema()

  // TODO 2 JOHN Look for each film's director(s) information is store in title.crew. try to append directos name to dataframe


  val df_crew = spark.read.option("sep", "\t").option("header", "true").csv("dataset/title.crew.tsv")
  df_crew.createOrReplaceTempView("crew")

  // inner join to remove unwanted ids
  val df_crew_movie=df_global.join(df_crew, df_crew("tconst")===df_global("tconst")).drop(df_crew("tconst")).na.drop()

  df_crew_movie.printSchema()

  println(df_crew_movie.count())
  val nb_directors = df_crew_movie.collect().map(r => (r.get(8).toString.split(",").size))

  1 to 10 foreach { i =>
    println("#directors : " + i + " => " +nb_directors.foldLeft(0)((over_4, nb_dir) => if (nb_dir > i) over_4 + 1 else over_4))
  }

//  df_crew.map(row: Row => Int){
//    row.get(1).toString.split(",").size
//  })


  // for each movie, get list of all directors



  // TODO 3 PEDRO split genres into n columns, n beeing the maximum number of genre a movie can have

}