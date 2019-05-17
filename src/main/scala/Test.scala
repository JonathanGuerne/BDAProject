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

  val df_new = df.filter("titleType == \"movie\"")
  val df3 = df_new.drop("titleType").drop("originalTitle").drop("endYear").na.drop()

  // TODO 1 THIBAUT Look for the rating of each movie. Create a new dataframe with the name of the movie and it's rating

  val all_films_ids = df3.collect().map(r => r.get(0))

  val df_ratings = spark.read.option("sep", "\t").option("header","true").csv("dataset/title.ratings.tsv")

  val df_label = df_ratings.drop("numVotes")

  println(df_ratings.count())

  val df_global=df3.join(df_ratings, df_ratings("tconst")===df3("tconst")).drop(df_ratings("tconst"))

  df_global.printSchema()

  // TODO 2 JOHN Look for each film's director(s) information is store in title.crew. try to append directos name to dataframe


  val df_crew = spark.read.option("sep", "\t").option("header", "true").csv("dataset/title.crew.tsv")
  df_crew.createOrReplaceTempView("crew")

  println(df_crew.count())

  // inner join to remove unwanted ids
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


  val parse_genres: Column => Column = x => { split(x, ",") }

  val df_genres = df3.withColumn("genres", parse_genres($"genres")).select(
    $"genres".getItem(0).as("genre1"),
    $"genres".getItem(1).as("genre2"),
    $"genres".getItem(2).as("genre3")
  ).na.fill(Map("genre2" -> 0)).na.fill(Map("genre3" -> 0))

  /*def countNull(colName:String) = sum(col(colName).isNull.cast("integer")).alias(colName)

  print(df_genres.count())

  df_genres.agg(countNull("genre4")).show()*/

  df_genres.show(30, false)
}