import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._


object DataExtraction extends App {

  // TODO pedro, john do stats

  def countNull(colName: String) = sum(col(colName).isNull.cast("integer")).alias(colName)

  def countNotNull(colName: String) = sum(col(colName).isNotNull.cast("integer")).alias(colName)

  val spark = SparkSession.builder.config("spark.master", "local[2]").getOrCreate()

  import spark.implicits._

  val df_basics = spark.read
    .option("sep", "\t")
    .option("header", "true")
    .option("nullValue", "\\N")
    .csv("dataset/title.basics.tsv")

  df_basics.createOrReplaceTempView("titles")

  //println("Number of rows of basic.tsv " + df_basics.count())

  val df_basics_filtered = df_basics.filter("titleType == \"movie\"")
  val df_step0 = df_basics_filtered.drop("titleType")
    .drop("originalTitle")
    .drop("endYear")


  // DONE 1 THIBAUT Look for the rating of each movie.
  // Create a new dataframe with the name of the movie and it's rating

  val df_ratings = spark.read
    .option("sep", "\t")
    .option("header", "true")
    .option("nullValue", "\\N")
    .csv("dataset/title.ratings.tsv")

  //println("Number of rows of ratings.tsv " + df_ratings.count())

  val df_step1 = df_step0.join(df_ratings,
    df_ratings("tconst") === df_step0("tconst"))
    .drop(df_ratings("tconst"))
    .drop("numVotes")

  //df_step1.printSchema()


  // DONE 2 JOHN Look for each film's director(s) information is store in title.crew.
  // Try to append directors name to dataframe

  val df_crew = spark.read
    .option("sep", "\t")
    .option("header", "true")
    .option("nullValue", "\\N")
    .csv("dataset/title.crew.tsv")

  df_crew.createOrReplaceTempView("crew")

  //println("Number of rows of crew.tsv " + df_crew.count())

  // Inner join to remove unwanted ids
  val df_crew_movie = df_step1.join(df_crew,
    df_crew("tconst") === df_step1("tconst"))
    .drop(df_crew("tconst"))
    .na.drop()

  // df_crew_movie.agg(countNull("genres")).show()

  //println("Number of rows of join with ratings, crew and basic.tsv " + df_crew_movie.count())

  val parse_string: Column => Column = x => {
    split(x, ",")
  }

  var df_directors = df_crew_movie
    .withColumn("directors", parse_string($"directors"))
    .select(
      $"tconst",
      $"directors".getItem(0).as("directors1"),
      $"directors".getItem(1).as("directors2"),
      $"directors".getItem(2).as("directors3"),
      $"directors".getItem(3).as("directors4")
    )
    .na.fill(Map("directors2" -> 0))
    .na.fill(Map("directors3" -> 0))
    .na.fill(Map("directors4" -> 0))

  df_directors = df_directors.drop("directors")

  df_directors = df_directors.
    withColumn("directors", concat_ws(",",$"directors1", $"directors2", $"directors3", $"directors4"))

  df_directors = df_directors.drop("directors1", "directors2", "directors3", "directors4")

  var df_writers = df_crew_movie
    .withColumn("writers", parse_string($"writers"))
    .select(
      $"tconst",
      $"writers".getItem(0).as("writers1"),
      $"writers".getItem(1).as("writers2"),
      $"writers".getItem(2).as("writers3"),
      $"writers".getItem(3).as("writers4"),
      $"writers".getItem(3).as("writers5")
    )
    .na.fill(Map("writers2" -> 0))
    .na.fill(Map("writers3" -> 0))
    .na.fill(Map("writers4" -> 0))
    .na.fill(Map("writers5" -> 0))

  df_writers = df_writers.drop("writers")

  df_writers = df_writers.
    withColumn("writers", concat_ws(",",$"writers1", $"writers2", $"writers3", $"writers4", $"writers5"))

  df_writers = df_writers.drop("writers1", "writers2", "writers3", "writers4", "writers5")

  val df_step2_1 = df_step1.join(df_directors,
    df_directors("tconst") === df_step1("tconst")).drop(df_directors("tconst"))

  val df_step2 = df_step2_1.join(df_writers,
    df_writers("tconst") === df_step2_1("tconst")).drop(df_writers("tconst"))

  df_step2.show(50, false)

  df_step2.printSchema()


  //DONE 3 PEDRO split genres into n columns, n beeing the maximum number of genre a movie can have

  /*
  val df_genres = df_crew_movie.withColumn("genres", parse_string($"genres")).select(
    $"tconst",
    $"genres".getItem(0).as("genre1"),
    $"genres".getItem(1).as("genre2"),
    $"genres".getItem(2).as("genre3")
  ).na.fill(Map("genre2" -> 0))
    .na.fill(Map("genre3" -> 0))

  /*
  df_genres.agg(countNotNull("genre1")).show()
  df_genres.agg(countNotNull("genre2")).show()
  df_genres.agg(countNotNull("genre3")).show()
   */

  val df_step3 = df_step2.join(df_genres,
    df_genres("tconst") === df_step2("tconst")).drop(df_genres("tconst")).drop(df_step2("genres"))

  val df_step3 = df_step2.join(df_crew_movie,
    df_crew_movie("tconst") === df_step2("tconst")).drop(df_crew_movie("tconst")).drop(df_step2("genres"))
  */

  // TODO thibaut export to csv

  df_step2.orderBy("startYear").show(50, false)
  df_step2.write.mode(SaveMode.Overwrite).parquet("features_extract.parquet")
}