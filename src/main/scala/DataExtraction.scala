import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._


object DataExtraction extends App {

  // Function to count null and not null rows
  def countNull(colName: String) = sum(col(colName).isNull.cast("integer")).alias(colName)
  def countNotNull(colName: String) = sum(col(colName).isNotNull.cast("integer")).alias(colName)

  // Parsing function
  def parse_string(x : String, char: Char): Array[String] = x.split(char)

  // Spark session
  val spark = SparkSession.builder.config("spark.master", "local[2]").getOrCreate()

  import spark.implicits._

  /*************************
  *    Basic Dataframe     *
  **************************/

  var df_basics = spark.read
    .option("sep", "\t")
    .option("header", "true")
    .option("nullValue", "\\N")
    .csv("dataset/title.basics.tsv")
  df_basics.createOrReplaceTempView("titles")

  // Filter the dataframe to only keep movies
  df_basics = df_basics.filter("titleType == \"movie\"")

  // Dropout useless columns
  df_basics = df_basics.drop("titleType", "originalTitle", "endYear")

  /*************************
  *    Ratings Dataframe   *
  **************************/

  val df_ratings = spark.read
    .option("sep", "\t")
    .option("header", "true")
    .option("nullValue", "\\N")
    .csv("dataset/title.ratings.tsv")

  // Join basics dataframe and ratings dataframe
  val df_basics_ratings = df_basics
    .join(df_ratings, df_ratings("tconst") === df_basics("tconst"))
    .drop(df_ratings("tconst"))
    .drop("numVotes")

  /*************************
  *    Crew Dataframe      *
  **************************/

  val df_crew = spark.read
    .option("sep", "\t")
    .option("header", "true")
    .option("nullValue", "\\N")
    .csv("dataset/title.crew.tsv")
  df_crew.createOrReplaceTempView("crew")

  // Inner join to remove unwanted ids
  var df_basics_crew_ratings = df_basics_ratings
    .join(df_crew, df_crew("tconst") === df_basics_ratings("tconst"))
    .drop(df_crew("tconst"))
    .na.drop()

  // Split the string in the columns title/genres/directors/writers to separate each words
  df_basics_crew_ratings = df_basics_crew_ratings
    .map {
      case Row(id: String, title: String, isAdult: String, startYear: String, duration: String, genres: String,
      ratings: String, directors: String, writers: String) =>
        (id,
          parse_string(title.replaceAll("[^A-Za-z0-9 ]", ""), ' ').length,
          parse_string(title.replaceAll("[^A-Za-z0-9 ]", ""), ' '),
          isAdult.toInt, startYear, duration,
          parse_string(genres, ',').length,
          parse_string(genres, ','),
          ratings.toDouble,
          parse_string(directors, ',').length,
          parse_string(directors, ',').slice(0, 4),
          parse_string(writers, ',').length,
          parse_string(writers, ',').slice(0, 5))
    }.toDF("id", "nb_words_title","title", "isAdult", "startYear", "duration",
    "nb_genres","genres", "ratings", "nb_directors", "directors", "nb_writers","writers")

  // Order the final dataframe and show the final extraction and then save it to the corresponding parquet
  df_basics_crew_ratings.orderBy("startYear").show(50, false)
  df_basics_crew_ratings.write.mode(SaveMode.Overwrite).parquet("features_extract.parquet")
}