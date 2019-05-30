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

  def countNull(colName:String) = sum(col(colName).isNull.cast("integer")).alias(colName)
  def countNotNull(colName:String) = sum(col(colName).isNotNull.cast("integer")).alias(colName)

  val spark = SparkSession.builder.config("spark.master", "local[2]").getOrCreate()

  import spark.implicits._

  val df_basics = spark.read
                       .option("sep", "\t")
                       .option("header", "true")
                       .option("nullValue", "\\N")
                       .csv("dataset/title.basics.tsv")

  df_basics.createOrReplaceTempView("titles")

  println("Number of rows of basic.tsv " + df_basics.count())

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

  println("Number of rows of ratings.tsv " + df_ratings.count())

  val df_step1 = df_step0.join(df_ratings,
                               df_ratings("tconst") === df_step0("tconst"))
                               .drop(df_ratings("tconst"))
                               .drop("numVotes")
  df_step1.printSchema()

  println("Number of rows of join with ratings and basic.tsv " + df_step1.count())


  // DONE 2 JOHN Look for each film's director(s) information is store in title.crew. T
  // ry to append directos name to dataframe

  val df_crew = spark.read
                     .option("sep", "\t")
                     .option("header", "true")
                     .option("nullValue", "\\N")
                     .csv("dataset/title.crew.tsv")

  df_crew.createOrReplaceTempView("crew")

  println("Number of rows of crew.tsv " + df_crew.count())

  // Inner join to remove unwanted ids
  val df_crew_movie = df_step1.join(df_crew,
                                    df_crew("tconst") === df_step1("tconst"))
                              .drop(df_crew("tconst"))
                              .na.drop()

  // df_crew_movie.agg(countNull("genres")).show()

  println("Number of rows of join with ratings, crew and basic.tsv " + df_crew_movie.count())

  //val nb_directors = df_crew_movie.collect().map(r => (r.get(8).toString.split(",").size))
  //  1 to 10 foreach { i =>
  //    println("#directors : " + i + " => " +nb_directors.foldLeft(0)((over_4, nb_dir) => if (nb_dir > i) over_4 + 1 else over_4))
  //  }
  // 4 is the best value to use


  //  val nb_writers = df_crew_movie.collect().map(r => (r.get(9).toString.split(",").size))
  //  1 to 10 foreach { i =>
  //      println("#writers : " + i + " => " +nb_writers.foldLeft(0)((over_4, nb_dir) => if (nb_dir > i) over_4 + 1 else over_4))
  //  }
  // 5 is the best value to use

  val parse_string: Column => Column = x => {
    split(x, ",")
  }

  val df_directors = df_crew_movie.withColumn("directors", parse_string($"directors")).select(
    $"tconst",
    $"directors".getItem(0).as("director1"),
    $"directors".getItem(1).as("director2"),
    $"directors".getItem(2).as("director3"),
    $"directors".getItem(3).as("director4"),
  ).na.fill(Map("director2" -> 0))
   .na.fill(Map("director3" -> 0))
   .na.fill(Map("director4" -> 0))

  val df_directors2 = df_directors.withColumn("directors",
    struct(df_directors("director1"), df_directors("director2"), df_directors("director3"), df_directors("director4")))
    .drop()

  val df_writers = df_crew_movie.withColumn("writers", parse_string($"writers")).select(
    $"tconst",
    $"writers".getItem(0).as("writer1"),
    $"writers".getItem(1).as("writer2"),
    $"writers".getItem(2).as("writer3"),
    $"writers".getItem(3).as("writer4"),
    $"writers".getItem(4).as("writer5"),
  ).na.fill(Map("writer2" -> 0))
   .na.fill(Map("writer3" -> 0))
   .na.fill(Map("writer4" -> 0))
   .na.fill(Map("writer5" -> 0))

  val df_step2_1 = df_step1.join(df_directors,
    df_directors("tconst") === df_step1("tconst")).drop(df_directors("tconst"))

  val df_step2 = df_step2_1.join(df_writers,
    df_writers("tconst") === df_step2_1("tconst")).drop(df_writers("tconst"))

  df_step2.printSchema()


  //DONE 3 PEDRO split genres into n columns, n beeing the maximum number of genre a movie can have

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

  df_step3.printSchema()

  // TODO thibaut export to csv

  df_step3.orderBy("startYear").show(50, false)
  df_step3.write.mode(SaveMode.Overwrite).parquet("features_extract.parquet")
}