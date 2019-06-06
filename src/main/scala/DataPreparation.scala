import org.apache.spark.sql.SparkSession
import org.apache.spark.SparkContext
import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark
import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.ml.feature.MinMaxScaler


object DataPreparation extends App {
  val spark = SparkSession.builder.config("spark.master", "local[2]").getOrCreate()

  import spark.implicits._

  def vectorize(dataFrame: DataFrame, inputCol: String, outputCol: String, size: Int): DataFrame = {
    val word2Vec = new Word2Vec()
      .setInputCol(inputCol)
      .setOutputCol(outputCol)
      .setVectorSize(size)
      .setMinCount(0)
    val model = word2Vec.fit(dataFrame)

    model.transform(dataFrame)
  }

  def normalize(dataFrame: DataFrame, inputCol: String, outputCol: String): DataFrame = {
    val vectorizeCol = udf( (v:Double) => Vectors.dense(Array(v)) )
    val df2 = dataFrame.withColumn(inputCol+"Vec", vectorizeCol(dataFrame(inputCol)))

    val scaler = new MinMaxScaler()
      .setInputCol(inputCol+"Vec")
      .setOutputCol(outputCol)
      .setMax(1)
      .setMin(-1)

    val scalerModel = scaler.fit(df2)

    scalerModel.transform(df2).drop(inputCol+"Vec")
  }

  var features = spark.read.parquet("features_extract.parquet")
  features.createOrReplaceTempView("features")

  features.printSchema()

  val remover = new StopWordsRemover()
    .setInputCol("title")
    .setOutputCol("titleStopWords")
  features = remover.transform(features)

  /*
  val features_titles_count = features.select(size($"titleStopWords"))
  features_titles_count.describe().show()
  features_titles_count.orderBy("size(titleStopWords)")
  print(features_titles_count.take((features_titles_count.count()/2).toInt).head.toString()) // median
   */

  //The mean size of title is 2.3 but we up this number to 3.

  features = vectorize(features, "titleStopWords", "titleVec", 3)
  features = vectorize(features, "directors", "directorsVec", 4)
  features = vectorize(features, "writers", "writersVec", 5)
  features = vectorize(features, "genres", "genresVec", 3)
  features = features.drop("title", "titleStopWords", "directors", "writers", "genres")

  features = normalize(features, "startYear", "startYearNormalize")
  features = normalize(features, "duration", "durationNormalize")
  features = features.drop("startYear", "duration")

  features.show(50, false)
  features.write.mode(SaveMode.Overwrite).parquet("features_prepared.parquet")
}