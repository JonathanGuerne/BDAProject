import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.ml.evaluation.{ClusteringEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.{RFormula, VectorAssembler}
import org.apache.spark.ml.regression.{DecisionTreeRegressor, GBTRegressor, GeneralizedLinearRegression, LinearRegression, RandomForestRegressor, _}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{desc, stddev_pop}


object ModelComparison extends App {

  val nbModel: Int = 2



  val spark = SparkSession
    .builder()
    .master("local")
    .appName("Spark MLlib basic example")
    .getOrCreate()

  import spark.implicits._



  val set_apart_movies = List("tt1213644", "tt0060666", "tt0926129", "tt0111161", "tt0068646", "tt0119217")

  //reading data and splitting to train and test
  var features = spark.read.parquet("features_prepared.parquet")
  features.createOrReplaceTempView("features")

  val df_apart_movies = features.filter($"id" isin(set_apart_movies:_*))
  features = features.filter(!($"id" isin(set_apart_movies:_*)))

  var Array(train, test) = features.randomSplit(Array(0.7, 0.3))

  train = train.drop("id")
  test = test.drop("id")

  //  test.select("ratings").write.option("header",true).csv("out.csv")

  val features_name = Array("isAdult", "nb_words_title", "nb_genres", "nb_directors", "nb_writers", "titleVec",
    "directorsVec", "writersVec", "genresVec", "startYearNormalize", "durationNormalize")

  val features_r = Array("ratings ~ .")

  val label_name = "ratings"

  print("train " + train.count() + "\n")
  print("test " + test.count() + "\n")

  /*

  val assembler = new VectorAssembler()
    .setInputCols(features_name)
    .setOutputCol("features")

  val df_ = assembler.transform(train)
    .select("features")

  val Row(coeff1: Matrix) = Correlation.corr(df_, "features").head
  println("Pearson correlation matrix:\n" + coeff1.toString(22, Int.MaxValue))

  df_.show()

  // PCA (2)
  val pca = new PCA()
    .setInputCol("features")
    .setOutputCol("pcaFeatures")
    .setK(2)
    .fit(df_)

  val result = pca.transform(df_).select("pcaFeatures")
  result.show(false)

   */

  // TODO john linear regression
  //creating estimators
  val rForm_lr = new RFormula()
  val lr = new LinearRegression().setLabelCol(label_name).setFeaturesCol("features")

  // making them as pipeline stages
  val stages_lr = Array(rForm_lr, lr)
  val pipeline_lr = new Pipeline().setStages(stages_lr)

  //training
  val params_lr = new ParamGridBuilder()
    .addGrid(rForm_lr.formula, features_r)
    .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
    .addGrid(lr.regParam, Array(0.1, 2.0))
    .build()


  // TODO john setup a generalised linear regression
  //creating estimators
  val rForm_glr = new RFormula()
  val glr = new GeneralizedLinearRegression().setLabelCol(label_name).setFeaturesCol("features")

  // making them as pipeline stages
  val stages_glr = Array(rForm_glr, glr)
  val pipeline_glr = new Pipeline().setStages(stages_glr)

  //training
  val params_glr = new ParamGridBuilder()
    .addGrid(rForm_glr.formula, features_r)
    .addGrid(glr.family, Array("Gaussian", "Poisson", "Gamma"))
    .addGrid(glr.link, Array("Identity","Log"))
    .addGrid(glr.regParam, Array(0.1, 2.0))
    .build()

  // TODO thibaut setup a random forsest
  //creating estimators
  val rForm_rf = new RFormula()
  val rf = new RandomForestRegressor().setLabelCol(label_name).setFeaturesCol("features")

  // making them as pipeline stages
  val stages_rf = Array(rForm_rf, rf)
  val pipeline_rf = new Pipeline().setStages(stages_rf)

  //training
  val params_rf = new ParamGridBuilder()
    .addGrid(rForm_rf.formula, features_r)
    .addGrid(rf.maxDepth, Array(10, 25))
    .addGrid(rf.numTrees, Array(5, 10))
    .build()

  // TODO thibaut decision tree regression
  //creating estimators
  val rForm_dt = new RFormula()
  val dt = new DecisionTreeRegressor().setLabelCol(label_name).setFeaturesCol("features")

  // making them as pipeline stages
  val stages_dt = Array(rForm_dt, dt)
  val pipeline_dt = new Pipeline().setStages(stages_dt)

  //training
  val params_dt = new ParamGridBuilder()
    .addGrid(rForm_dt.formula, features_r)
    .addGrid(dt.maxDepth, Array(10, 25))
    .build()


  // TODO john setup a gradient boosted tree regression
  //creating estimators
  val rForm_gbt = new RFormula()
  val gbt = new GBTRegressor().setLabelCol(label_name).setFeaturesCol("features")

  // making them as pipeline stages
  val stages_gbt = Array(rForm_gbt, gbt)
  val pipeline_gbt = new Pipeline().setStages(stages_gbt)

  //training
  val params_gbt = new ParamGridBuilder()
    .addGrid(rForm_gbt.formula, features_r)
    .addGrid(gbt.maxDepth, Array(10, 25))
    .build()

  // TODO thibaut setup a survival regression


  // TODO john setup an isotonic regression
  //creating estimators
  val rForm_iso = new RFormula()
  val iso = new IsotonicRegression().setLabelCol(label_name).setFeaturesCol("features")

  // making them as pipeline stages
  val stages_iso = Array(rForm_iso, iso)
  val pipeline_iso = new Pipeline().setStages(stages_iso)

  //training
  val params_iso = new ParamGridBuilder()
    .addGrid(rForm_iso.formula, features_r)
    .addGrid(iso.isotonic, Array(true, false))
    .build()

  //evaluation
  val evaluator = new RegressionEvaluator()
    .setMetricName("mse")
    .setPredictionCol("prediction")
    .setLabelCol(label_name)


  val models_arr = List(
    ("Linear Regression", params_lr, pipeline_lr),
    ("Generalized Linear Regression", params_glr, pipeline_glr),
    ("Decision Tree Regression", params_dt, pipeline_dt),
    ("Random Forest Regression", params_rf, pipeline_rf),
    ("Isotonic Regression", params_iso, pipeline_iso),
    ("Gradient Boosted Tree Regression", params_gbt, pipeline_gbt)
  )

  for (el <- models_arr) {

    val cv = new CrossValidator()
      .setEstimator(el._3)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(el._2)
      .setNumFolds(3)  // WARNING 10 fold make training time very long
      .setParallelism(3)  // Evaluate up to 5 parameter settings in parallel

    // Run cross-validation, and choose the best set of parameters.
    val cvModel = cv.fit(train)

    /*
    //model selection
    val tvs = new TrainValidationSplit()
      .setTrainRatio(0.75) // also the default.
      .setEstimatorParamMaps(el._2)
      .setEstimator(el._3)
      .setEvaluator(evaluator)

    //running the pipeline
    val tvsFitted = tvs.fit(train)
    */

    //evaluating on the test set
    val testTransformed = cvModel.transform(test)
    val result = evaluator.evaluate(testTransformed)

    println(el._1 + " evaluation result : " + result)

    val apartMoviesPreds = cvModel.transform(df_apart_movies)
    apartMoviesPreds.select("id","ratings","prediction").show(false)

//    testTransformed.select("prediction").write.option("header",true).csv("prediction.csv")

  }


  // https://spark.apache.org/docs/2.4.3/ml-classification-regression.html
  // https://spark.apache.org/docs/2.4.3/ml-clustering.html#latent-dirichlet-allocation-lda

  //K-means

  val cl_assembler = new VectorAssembler()
    .setInputCols(features_name)
    .setOutputCol("features")

  val cl_df_ = cl_assembler.transform(train)
    .select("features", "ratings")


  //Trains a k-means model.
  val kmeans = new KMeans().setFeaturesCol("features").setK(10).setSeed(3L)
  val model = kmeans.fit(cl_df_)

  // Make predictions
  val predictions = model.transform(cl_df_)

  //  Evaluate clustering by computing Silhouette score
  val evaluatorCluster = new ClusteringEvaluator()

  val silhouette = evaluatorCluster.evaluate(predictions)
  println(s"Silhouette with squared euclidean distance = $silhouette")

  // Shows the result.
  println("Cluster Centers: ")
  model.clusterCenters.foreach(println)

  // Show the performance
  val df_avg=predictions.groupBy($"prediction").avg("ratings")
  val df_min=predictions.groupBy($"prediction").min("ratings")
  val df_max=predictions.groupBy($"prediction").max("ratings")
  val df_std=predictions.groupBy($"prediction").agg(stddev_pop($"ratings"))

  val df_temp=df_avg.join(df_min, df_avg("prediction")===df_min("prediction")).drop(df_min("prediction"))
  val df_temp2=df_temp.join(df_std, df_temp("prediction")===df_std("prediction")).drop(df_std("prediction"))
  val df_stats=df_temp2.join(df_max, df_temp2("prediction")===df_max("prediction")).drop(df_max("prediction"))

  df_stats.sort("prediction").show()

  // NOTE: add minimum and maximum values to thresholds
  val thresholds: Array[Double] = (1.0 until 11.0 by 1).toArray

  for (i <- 0 to 9)
    {
      // Convert DataFrame to RDD and calculate histogram values
      val _tmpHist = predictions.filter($"prediction"===i)
        .select($"ratings" cast "double")
        .rdd.map(r => r.getDouble(0))
        .histogram(thresholds)

      // Result DataFrame contains `from`, `to` range and the `value`.
      val histogram = spark.sparkContext.parallelize((thresholds, thresholds.tail, _tmpHist).zipped.toList).toDF("from", "to", "value")

      val rating_cluster=histogram.orderBy(desc("value")).select("from").first().get(0)


      println("Cluster : "+i+"      Rating : "+rating_cluster)
    }

  val _tmpHist = cl_df_
    .select($"ratings" cast "double")
    .rdd.map(r => r.getDouble(0))
    .histogram(thresholds)

  // Result DataFrame contains `from`, `to` range and the `value`.
  val histogram = spark.sparkContext.parallelize((thresholds, thresholds.tail, _tmpHist).zipped.toList).toDF("from", "to", "value")

  histogram.show(10)

}
