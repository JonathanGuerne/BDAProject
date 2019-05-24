import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.GeneralizedLinearRegression
import org.apache.spark.ml.regression.RandomForestRegressor
import org.apache.spark.ml.regression.DecisionTreeRegressor
import org.apache.spark.ml.regression.{GBTRegressionModel, GBTRegressor}
import org.apache.spark.ml.feature.{RFormula, VectorAssembler}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.AFTSurvivalRegression
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.ml.evaluation.ClusteringEvaluator


object ModelComparison extends App {

  val nbModel: Int = 2

  val spark = SparkSession
    .builder()
    .master("local")
    .appName("Spark MLlib basic example")
    .getOrCreate()


  //reading data and splitting to train and test
  val df = spark.read.json("dataset/test.json")
  val Array(train, test) = df.randomSplit(Array(0.7, 0.3))

  // TODO john linear regression
  //creating estimators
  val rForm_lr = new RFormula()
  val lr = new LinearRegression().setLabelCol("label").setFeaturesCol("features")

  // making them as pipeline stages
  val stages_lr = Array(rForm_lr, lr)
  val pipeline_lr = new Pipeline().setStages(stages_lr)

  //training
  val params_lr = new ParamGridBuilder()
    .addGrid(rForm_lr.formula, Array(
      "lab ~ . + color:value1")) //,
    //      "lab ~ . + color:value1 + color:value2"))
    .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
    .addGrid(lr.regParam, Array(0.1, 2.0))
    .build()


  // TODO john setup a generalised linear regression
  //creating estimators
  val rForm_glr = new RFormula()
  val glr = new GeneralizedLinearRegression().setLabelCol("label").setFeaturesCol("features")

  // making them as pipeline stages
  val stages_glr = Array(rForm_glr, glr)
  val pipeline_glr = new Pipeline().setStages(stages_glr)

  //training
  val params_glr = new ParamGridBuilder()
    .addGrid(rForm_glr.formula, Array(
      "lab ~ . + color:value1")) //,
    //      "lab ~ . + color:value1 + color:value2"))
    .addGrid(glr.family, Array("Gaussian", "Binomial", "Poisson"))
    .addGrid(glr.regParam, Array(0.1, 2.0))
    .build()

  // TODO thibaut setup a random forsest
  //creating estimators
  val rForm_rf = new RFormula()
  val rf = new RandomForestRegressor().setLabelCol("label").setFeaturesCol("features")

  // making them as pipeline stages
  val stages_rf = Array(rForm_rf, rf)
  val pipeline_rf = new Pipeline().setStages(stages_rf)

  //training
  val params_rf = new ParamGridBuilder()
    .addGrid(rForm_rf.formula, Array(
      "lab ~ . + color:value1")) //,
    //      "lab ~ . + color:value1 + color:value2"))
    .addGrid(rf.maxDepth, Array(10, 25))
    .addGrid(rf.numTrees, Array(5, 10))
    .build()

  // TODO thibaut decision tree regression
  //creating estimators
  val rForm_dt = new RFormula()
  val dt = new DecisionTreeRegressor().setLabelCol("label").setFeaturesCol("features")

  // making them as pipeline stages
  val stages_dt = Array(rForm_dt, dt)
  val pipeline_dt = new Pipeline().setStages(stages_dt)

  //training
  val params_dt = new ParamGridBuilder()
    .addGrid(rForm_dt.formula, Array(
      "lab ~ . + color:value1")) //,
    //      "lab ~ . + color:value1 + color:value2"))
    .addGrid(dt.maxDepth, Array(10, 25))
    .build()


  // TODO john setup a gradient boosted tree regression
  //creating estimators
  val rForm_gbt = new RFormula()
  val gbt = new GBTRegressor().setLabelCol("label").setFeaturesCol("features")

  // making them as pipeline stages
  val stages_gbt = Array(rForm_gbt, gbt)
  val pipeline_gbt = new Pipeline().setStages(stages_gbt)

  //training
  val params_gbt = new ParamGridBuilder()
    .addGrid(rForm_gbt.formula, Array(
      "lab ~ . + color:value1")) //,
    //      "lab ~ . + color:value1 + color:value2"))
    .addGrid(gbt.maxDepth, Array(10, 25))
    .build()





  //evaluation
    val evaluator = new RegressionEvaluator()
      .setMetricName("r2")
      .setPredictionCol("prediction")
      .setLabelCol("label")

    val models_arr = List(
      ("Linear Regression", params_lr, pipeline_lr),
      ("Generalized Linear Regression", params_glr, pipeline_glr),
      ("Decision Tree Regression", params_dt, pipeline_dt),
      ("Random Forest Regression", params_rf, pipeline_rf),
      ("Gradient Boosted Tree Regression", params_gbt, pipeline_gbt))

    for (el <- models_arr) {

      //model selection
      val tvs = new TrainValidationSplit()
        .setTrainRatio(0.75) // also the default.
        .setEstimatorParamMaps(el._2)
        .setEstimator(el._3)
        .setEvaluator(evaluator)

      //running the pipeline
      val tvsFitted = tvs.fit(train)

      //evaluating on the test set
      val testTransformed = tvsFitted.transform(test)
      val result = evaluator.evaluate(testTransformed)

      println(el._1 + " evaluation result : " + result)
    }


  // https://spark.apache.org/docs/2.4.3/ml-classification-regression.html


  // TODO john setup an isontonic regression

  // https://spark.apache.org/docs/2.4.3/ml-clustering.html#latent-dirichlet-allocation-lda


  val assembler = new VectorAssembler()
    .setInputCols(Array("value1", "value2"))
    .setOutputCol("features")

  val df_ = assembler.transform(train)
    .select("features")


  //Trains a k-means model.
  val kmeans = new KMeans().setFeaturesCol("features").setK(2).setSeed(1L)
  val model = kmeans.fit(df_)

  // Make predictions
  val predictions = model.transform(df_)

  //  Evaluate clustering by computing Silhouette score
  val evaluatorCluster = new ClusteringEvaluator()

  val silhouette = evaluatorCluster.evaluate(predictions)
  println(s"Silhouette with squared euclidean distance = $silhouette")

  // Shows the result.
  println("Cluster Centers: ")
  model.clusterCenters.foreach(println)


  predictions.show(200)

}
