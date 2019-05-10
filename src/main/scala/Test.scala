


object BDA_Project extends App {
  import org.apache.spark.SparkContext
import org.apache.spark.SparkConf

val conf = new SparkConf().setAppName("myApp").setMaster("local")

val sc: SparkContext = new SparkContext(conf)

val data = List("this", "is", "a", "test")
val distData = sc.parallelize(data)

distData.take(2).foreach(println)


}