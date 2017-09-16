import scala.util.Random

/*
TODO
- write config parser with typesafe.config
- write iterator for training loops
*/


/**
  * Created by hitoshi-ma on 2017/09/14.
  */
object Train {
    def main(args: Array[String]): Unit ={

      val dim = 5
      val numNegative = 5
      val lr = 0.001
      val maxEpoch = 100

      val vocab = io.Vocabulary.load("./data/mammal/list")
      val train = io.Dataset.load("./data/mammal/mammal_subtree.tsv", vocab)
      val model = PoincareEmbedding.build(vocab.size, dim, lr)

      val rng = new Random(46)
      for (epoch <- 1 to maxEpoch){
        println(s"start $epoch epoch")
        var sumLoss: Double = 0
        for (i <- rng.shuffle(0 to train.size-1)) {
          val sample = train.take(i)
          val negatives = (for (i <- 1 to numNegative) yield rng.nextInt(vocab.size - 1)).toArray
          val loss = model.update(sample._1, sample._2, negatives)
          sumLoss += loss
        }
        println(s"sum loss: $sumLoss")
      }

      println("DONE ALL")
    }
}
