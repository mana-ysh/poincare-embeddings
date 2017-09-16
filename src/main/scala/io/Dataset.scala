package io

import scala.io.Source

/**
  * Created by hitoshi-ma on 2017/09/14.
  */
class Dataset(val samples: Array[(Int, Int)]) {
  def take(index: Int): (Int, Int) ={
    samples(index)
  }

  def size(): Int ={
    samples.length
  }
}

object Dataset{
  def load(fileName: String, Vocab: Vocabulary): Dataset ={
    val lines = Source.fromFile(fileName).getLines.toArray
    val samples = lines.map({ line =>
      val triple = line.split("\t")
      (Vocab.getID(triple(0)), Vocab.getID(triple(1)))
    })
    new Dataset(samples)
  }
}