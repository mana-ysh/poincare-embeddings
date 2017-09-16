package io

import scala.io.Source
import scala.collection.mutable.Map

/**
  * Created by hitoshi-ma on 2017/09/14.
  */
class Vocabulary {
  private val wordToId = Map[String, Int]()
  private def add(word: String): Unit ={
    if (!wordToId.contains(word)){
      wordToId += word -> wordToId.size
    }
  }

  def getID(word: String): Int ={
    wordToId(word)
  }

  def size(): Int ={
    wordToId.size
  }

}

object Vocabulary{
  def load(fileName: String): Vocabulary ={
    val lines = Source.fromFile(fileName).getLines.toVector
    val vocab = new Vocabulary
    lines.foreach(line => vocab.add(line))
    vocab
  }
}
