/*
TODO: must refactor (may be inefficient calculation in gradients, and modify to be more functional)
*/


import breeze.linalg._
import breeze.numerics._

/**
  * Created by hitoshi-ma on 2017/09/14.
  */

class PoincareEmbedding(val numNode: Int, val dim: Int, val lr: Double){

  // initializer
  val sampler = breeze.stats.distributions.Uniform(-0.0001, 0.0001)
  val lookupTable = DenseMatrix.rand(numNode, dim, sampler)
  val eps = 1e-6

  // assuming only child is corrupted
  def update(parent: Int, childPositive: Int, childNegatives: Array[Int]): Double ={
    val allChild = childPositive +: childNegatives
    val scores = allChild.map({ child =>
      val distance = computeDistance(parent, child)
      exp(-distance)
    })
    val z = scores.sum

    val loss = scores(0) / scores.sum

    val parentGrad = DenseMatrix.zeros[Double](allChild.length, dim)
    val childGrad = DenseMatrix.zeros[Double](allChild.length, dim)
    for (i <- allChild.indices){
      if (i == 0){
        val (parentG, childG) = computeGradient(allChild(i), parent, scores(i), 1-scores(i)/z)
        parentGrad(i, ::).t := parentG
        childGrad(i, ::).t := childG
      }
      else{
        val (parentG, childG) = computeGradient(allChild(0), parent, scores(0), -scores(i)/z)
        parentGrad(i, ::).t := parentG
        childGrad(i, ::).t := childG
      }
    }

    // update with Riemannian SGD
    for (i <- allChild.indices){
      val childId = allChild(i)
      lookupTable(childId, ::).t := projection(lookupTable(childId, ::).t + lr * childGrad(i, ::).t
                                                * pow(1.0 - pow(lookupTable(childId, ::).t, 2), 2) / 4.0)
      lookupTable(parent, ::).t := projection(lookupTable(parent, ::).t + lr * parentGrad(i, ::).t
                                                * pow(1.0 - pow(lookupTable(childId, ::).t, 2), 2) / 4.0) // is it appropriate?
    }

    loss
  }

  def projection(v: DenseVector[Double]): DenseVector[Double] ={
    if (norm(v) < 1.0) v else v / norm(v) - eps
  }

  def computeGradient(child: Int, parent: Int, score: Double, gradOut: Double): (DenseVector[Double], DenseVector[Double]) ={
    val parentNorm = norm(pick(parent))
    val childNorm = norm(pick(child))
    val alpha = 1 - pow(parentNorm, 2)
    val beta = 1 - pow(childNorm, 2)
    val gamma = 1 + 2 * norm(pick(parent) - pick(child)) / alpha / beta
    val coeff = gradOut * 4 / sqrt(gamma * gamma - 1) / alpha / beta
    val inner = pick(parent).t * pick(child)
    val parentG = pow(1 - pow(parentNorm, 2), 2) / 4 * coeff *
                  ((pow(childNorm, 2) - 2 * inner + 1) / alpha * pick(parent) - pick(child))
    val childG = pow(1 - pow(childNorm, 2), 2) / 4 * coeff *
                  ((pow(parentNorm, 2) - 2 * inner + 1) / beta * pick(child) - pick(parent))
    (parentG, childG)
  }

  def pick(nodeID: Int): DenseVector[Double] ={
    lookupTable(nodeID, ::).t
  }

  def computeDistance(parent: Int, child: Int): Double ={
    val parentV = pick(parent)
    val childV = pick(child)
    val value = 1 + 2 * norm(parentV - childV) / (1 - norm(parentV)) / (1 - norm(childV))
    arcosh(value)
  }


  // TODO: separate from this class
  def arcosh(x: Double): Double ={
    require(x >= 1)
    log(x + sqrt(x*x -1))
  }

  def save: Unit ={
    throw new Exception()
  }
}

object PoincareEmbedding {
  def build(numVocab: Int, dim: Int, lr: Double): PoincareEmbedding ={
    new PoincareEmbedding(numVocab, dim, lr)
  }
}
