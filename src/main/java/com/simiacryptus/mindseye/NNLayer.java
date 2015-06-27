package com.simiacryptus.mindseye;

import com.simiacryptus.mindseye.learning.NNResult;

/**
 * Nonlinear Network Layer (aka Neural Network Layer)
 * 
 * @author Andrew Charneski
 *
 */
public abstract class NNLayer {

  public final NNResult eval(NDArray array) {
    return eval(new NNResult(array));
  }

  public abstract NNResult eval(NNResult array);

}
