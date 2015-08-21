package com.simiacryptus.mindseye.layers;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.learning.DeltaTransaction;
import com.simiacryptus.mindseye.learning.NNResult;

/**
 * Nonlinear Network Layer (aka Neural Network Layer)
 *
 * @author Andrew Charneski
 */
public abstract class NNLayer {
  
  public final NNResult eval(final NDArray array) {
    return eval(new NNResult(array) {
      @Override
      public void feedback(final NDArray data) {
        // Do Nothing
      }

      @Override
      public boolean isAlive() {
        return false;
      }
    });
  }

  public abstract NNResult eval(NNResult array);

  private DeltaTransaction vector;
  
  public final DeltaTransaction newVector(double fraction) {
    vector = newVector(fraction,Util.R.get().nextLong());
    return vector;
  }

  protected DeltaTransaction newVector(double fraction,long mask) {
    return null;
  }

  public DeltaTransaction getVector() {
    if(null==vector) newVector(1);
    return vector;
  }
  
}
