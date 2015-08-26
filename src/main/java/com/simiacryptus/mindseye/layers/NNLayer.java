package com.simiacryptus.mindseye.layers;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.learning.DeltaVector;
import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.NDArray;

/**
 * Nonlinear Network Layer (aka Neural Network Layer)
 *
 * @author Andrew Charneski
 */
public abstract class NNLayer {
  
  public final NNResult eval(final NDArray array) {
    return eval(new NNResult(array) {
      @Override
      public void feedback(final LogNDArray data) {
        // Do Nothing
      }

      @Override
      public boolean isAlive() {
        return false;
      }
    });
  }

  public abstract NNResult eval(NNResult array);

  private DeltaVector vector;
  private Double fraction = null;
  private Long mask = null;
  private double currentStatusValue = Double.MAX_VALUE;
  
  public final DeltaVector newVector(double fraction) {
    if(null != this.fraction && this.fraction.equals(fraction)) return this.vector;
    this.fraction = fraction;
    this.mask = Util.R.get().nextLong();
    vector = newVector(fraction,mask);
    return vector;
  }

  protected DeltaVector newVector(double fraction,long mask) {
    return null;
  }

  public DeltaVector getVector() {
    if(null==vector) newVector(1);
    return vector;
  }

  public void setStatus(double value) {
    this.currentStatusValue = value;
  }

  public double getStatus() {
    return currentStatusValue;
  }
  
}
