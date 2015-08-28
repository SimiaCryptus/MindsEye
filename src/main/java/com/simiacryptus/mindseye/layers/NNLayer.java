package com.simiacryptus.mindseye.layers;

import java.util.UUID;

import com.simiacryptus.mindseye.learning.DeltaBuffer;
import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.NDArray;

/**
 * Nonlinear Network Layer (aka Neural Network Layer)
 *
 * @author Andrew Charneski
 */
public abstract class NNLayer {

  private double currentStatusValue = Double.MAX_VALUE;

  private String id = UUID.randomUUID().toString();

  public final NNResult eval(final NDArray array) {
    return eval(new NNResult(array) {
      @Override
      public void feedback(final LogNDArray data, final DeltaBuffer buffer) {
        // Do Nothing
      }

      @Override
      public boolean isAlive() {
        return false;
      }
    });
  }
  
  public abstract NNResult eval(NNResult array);

  public String getId() {
    return this.id;
  }

  public double getStatus() {
    return this.currentStatusValue;
  }

  public NNLayer setId(final String id) {
    this.id = id;
    return this;
  }

  public void setStatus(final double value) {
    this.currentStatusValue = value;
  }

}
