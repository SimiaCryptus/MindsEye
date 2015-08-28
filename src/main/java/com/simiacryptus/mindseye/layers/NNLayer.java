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
  
  public final NNResult eval(final NDArray array) {
    return eval(new NNResult(array) {
      @Override
      public void feedback(final LogNDArray data, DeltaBuffer buffer) {
        // Do Nothing
      }
      
      @Override
      public boolean isAlive() {
        return false;
      }
    });
  }
  
  public abstract NNResult eval(NNResult array);
  
  private double currentStatusValue = Double.MAX_VALUE;
  private String id = UUID.randomUUID().toString();
  
  public void setStatus(double value) {
    this.currentStatusValue = value;
  }
  
  public double getStatus() {
    return currentStatusValue;
  }
  
  public String getId() {
    return id;
  }
  
  public NNLayer setId(String id) {
    this.id = id;
    return this;
  }
  
}
