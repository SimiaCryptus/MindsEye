package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.function.DoubleSupplier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.learning.DeltaBuffer;
import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.NDArray;

public class BiasLayer extends NNLayer {
  
  private static final Logger log = LoggerFactory.getLogger(BiasLayer.class);
  
  public final double[] bias;
  private boolean frozen = false;
  private boolean verbose = false;
  
  protected BiasLayer() {
    super();
    this.bias = null;
  }
  
  public BiasLayer(final int[] outputDims) {
    this.bias = new double[NDArray.dim(outputDims)];
  }

  public BiasLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.bias);
    return this;
  }

  @Override
  public NNResult eval(final NNResult inObj) {
    final NDArray translated = inObj.data.map((v, i) -> {
      return v + this.bias[i.index];
    });
    if (isVerbose()) {
      BiasLayer.log.debug(String.format("Feed forward: %s => %s", inObj.data, translated));
    }
    return new NNResult(translated) {
      @Override
      public void feedback(final LogNDArray data, DeltaBuffer buffer) {
        if (isVerbose()) {
          log.debug(String.format("Feed back: %s", data));
        }
        buffer.get(BiasLayer.this, BiasLayer.this.bias).feed(data.getData());
        if (inObj.isAlive())
        {
          inObj.feedback(data, buffer);
        }
      }
      
      @Override
      public boolean isAlive() {
        return inObj.isAlive() || !isFrozen();
      }
    };
  }

  public BiasLayer freeze() {
    return setFrozen(true);
  }

  public boolean isFrozen() {
    return this.frozen;
  }

  public boolean isVerbose() {
    return this.verbose;
  }

  public NNLayer set(final double[] ds) {
    for (int i = 0; i < ds.length; i++) {
      this.bias[i] = ds[i];
    }
    return this;
  }

  public BiasLayer setFrozen(final boolean frozen) {
    this.frozen = frozen;
    return this;
  }

  public BiasLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  @Override
  public String toString() {
    return "BiasLayer " + Arrays.toString(this.bias);
  }

  protected double getMobility() {
    return 1;
  }

}
