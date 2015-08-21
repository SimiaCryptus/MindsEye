package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.function.DoubleSupplier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.learning.DeltaFlushBuffer;
import com.simiacryptus.mindseye.learning.DeltaMemoryWriter;
import com.simiacryptus.mindseye.learning.DeltaSampler;
import com.simiacryptus.mindseye.learning.DeltaTransaction;
import com.simiacryptus.mindseye.learning.NNResult;

public class BiasLayer extends NNLayer {
  
  private static final Logger log = LoggerFactory.getLogger(BiasLayer.class);
  
  public final double[] bias;
  private DeltaFlushBuffer flush;
  private boolean frozen = false;
  private DeltaSampler sampler;
  private boolean verbose = false;
  
  protected BiasLayer() {
    super();
    this.bias = null;
  }
  
  public BiasLayer(final int[] outputDims) {
    this.bias = new double[NDArray.dim(outputDims)];
    final DeltaMemoryWriter writer = new DeltaMemoryWriter(this.bias);
    this.flush = new DeltaFlushBuffer(writer);
    this.sampler = new DeltaSampler(this.flush);
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
      public void feedback(final NDArray data) {
        if (isVerbose()) {
          BiasLayer.log.debug(String.format("Feed back: %s", data));
        }
        BiasLayer.this.sampler.feed(data.getData());
        if (inObj.isAlive())
        {
          inObj.feedback(data);
        }
      }
      
      @Override
      public boolean isAlive() {
        return true;
      }
    };
  }

  public BiasLayer freeze() {
    return setFrozen(true);
  }

  public double getRate() {
    return this.flush.getRate();
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

  public void setRate(final double rate) {
    this.flush.setRate(rate);
  }

  public BiasLayer setSampling(final double sampling) {
    this.sampler.setSampling(sampling);
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
  

  protected DeltaTransaction newVector(double fraction,long mask) {
    return new DeltaTransaction() {
      
      @Override
      public void write(double factor) {
        if (isFrozen()) return;
        flush.write(factor, fraction, mask);
      }
      
      @Override
      public void setRate(double rate) {
        BiasLayer.this.setRate(rate);
      }
      
      @Override
      public boolean isFrozen() {
        return BiasLayer.this.isFrozen();
      }
      
      @Override
      public double getRate() {
        return BiasLayer.this.getRate();
      }
    };
  }

}
