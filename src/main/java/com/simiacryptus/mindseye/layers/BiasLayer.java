package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.function.DoubleSupplier;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.learning.DeltaMassMomentum;
import com.simiacryptus.mindseye.learning.DeltaFlushBuffer;
import com.simiacryptus.mindseye.learning.DeltaMemoryWriter;
import com.simiacryptus.mindseye.learning.DeltaStochasticSampler;
import com.simiacryptus.mindseye.learning.DeltaTransaction;
import com.simiacryptus.mindseye.learning.MassParameters;
import com.simiacryptus.mindseye.learning.NNResult;

public class BiasLayer extends NNLayer implements MassParameters<BiasLayer>, DeltaTransaction {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BiasLayer.class);
  
  public final double[] bias;
  private final DeltaMassMomentum deltaBuffer;

  private DeltaFlushBuffer flush;

  private DeltaStochasticSampler sampler;
  
  protected BiasLayer() {
    super();
    this.bias = null;
    this.deltaBuffer = null;
  }
  
  public BiasLayer(final int[] outputDims) {
    this.bias = new double[NDArray.dim(outputDims)];
    DeltaMemoryWriter writer = new DeltaMemoryWriter(this.bias);
    this.deltaBuffer = new DeltaMassMomentum(writer);
    this.flush = new DeltaFlushBuffer(this.deltaBuffer);
    this.sampler = new DeltaStochasticSampler(this.flush);
  }
  
  @Override
  public NNResult eval(final NNResult inObj) {
    final NDArray translated = inObj.data.map((v, i) -> {
      return v + this.bias[i.index];
    });
    return new NNResult(translated) {
      @Override
      public void feedback(final NDArray data) {
        sampler.feed(data.data);
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

  @Override
  public double getMass() {
    return this.deltaBuffer.getMass();
  }

  @Override
  public double getMomentumDecay() {
    return this.deltaBuffer.getMomentumDecay();
  }

  @Override
  public BiasLayer setMass(final double mass) {
    this.deltaBuffer.setMass(mass);
    return this;
  }

  @Override
  public BiasLayer setMomentumDecay(final double momentumDecay) {
    this.deltaBuffer.setMomentumDecay(momentumDecay);
    return this;
  }


  public BiasLayer addWeights(final DoubleSupplier f) {
    Util.add(f, bias);
    return this;
  }

  @Override
  public String toString() {
    return "BiasLayer " + Arrays.toString(bias);
  }

  @Override
  public void write() {
    flush.write();
  }

  public BiasLayer setHalflife(final double halflife) {
    return setMomentumDecay(Math.exp(2 * Math.log(0.5) / halflife));
  }

  public BiasLayer setSampling(double sampling) {
    this.sampler.setSampling(sampling);
    return this;
  }

}
