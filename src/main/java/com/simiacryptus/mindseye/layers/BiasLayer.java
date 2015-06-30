package com.simiacryptus.mindseye.layers;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.learning.DeltaMassMomentumBuffer;
import com.simiacryptus.mindseye.learning.MassParameters;
import com.simiacryptus.mindseye.learning.NNResult;

public class BiasLayer extends NNLayer implements MassParameters<BiasLayer> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BiasLayer.class);

  private final double[] bias;
  private final DeltaMassMomentumBuffer deltaBuffer;

  protected BiasLayer() {
    super();
    this.bias = null;
    this.deltaBuffer = null;
  }

  public BiasLayer(final int[] outputDims) {
    this.bias = new double[NDArray.dim(outputDims)];
    this.deltaBuffer = new DeltaMassMomentumBuffer(this.bias);
  }

  @Override
  public NNResult eval(final NNResult inObj) {
    final NDArray translated = inObj.data.map((v, i) -> {
      return v + this.bias[i.index];
    });
    return new NNResult(translated) {
      @Override
      public void feedback(final NDArray data) {
        for (int i = 0; i < BiasLayer.this.bias.length; i++)
        {
          BiasLayer.this.bias[i] += data.data[i];
        }
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
  
}
