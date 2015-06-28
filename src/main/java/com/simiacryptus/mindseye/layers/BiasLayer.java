package com.simiacryptus.mindseye.layers;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNLayer;
import com.simiacryptus.mindseye.learning.DeltaInversionBuffer;
import com.simiacryptus.mindseye.learning.DeltaMassMomentumBuffer;
import com.simiacryptus.mindseye.learning.NNResult;

public class BiasLayer extends NNLayer implements MassParameters<BiasLayer> {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(BiasLayer.class);
  
  private final double[] bias;
  private final DeltaMassMomentumBuffer deltaBuffer;
  
  public BiasLayer(int[] outputDims) {
    this.bias = new double[NDArray.dim(outputDims)];
    this.deltaBuffer = new DeltaMassMomentumBuffer(bias);
  }
  
  public NNResult eval(final NNResult inObj) {
    NDArray translated = inObj.data.map((v, i) -> {
      return v + bias[i.index];
    });
    return new NNResult(translated) {
      @Override
      public void feedback(NDArray data) {
        for (int i = 0; i < bias.length; i++)
        {
          bias[i] += data.data[i];
        }
        if (inObj.isAlive())
        {
          inObj.feedback(data);
        }
      }
      
      public boolean isAlive() {
        return true;
      }
    };
  }

  public double getMomentumDecay() {
    return deltaBuffer.getMomentumDecay();  
  }

  public BiasLayer setMomentumDecay(double momentumDecay) {
    deltaBuffer.setMomentumDecay(momentumDecay);
    return this;
  }

  public double getMass() {
    return deltaBuffer.getMass();
  }

  public BiasLayer setMass(double mass) {
    deltaBuffer.setMass(mass);
    return this;
  }

}
