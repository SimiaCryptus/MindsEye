package com.simiacryptus.mindseye.net.loss;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

public class EntropyLossLayer extends NNLayer<EntropyLossLayer> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(EntropyLossLayer.class);
  /**
   * 
   */
  private static final long serialVersionUID = -6257785994031662519L;

  public EntropyLossLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    final NDArray l = inObj[0].data;
    final NDArray b = inObj[1].data;
    final NDArray gradient = new NDArray(l.getDims());
    final double[] gradientData = gradient.getData();
    final double descriptiveNats;
    {
      double total = 0;
      for (int i = 0; i < l.dim(); i++) {
        final double ad = Math.max(Math.min(l.getData()[i], 1.), 1e-12);
        final double bd = b.getData()[i];
        gradientData[i] = -bd / ad;
        total += -bd * Math.log(ad);
      }
      descriptiveNats = total;
    }

    final NDArray output = new NDArray(new int[] { 1 }, new double[] { descriptiveNats });
    return new NNResult(output) {
      @Override
      public void accumulate(final DeltaSet buffer, final NDArray data) {
        if (inObj[0].isAlive() || inObj[1].isAlive()) {
          final NDArray passback = new NDArray(gradient.getDims());
          for (int i = 0; i < l.dim(); i++) {
            passback.set(i, data.get(0) * gradient.get(i));
          }
          if (inObj[0].isAlive()) {
            inObj[0].accumulate(buffer, passback);
          }
          if (inObj[1].isAlive())
            throw new RuntimeException();
        }
      }

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }

    };
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
