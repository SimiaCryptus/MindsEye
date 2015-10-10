package com.simiacryptus.mindseye.net.dev;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.DeltaSet;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNResult;
import com.simiacryptus.mindseye.net.NNLayer;

public class MaxEntropyLossLayer extends NNLayer<MaxEntropyLossLayer> {

  /**
   * 
   */
  private static final long serialVersionUID = 4246204583991554340L;
  private static final Logger log = LoggerFactory.getLogger(MaxEntropyLossLayer.class);

  public MaxEntropyLossLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    final NDArray a = inObj[0].data;
    final NDArray b = inObj[1].data;
    final NDArray r = new NDArray(a.getDims());
    double total = 0;
    assert 2 == a.dim();
    for (int i = 0; i < a.dim(); i++) {
      final double bd = b.getData()[1 - i];
      final double ad = Math.max(Math.min(a.getData()[i], 1.), 1e-18);
      r.getData()[i] = bd / ad;
      total += bd * Math.log(ad);
    }
    final double rms = total / a.dim();
    final NDArray output = new NDArray(new int[] { 1 }, new double[] { rms });
    if (isVerbose()) {
      MaxEntropyLossLayer.log.debug(String.format("Feed forward: %s - %s => %s", inObj[0].data, inObj[1].data, rms));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data, final DeltaSet buffer) {
        if (inObj[0].isAlive()) {
          final NDArray passback = new NDArray(r.getDims());
          final double v = data.get(0) / a.dim();
          for (int i = 0; i < a.dim(); i++) {
            passback.set(i, v * r.get(i));
          }
          if (isVerbose()) {
            MaxEntropyLossLayer.log.debug(String.format("Feed back @ %s: %s => %s", output, data, passback));
          }
          inObj[0].feedback(passback, buffer);
        }
        if (inObj[1].isAlive())
          throw new RuntimeException();
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
