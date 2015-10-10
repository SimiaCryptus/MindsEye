package com.simiacryptus.mindseye.net.loss;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

public class SqLossLayer extends NNLayer<SqLossLayer> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SqLossLayer.class);
  /**
   * 
   */
  private static final long serialVersionUID = 7589211270512485408L;

  public SqLossLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    final NDArray a = inObj[0].data;
    final NDArray b = inObj[1].data;
    final NDArray r = new NDArray(a.getDims());
    double total = 0;
    for (int i = 0; i < a.dim(); i++) {
      final double x = a.getData()[i] - b.getData()[i];
      r.getData()[i] = x;
      total += x * x;
    }
    final double rms = total / a.dim();
    final NDArray output = new NDArray(new int[] { 1 }, new double[] { rms });
    return new NNResult(output) {
      @Override
      public void accumulate(final DeltaSet buffer, final NDArray data) {
        if (inObj[0].isAlive() || inObj[1].isAlive()) {
          final NDArray passback = new NDArray(r.getDims());
          final int adim = a.dim();
          final double data0 = data.get(0);
          for (int i = 0; i < adim; i++) {
            passback.set(i, data0 * r.get(i) * 2 / adim);
          }
          if (inObj[0].isAlive()) {
            inObj[0].accumulate(buffer, passback);
          }
          if (inObj[1].isAlive()) {
            inObj[1].accumulate(buffer, passback.scale(-1));
          }
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
