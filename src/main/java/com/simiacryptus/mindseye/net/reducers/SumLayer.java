package com.simiacryptus.mindseye.net.reducers;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.DeltaSet;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNResult;
import com.simiacryptus.mindseye.net.NNLayer;

public class SumLayer extends NNLayer<SumLayer> {

  private static final Logger log = LoggerFactory.getLogger(SumLayer.class);
  /**
   * 
   */
  private static final long serialVersionUID = -5171545060770814729L;

  public SumLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    double sum = 0;
    for (final NNResult element : inObj) {
      final double[] input = element.data.getData();
      for (final double element2 : input) {
        sum += element2;
      }
    }
    final NDArray output = new NDArray(new int[] { 1 }, new double[] { sum });
    if (isVerbose()) {
      SumLayer.log.debug(String.format("Feed forward: %s - %s => %s", inObj[0].data, inObj[1].data, sum));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data, final DeltaSet buffer) {
        final double delta = data.get(0);
        for (final NNResult in_l : inObj) {
          if (in_l.isAlive()) {
            final NDArray passback = new NDArray(in_l.data.getDims());
            for (int i = 0; i < in_l.data.dim(); i++) {
              passback.set(i, delta);
            }
            if (isVerbose()) {
              SumLayer.log.debug(String.format("Feed back @ %s: %s => %s", output, data, passback));
            }
            in_l.feedback(passback, buffer);
          }
        }
      }

      @Override
      public boolean isAlive() {
        for (final NNResult element : inObj)
          if (element.isAlive())
            return true;
        return false;
      }

    };
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
