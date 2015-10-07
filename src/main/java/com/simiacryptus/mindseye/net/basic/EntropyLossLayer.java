package com.simiacryptus.mindseye.net.basic;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.DeltaSet;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNResult;
import com.simiacryptus.mindseye.net.NNLayer;

import groovy.lang.Tuple2;

public class EntropyLossLayer extends NNLayer<EntropyLossLayer> {

  /**
   * 
   */
  private static final long serialVersionUID = -6257785994031662519L;
  private static final Logger log = LoggerFactory.getLogger(EntropyLossLayer.class);

  public EntropyLossLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    final NDArray l = inObj[0].data;
    final NDArray b = inObj[1].data;
    final NDArray gradient = new NDArray(l.getDims());
    double[] gradientData = gradient.getData();
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
    if (isVerbose()) {
      EntropyLossLayer.log.debug(String.format("Feed forward: %s - %s => %s", inObj[0].data, inObj[1].data, descriptiveNats));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data, final DeltaSet buffer) {
        if (inObj[0].isAlive() || inObj[1].isAlive()) {
          final NDArray passback = new NDArray(gradient.getDims());
          for (int i = 0; i < l.dim(); i++) {
            passback.set(i, data.get(0) * gradient.get(i));
          }
          if (isVerbose()) {
            EntropyLossLayer.log.debug(String.format("Feed back @ %s: %s => %s", output, data, passback));
          }
          if (inObj[0].isAlive()) {
            inObj[0].feedback(passback, buffer);
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
  public List<Tuple2<Integer, Integer>> permuteInput(final List<Tuple2<Integer, Integer>> permute) {
    return permute;
  }

  @Override
  public List<Tuple2<Integer, Integer>> permuteOutput(final List<Tuple2<Integer, Integer>> permute) {
    return permute;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
