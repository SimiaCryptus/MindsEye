package com.simiacryptus.mindseye.net.dev;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.DeltaSet;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNResult;
import com.simiacryptus.mindseye.net.NNLayer;

import groovy.lang.Tuple2;

public class StrangeProductLayer extends NNLayer<StrangeProductLayer> {

  /**
   * 
   */
  private static final long serialVersionUID = -5171545060770814729L;
  private static final Logger log = LoggerFactory.getLogger(StrangeProductLayer.class);

  public StrangeProductLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    double sum = 1;
    for(int l=0;l<inObj.length;l++){
      final double[] input = inObj[l].data.getData();
      for (int i = 0; i < input.length; i++) {
        sum *= input[i];
      }
    }
    final double sum_ = sum;
    final NDArray output = new NDArray(new int[] { 1 }, new double[] { sum });
    if (isVerbose()) {
      StrangeProductLayer.log.debug(String.format("Feed forward: %s - %s => %s", inObj[0].data, inObj[1].data, sum));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data, final DeltaSet buffer) {
        double delta = data.get(0);
        for(int l=0;l<inObj.length;l++){
          NNResult in_l = inObj[l];
          if (in_l.isAlive()) {
            final NDArray passback = new NDArray(in_l.data.getDims());
            for (int i = 0; i < in_l.data.dim(); i++) {
              passback.set(i, delta / (sum_ / inObj[l].data.getData()[i]));
            }
            if (isVerbose()) {
              StrangeProductLayer.log.debug(String.format("Feed back @ %s: %s => %s", output, data, passback));
            }
            in_l.feedback(passback, buffer);
          }
        }
      }

      @Override
      public boolean isAlive() {
        for(int l=0;l<inObj.length;l++) if(inObj[0].isAlive()) return true;
        return false;
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
