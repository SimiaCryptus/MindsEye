package com.simiacryptus.mindseye.net.basic;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.dag.EvaluationContext;

import groovy.lang.Tuple2;

public class SumLayer extends NNLayer<SumLayer> {

  private static final Logger log = LoggerFactory.getLogger(SumLayer.class);

  public SumLayer() {
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj) {
    double sum = 0;
    for(int l=0;l<inObj.length;l++){
      final NDArray input = inObj[l].data;
      {
        double total = 0;
        for (int i = 0; i < input.dim(); i++) {
          total += input.getData()[i];
        }
        sum += total;
      }
    }
    
    final NDArray output = new NDArray(new int[] { 1 }, new double[] { sum });
    if (isVerbose()) {
      SumLayer.log.debug(String.format("Feed forward: %s - %s => %s", inObj[0].data, inObj[1].data, sum));
    }
    return new NNResult(evaluationContext, output) {
      @Override
      public void feedback(final NDArray data, final DeltaBuffer buffer) {
        for(int l=0;l<inObj.length;l++){
          NNResult in_l = inObj[l];
          if (in_l.isAlive()) {
            final NDArray passback = new NDArray(in_l.data.getDims());
            for (int i = 0; i < in_l.data.dim(); i++) {
              passback.set(i, data.get(0));
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
