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

public class EntropyLossLayer extends NNLayer<EntropyLossLayer> {

  private static final Logger log = LoggerFactory.getLogger(EntropyLossLayer.class);

  public EntropyLossLayer() {
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj) {
    final NDArray a = inObj[0].data;
    final NDArray b = inObj[1].data;
    final NDArray r = new NDArray(a.getDims());
    double total = 0;
    for(int i=0;i<a.dim();i++) {
      double bd = b.getData()[i];
      double ad = Math.max(Math.min(a.getData()[i], 1.), 1e-12);
      r.getData()[i] = -bd / ad;
      total += -bd * Math.log(ad);
    }
    double rms = total/a.dim();
    final NDArray output = new NDArray(new int[]{1}, new double[]{rms});
    if (isVerbose()) {
      EntropyLossLayer.log.debug(String.format("Feed forward: %s - %s => %s", inObj[0].data, inObj[1].data, rms));
    }
    return new NNResult(evaluationContext, output) {
      @Override
      public void feedback(final NDArray data, final DeltaBuffer buffer) {
        if (inObj[0].isAlive()||inObj[1].isAlive()) {
          final NDArray passback = new NDArray(r.getDims());
          double v = data.get(0) * a.dim();
          for (int i = 0; i < a.dim(); i++) {
            passback.set(i, v * r.get(i));
          }
          if (isVerbose()) {
            EntropyLossLayer.log.debug(String.format("Feed back @ %s: %s => %s", output, data, passback));
          }
          if (inObj[0].isAlive()) {
            inObj[0].feedback(passback, buffer);
          }
          if (inObj[1].isAlive()) {
            throw new RuntimeException();
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
