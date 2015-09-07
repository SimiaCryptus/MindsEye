package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.EvaluationContext;

import groovy.lang.Tuple2;

public abstract class SimpleActivationLayer extends NNLayer {

  private static final Logger log = LoggerFactory.getLogger(SigmoidActivationLayer.class);

  protected abstract void eval(final double x, double[] results);

  private boolean verbose;

  public SimpleActivationLayer() {
    super();
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final NDArray output = new NDArray(inObj[0].data.getDims());
    final NDArray inputGradient = new NDArray(input.dim());
    double[] results = new double[2];
    for(int i=0;i<input.dim();i++) {
      eval(input.getData()[i], results);
      inputGradient.add(new int[] { i }, results[1]);
      output.set(i, results[0]);
    }
    if (isVerbose()) {
      log.debug(String.format("Feed forward: %s => %s", inObj[0].data, output));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final LogNDArray data, final DeltaBuffer buffer) {
        if (inObj[0].isAlive()) {
          final LogNDArray inputGradientLog = inputGradient.log();
          final LogNDArray passback = new LogNDArray(data.getDims());
          IntStream.range(0, passback.dim()).forEach(i -> {
            if (inputGradientLog.getData()[i].isFinite()) {
              passback.set(i, data.getData()[i].multiply(inputGradientLog.getData()[i]));
            }
          });
          if (isVerbose()) {
            log.debug(String.format("Feed back @ %s: %s => %s", output, data, passback));
          }
          inObj[0].feedback(passback, buffer);
        }
      }
  
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
    };
  }

  @Override
  public boolean isVerbose() {
    return this.verbose;
  }

  @Override
  public List<Tuple2<Integer, Integer>> permuteInput(final List<Tuple2<Integer, Integer>> permute) {
    return permute;
  }

  @Override
  public List<Tuple2<Integer, Integer>> permuteOutput(final List<Tuple2<Integer, Integer>> permute) {
    return permute;
  }

  public SimpleActivationLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

}
