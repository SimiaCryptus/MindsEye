package com.simiacryptus.mindseye.layers;

import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.EvaluationContext;

public class ExpActivationLayer extends NNLayer {
  
  private static final Logger log = LoggerFactory.getLogger(ExpActivationLayer.class);
  
  private boolean verbose;
  
  public ExpActivationLayer() {
  }
  
  @Override
  public NNResult eval(EvaluationContext evaluationContext, final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final NDArray output = new NDArray(inObj[0].data.getDims());
    final NDArray inputGradient = new NDArray(input.dim());
    IntStream.range(0, input.dim()).forEach(i -> {
      final double x = input.getData()[i];
      final double ex = Math.exp(x);
      double d = ex;
      double f = ex;
      inputGradient.add(new int[] { i }, d);
      output.set(i, f);
    });
    if (isVerbose()) {
      ExpActivationLayer.log.debug(String.format("Feed forward: %s => %s", inObj[0].data, output));
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
            ExpActivationLayer.log.debug(String.format("Feed back @ %s: %s => %s", output, data, passback));
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
  
  public boolean isVerbose() {
    return this.verbose;
  }
  
  public ExpActivationLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }
}
