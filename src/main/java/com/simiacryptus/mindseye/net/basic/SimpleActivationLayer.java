package com.simiacryptus.mindseye.net.basic;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.DeltaSet;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNResult;
import com.simiacryptus.mindseye.net.NNLayer;

public abstract class SimpleActivationLayer<T extends SimpleActivationLayer<T>> extends NNLayer<T> {

  /**
   * 
   */
  private static final long serialVersionUID = -5439874559292833041L;
  private static final Logger log = LoggerFactory.getLogger(SigmoidActivationLayer.class);

  public SimpleActivationLayer() {
    super();
  }

  protected abstract void eval(final double x, double[] results);

  @Override
  public NNResult eval(final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final NDArray output = new NDArray(inObj[0].data.getDims());
    final NDArray inputGradient = new NDArray(input.dim());
    final double[] results = new double[2];
    for (int i = 0; i < input.dim(); i++) {
      eval(input.getData()[i], results);
      inputGradient.set(i, results[1]);
      output.set(i, results[0]);
    }
    if (isVerbose()) {
      log.debug(String.format("Feed forward: %s => %s", inObj[0].data, output));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data, final DeltaSet buffer) {
        if (inObj[0].isAlive()) {
          final NDArray inputGradientLog = inputGradient;
          final NDArray passback = new NDArray(data.getDims());
          double[] gradientData = inputGradientLog.getData();
          IntStream.range(0, passback.dim()).forEach(i -> {
            double v = gradientData[i];
            if (Double.isFinite(v)) {
              passback.set(i, data.getData()[i] * v);
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
  public List<double[]> state() {
    return Arrays.asList();
  }

}
