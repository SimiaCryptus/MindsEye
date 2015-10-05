package com.simiacryptus.mindseye.net.dev;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import groovy.lang.Tuple2;

public class MinMaxFilterLayer extends NNLayer<MinMaxFilterLayer> {
  /**
   * 
   */
  private static final long serialVersionUID = -3791321603590434332L;

  private final class Result extends NNResult {
    private final NNResult inObj;

    private Result(final NDArray data, final NNResult inObj) {
      super(data);
      this.inObj = inObj;
    }

    @Override
    public void feedback(final NDArray delta, final DeltaBuffer buffer) {
      if (isVerbose()) {
        MinMaxFilterLayer.log.debug(String.format("Feed back: %s", this.data));
      }
      final double[] deltaData = delta.getData();
      if (this.inObj.isAlive()) {
        final int[] dims = this.inObj.data.getDims();
        final NDArray passback = new NDArray(dims);
        for (int i = 0; i < passback.dim(); i++) {
          if (this.inObj.data.getData()[i] > getThreshold()) {
            if (0 > deltaData[i]) {
              passback.set(i, deltaData[i]);
            }
          } else if (this.inObj.data.getData()[i] < -getThreshold()) {
            if (0 <= deltaData[i]) {
              passback.set(i, deltaData[i]);
            }
          } else {
            passback.set(i, deltaData[i]);
          }
        }
        this.inObj.feedback(passback, buffer);
        if (isVerbose()) {
          MinMaxFilterLayer.log.debug(String.format("Feed back @ %s=>%s: %s => %s", this.inObj.data, Result.this.data, delta, passback));
        }
      } else {
        if (isVerbose()) {
          MinMaxFilterLayer.log.debug(String.format("Feed back via @ %s=>%s: %s => null", this.inObj.data, Result.this.data, delta));
        }
      }
    }

    @Override
    public boolean isAlive() {
      return this.inObj.isAlive();
    }

  }

  private static final Logger log = LoggerFactory.getLogger(MinMaxFilterLayer.class);

  private double threshold = 20;

  public MinMaxFilterLayer() {
    super();
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final NDArray output = new NDArray(input.getDims());
    IntStream.range(0, input.dim()).forEach(i -> {
      final double x = input.getData()[i];
      final double r = Math.min(Math.max(x, -getThreshold()), getThreshold());
      output.set(i, r);
    });
    if (isVerbose()) {
      MinMaxFilterLayer.log.debug(String.format("Feed forward: %s => %s", inObj[0].data, output));
    }
    return new Result(output, inObj[0]);
  }

  protected double getMobility() {
    return 1;
  }

  public double getThreshold() {
    return this.threshold;
  }

  @Override
  public List<Tuple2<Integer, Integer>> permuteInput(final List<Tuple2<Integer, Integer>> permute) {
    return permute;
  }

  @Override
  public List<Tuple2<Integer, Integer>> permuteOutput(final List<Tuple2<Integer, Integer>> permute) {
    return permute;
  }

  public MinMaxFilterLayer setThreshold(final double threshold) {
    this.threshold = threshold;
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

}
