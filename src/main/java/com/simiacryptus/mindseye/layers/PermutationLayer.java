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

public class PermutationLayer extends NNLayer {

  private static final Logger log = LoggerFactory.getLogger(PermutationLayer.class);

  private boolean verbose;

  public PermutationLayer() {
  }

  private List<double[]> record = null;
  
  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    NDArray output = input;
    if (isVerbose()) {
      PermutationLayer.log.debug(String.format("Feed forward: %s => %s", input, output));
    }
    if(null != record)
    {
      record.add(Arrays.copyOf(input.getData(), input.getData().length));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final LogNDArray data, final DeltaBuffer buffer) {
        if (inObj[0].isAlive()) {
          final LogNDArray passback = new LogNDArray(data.getDims());
          IntStream.range(0, passback.dim()).forEach(i -> {
            passback.set(i, data.getData()[i]);
          });
          if (isVerbose()) {
            PermutationLayer.log.debug(String.format("Feed back @ %s: %s => %s", output, data, passback));
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

  public PermutationLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

  public List<double[]> getRecord() {
    assert(null != record);
    assert(0 < record.size());
    List<double[]> prev = record;
    this.record = new java.util.ArrayList<>();
    return prev;
  }

  public void record() {
    this.record = new java.util.ArrayList<>();
  }

}
