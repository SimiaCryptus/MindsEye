package com.simiacryptus.mindseye.net.dev;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.DeltaSet;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNResult;
import com.simiacryptus.mindseye.net.NNLayer;

public class PermutationLayer extends NNLayer<PermutationLayer> {

  private static final Logger log = LoggerFactory.getLogger(PermutationLayer.class);

  /**
   * 
   */
  private static final long serialVersionUID = -4908444887182647449L;

  private List<double[]> record = null;

  public PermutationLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final NDArray output = input;
    if (isVerbose()) {
      PermutationLayer.log.debug(String.format("Feed forward: %s => %s", input, output));
    }
    if (null != this.record) {
      this.record.add(Arrays.copyOf(input.getData(), input.getData().length));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data, final DeltaSet buffer) {
        if (inObj[0].isAlive()) {
          final NDArray passback = new NDArray(data.getDims());
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

  public List<double[]> getRecord() {
    assert null != this.record;
    assert 0 < this.record.size();
    final List<double[]> prev = this.record;
    this.record = new java.util.ArrayList<>();
    return prev;
  }

  public void record() {
    this.record = new java.util.ArrayList<>();
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }

}
