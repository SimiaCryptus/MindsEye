package com.simiacryptus.mindseye.net.dev;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.dag.EvaluationContext;
import com.simiacryptus.mindseye.util.Util;

import groovy.lang.Tuple2;

public class LinearActivationLayer extends NNLayer {
  private final class DenseSynapseResult extends NNResult {
    private final NNResult inObj;

    private DenseSynapseResult(final NDArray data, final NNResult inObj) {
      super(data);
      this.inObj = inObj;
    }

    @Override
    public void feedback(final NDArray delta, final DeltaBuffer buffer) {
      if (isVerbose()) {
        LinearActivationLayer.log.debug(String.format("Feed back: %s", this.data));
      }
      final double[] deltaData = delta.getData();

      if (!isFrozen()) {
        final double[] inputData = this.inObj.data.getData();
        final NDArray weightDelta = new NDArray(LinearActivationLayer.this.weights.getDims());
        for (int i = 0; i < deltaData.length; i++) {
          weightDelta.add(0, deltaData[i] * inputData[i]);
        }
        buffer.get(LinearActivationLayer.this, LinearActivationLayer.this.weights).feed(weightDelta.getData());
      }
      if (this.inObj.isAlive()) {
        final DoubleMatrix matrix = LinearActivationLayer.this.weights.asRowMatrix();
        final int[] dims = this.inObj.data.getDims();
        final NDArray passback = new NDArray(dims);
        for (int i = 0; i < passback.dim(); i++) {
          passback.set(i, deltaData[i] * matrix.get(0, 0));
        }
        this.inObj.feedback(passback, buffer);
        if (isVerbose()) {
          LinearActivationLayer.log.debug(String.format("Feed back @ %s=>%s: %s => %s", this.inObj.data, DenseSynapseResult.this.data, delta, passback));
        }
      } else {
        if (isVerbose()) {
          LinearActivationLayer.log.debug(String.format("Feed back via @ %s=>%s: %s => null", this.inObj.data, DenseSynapseResult.this.data, delta));
        }
      }
    }

    @Override
    public boolean isAlive() {
      return this.inObj.isAlive() || !isFrozen();
    }

  }

  private static final Logger log = LoggerFactory.getLogger(LinearActivationLayer.class);

  private boolean frozen = false;
  private boolean verbose = false;
  public final NDArray weights;

  public LinearActivationLayer() {
    super();
    this.weights = new NDArray(1);
    this.weights.set(0, 1.);
  }

  public LinearActivationLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.weights.getData());
    return this;
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final NDArray output = new NDArray(input.getDims());
    IntStream.range(0, input.dim()).forEach(i -> {
      final double a = this.weights.get(0);
      final double b = input.getData()[i];
      final double value = b * a;
      if (Double.isFinite(value)) {
        output.add(i, value);
      }
    });
    if (isVerbose()) {
      LinearActivationLayer.log.debug(String.format("Feed forward: %s * %s => %s", inObj[0].data, this.weights, output));
    }
    return new DenseSynapseResult(output, inObj[0]);
  }

  @Override
  public LinearActivationLayer freeze() {
    return freeze(true);
  }

  public LinearActivationLayer freeze(final boolean b) {
    this.frozen = b;
    return this;
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.addProperty("weights", this.weights.toString());
    return json;
  }

  protected double getMobility() {
    return 1;
  }

  public boolean isFrozen() {
    return this.frozen;
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

  public LinearActivationLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  public LinearActivationLayer setWeights(final double[] data) {
    this.weights.set(data);
    return this;
  }

  public LinearActivationLayer setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.weights.getData(), i -> f.getAsDouble());
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList(this.weights.getData());
  }

  public LinearActivationLayer thaw() {
    return freeze(false);
  }

}
