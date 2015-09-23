package com.simiacryptus.mindseye.net.basic;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.function.DoubleSupplier;

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

public class DenseSynapseLayer extends NNLayer<DenseSynapseLayer> {


  private final class Result extends NNResult {
    private final NNResult inObj;

    private Result(final NDArray data, final NNResult inObj, final EvaluationContext evaluationContext) {
      super(evaluationContext, data);
      this.inObj = inObj;
    }

    @Override
    public void feedback(final NDArray delta, final DeltaBuffer buffer) {
      if (isVerbose()) {
        DenseSynapseLayer.log.debug(String.format("Feed back: %s", this.data));
      }
      if (!isFrozen()) {
        learn(delta, buffer);
      }
      if (this.inObj.isAlive()) {
        final NDArray passback = backprop(delta, buffer);
        if (isVerbose()) {
          DenseSynapseLayer.log.debug(String.format("Feed back @ %s=>%s: %s => %s", this.inObj.data, Result.this.data, delta, passback));
        }
      } else {
        if (isVerbose()) {
          DenseSynapseLayer.log.debug(String.format("Feed back via @ %s=>%s: %s => null", this.inObj.data, Result.this.data, delta));
        }
      }
    }

    private NDArray backprop(final NDArray delta, final DeltaBuffer buffer) {
      final double[] deltaData = delta.getData();
      final DoubleMatrix matrix = DenseSynapseLayer.this.weights.asMatrix();
      final NDArray passback = new NDArray(this.inObj.data.getDims());
      for (int i = 0; i < matrix.columns; i++) {
        for (int j = 0; j < matrix.rows; j++) {
          passback.add(i, deltaData[j] * matrix.get(j, i));
        }
      }
      this.inObj.feedback(passback, buffer);
      return passback;
    }

    private void learn(final NDArray delta, final DeltaBuffer buffer) {
      final double[] deltaData = delta.getData();
      final double[] inputData = this.inObj.data.getData();
      final NDArray weightDelta = multiply(deltaData, inputData);
      buffer.get(DenseSynapseLayer.this, DenseSynapseLayer.this.weights).feed(weightDelta.getData());
    }

    @Override
    public boolean isAlive() {
      return this.inObj.isAlive() || !isFrozen();
    }

  }

  private static final Logger log = LoggerFactory.getLogger(DenseSynapseLayer.class);

  private final int[] outputDims;
  public final NDArray weights;

  protected DenseSynapseLayer() {
    super();
    this.outputDims = null;
    this.weights = null;
  }

  public DenseSynapseLayer(final int inputs, final int[] outputDims) {
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    this.weights = new NDArray(inputs, NDArray.dim(outputDims));
  }

  public DenseSynapseLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.weights.getData());
    return this;
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final NDArray output = multiply2(this.weights.getData(), input.getData());
    if (isVerbose()) {
      DenseSynapseLayer.log.debug(String.format("Feed forward: %s * %s => %s", inObj[0].data, this.weights, output));
    }
    return new Result(output, inObj[0], evaluationContext);
  }

  private NDArray multiply2(final double[] wdata, final double[] indata) {
    final NDArray output = new NDArray(this.outputDims);
    multiply2(wdata, indata, output);
    return output;
  }

  private static void multiply2(final double[] wdata, final double[] indata, final NDArray output) {
    final int outdim = output.dim();
    int idx = 0;
    for (int i = 0; i < indata.length; i++) {
      final double b = indata[i];
      for (int o = 0; o < outdim; o++) {
        final double a = wdata[idx++];
        final double value = b * a;
        output.add(o, value);
      }
    }
  }
  
  private static NDArray multiply(final double[] deltaData, final double[] inputData) {
    final NDArray weightDelta = new NDArray(inputData.length, deltaData.length);
    multiply(deltaData, inputData, weightDelta);
    return weightDelta;
  }

  private static void multiply(final double[] deltaData, final double[] inputData, final NDArray weightDelta) {
    int k = 0;
    for (int i = 0; i < inputData.length; i++) {
      for (int j = 0; j < deltaData.length; j++) {
        weightDelta.set(k++, deltaData[j] * inputData[i]);
      }
    }
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

  @Override
  public List<Tuple2<Integer, Integer>> permuteInput(final List<Tuple2<Integer, Integer>> permute) {
    final java.util.Map<Integer, Integer> shuffleMap = new HashMap<>();
    final int inputs = this.weights.getDims()[0];
    java.util.stream.IntStream.range(0, inputs).forEach(i -> shuffleMap.put(i, i));
    ;

    permute.forEach(t -> {
      Integer from = t.getFirst();
      final Integer to = t.getSecond();
      from = shuffleMap.get(from);

      final int outputs = this.weights.getDims()[1];
      for (int output = 0; output < outputs; output++) {
        final double temp = this.weights.get(to, output);
        this.weights.set(new int[] { to, output }, this.weights.get(from, output));
        this.weights.set(new int[] { from, output }, temp);
      }

      for (final int k : shuffleMap.keySet()) {
        int value = shuffleMap.get(k);
        if (value == from) {
          value = to;
        } else if (value == to) {
          value = from;
        }
        shuffleMap.put(k, value);
      }
    });
    return null;
  }

  @Override
  public List<Tuple2<Integer, Integer>> permuteOutput(final List<Tuple2<Integer, Integer>> permute) {
    final java.util.Map<Integer, Integer> shuffleMap = new HashMap<>();
    final int outputs = this.weights.getDims()[1];
    java.util.stream.IntStream.range(0, outputs).forEach(i -> shuffleMap.put(i, i));
    ;

    permute.forEach(t -> {
      int from = t.getFirst();
      final int to = t.getSecond();
      from = shuffleMap.get(from);

      final int inputs = this.weights.getDims()[0];
      for (int input = 0; input < inputs; input++) {
        final double temp = this.weights.get(input, to);
        final double x = this.weights.get(input, from);
        this.weights.set(new int[] { input, to }, x);
        this.weights.set(new int[] { input, from }, temp);
      }

      for (final int k : shuffleMap.keySet()) {
        int value = shuffleMap.get(k);
        if (value == from) {
          value = to;
        } else if (value == to) {
          value = from;
        }
        shuffleMap.put(k, value);
      }
    });
    return null;
  }

  public DenseSynapseLayer setWeights(final double[] data) {
    this.weights.set(data);
    return this;
  }

  public DenseSynapseLayer setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.weights.getData(), i -> f.getAsDouble());
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList(this.weights.getData());
  }

}
