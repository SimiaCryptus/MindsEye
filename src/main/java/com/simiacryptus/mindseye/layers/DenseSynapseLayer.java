package com.simiacryptus.mindseye.layers;

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
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.LogNumber;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.training.EvaluationContext;
import com.simiacryptus.mindseye.util.Util;

import groovy.lang.Tuple2;

public class DenseSynapseLayer extends NNLayer {
  private final class DenseSynapseResult extends NNResult {
    private final NNResult inObj;

    private DenseSynapseResult(final NDArray data, final NNResult inObj) {
      super(data);
      this.inObj = inObj;
    }

    @Override
    public void feedback(final LogNDArray delta, final DeltaBuffer buffer) {
      if (isVerbose()) {
        DenseSynapseLayer.log.debug(String.format("Feed back: %s", this.data));
      }
      final LogNumber[] deltaData = delta.getData();

      if (!isFrozen()) {
        final double[] inputData = this.inObj.data.getData();
        final LogNDArray weightDelta = new LogNDArray(DenseSynapseLayer.this.weights.getDims());
        for (int i = 0; i < weightDelta.getDims()[0]; i++) {
          for (int j = 0; j < weightDelta.getDims()[1]; j++) {
            weightDelta.set(new int[] { i, j }, deltaData[j].multiply(inputData[i]));
          }
        }
        buffer.get(DenseSynapseLayer.this, DenseSynapseLayer.this.weights).feed(weightDelta.exp().getData());
      }
      if (this.inObj.isAlive()) {
        final DoubleMatrix matrix = DenseSynapseLayer.this.weights.asMatrix();
        final LogNDArray passback = new LogNDArray(this.inObj.data.getDims());
        for (int i = 0; i < matrix.columns; i++) {
          for (int j = 0; j < matrix.rows; j++) {
            passback.add(i, deltaData[j].multiply(matrix.get(j, i)));
          }
        }
        this.inObj.feedback(passback, buffer);
        if (isVerbose()) {
          DenseSynapseLayer.log.debug(String.format("Feed back @ %s=>%s: %s => %s", this.inObj.data, DenseSynapseResult.this.data, delta, passback));
        }
      } else {
        if (isVerbose()) {
          DenseSynapseLayer.log.debug(String.format("Feed back via @ %s=>%s: %s => null", this.inObj.data, DenseSynapseResult.this.data, delta));
        }
      }
    }

    @Override
    public boolean isAlive() {
      return this.inObj.isAlive() || !isFrozen();
    }

  }

  private static final Logger log = LoggerFactory.getLogger(DenseSynapseLayer.class);

  private boolean frozen = false;
  private final int[] outputDims;
  private boolean verbose = false;
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
    final NDArray output = new NDArray(this.outputDims);
    for(int i=0;i<input.dim();i++) {
      for(int o=0;o<output.dim();o++) {
        final double a = this.weights.get(i, o);
        final double b = input.getData()[i];
        final double value = b * a;
        if (Double.isFinite(value)) {
          output.add(o, value);
        }
      }
    }
    if (isVerbose()) {
      DenseSynapseLayer.log.debug(String.format("Feed forward: %s * %s => %s", inObj[0].data, this.weights, output));
    }
    return new DenseSynapseResult(output, inObj[0]);
  }

  public DenseSynapseLayer freeze() {
    return freeze(true);
  }

  public DenseSynapseLayer freeze(final boolean b) {
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

  public DenseSynapseLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
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

  public DenseSynapseLayer thaw() {
    return freeze(false);
  }
}
