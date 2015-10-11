package com.simiacryptus.mindseye.net.basic;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.Coordinate;
import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

public class DenseSynapseLayer extends NNLayer<DenseSynapseLayer> {

  private final class Result extends NNResult {
    private final NNResult inObj;

    private Result(final NDArray data, final NNResult inObj) {
      super(data);
      this.inObj = inObj;
    }

    private NDArray backprop(final NDArray delta, final DeltaSet buffer) {
      final double[] deltaData = delta.getData();
      final NDArray r = DenseSynapseLayer.this.weights;
      final DoubleMatrix matrix = new DoubleMatrix(r.getDims()[1], r.getDims()[0], r.getData());
      final NDArray passback = new NDArray(this.inObj.data.getDims());
      for (int i = 0; i < matrix.columns; i++) {
        for (int j = 0; j < matrix.rows; j++) {
          passback.add(i, deltaData[j] * matrix.get(j, i));
        }
      }
      this.inObj.accumulate(buffer, passback);
      return passback;
    }

    @Override
    public void accumulate(final DeltaSet buffer, final NDArray delta) {
      if (!isFrozen()) {
        learn(delta, buffer);
      }
      if (this.inObj.isAlive()) {
        backprop(delta, buffer);
      }
    }

    @Override
    public boolean isAlive() {
      return this.inObj.isAlive() || !isFrozen();
    }

    private void learn(final NDArray delta, final DeltaSet buffer) {
      final double[] deltaData = delta.getData();
      final double[] inputData = this.inObj.data.getData();
      final NDArray weightDelta = multiply(deltaData, inputData);
      buffer.get(DenseSynapseLayer.this, DenseSynapseLayer.this.weights).feed(weightDelta.getData());
    }

  }

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DenseSynapseLayer.class);

  /**
   * 
   */
  private static final long serialVersionUID = 3538627887600182889L;

  private static NDArray multiply(final double[] deltaData, final double[] inputData) {
    final NDArray weightDelta = new NDArray(inputData.length, deltaData.length);
    multiply(deltaData, inputData, weightDelta);
    return weightDelta;
  }

  private static void multiply(final double[] deltaData, final double[] inputData, final NDArray weightDelta) {
    int k = 0;
    for (final double element : inputData) {
      for (final double element2 : deltaData) {
        weightDelta.set(k++, element2 * element);
      }
    }
  }

  private static void multiply2(final double[] wdata, final double[] indata, final NDArray output) {
    final int outdim = output.dim();
    for (int o = 0; o < outdim; o++) {
      double sum = 0;
      for (int i = 0; i < indata.length; i++) {
        sum += indata[i] * wdata[o+i*outdim];
      }
      output.set(o, sum);
    }
  }

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
    setWeights(() -> (1 - 2 * Util.R.get().nextDouble()) * Math.sqrt(6 / (inputs + NDArray.dim(outputDims))));
  }

  public DenseSynapseLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.weights.getData());
    return this;
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final NDArray output = multiply2(this.weights.getData(), input.getData());
    return new Result(output, inObj[0]);
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

  private NDArray multiply2(final double[] wdata, final double[] indata) {
    final NDArray output = new NDArray(this.outputDims);
    multiply2(wdata, indata, output);
    return output;
  }

  public DenseSynapseLayer setWeights(final double[] data) {
    this.weights.set(data);
    return this;
  }

  public DenseSynapseLayer setWeights(final java.util.function.ToDoubleFunction<Coordinate> f) {
    weights.coordStream().parallel().forEach(c->{
      weights.set(c, f.applyAsDouble(c));
    });
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
