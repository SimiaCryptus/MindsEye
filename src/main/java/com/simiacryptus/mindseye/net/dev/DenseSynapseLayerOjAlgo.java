package com.simiacryptus.mindseye.net.dev;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;

import org.ojalgo.array.PrimitiveArray;
import org.ojalgo.matrix.store.RawStore;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.core.Coordinate;
import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

public class DenseSynapseLayerOjAlgo extends NNLayer<DenseSynapseLayerOjAlgo> {

  private final class Result extends NNResult {
    private final NNResult inObj;

    private Result(final NDArray[] outputA, final NNResult inObj) {
      super(outputA);
      this.inObj = inObj;
    }

    private NDArray[] backprop(final NDArray[] delta, final DeltaSet buffer) {
      NDArray[] passbackA = java.util.stream.IntStream.range(0, inObj.data.length).parallel().mapToObj(dataIndex->{
        final double[] deltaData = delta[dataIndex].getData();
        final NDArray r = DenseSynapseLayerOjAlgo.this.getWeights();
        final NDArray passback = new NDArray(this.inObj.data[dataIndex].getDims());
        multiplyT(r.getData(), deltaData, passback.getData());
        return passback;
      }).toArray(i->new NDArray[i]);
      this.inObj.accumulate(buffer, passbackA);
      return passbackA;
    }

    @Override
    public void accumulate(final DeltaSet buffer, final NDArray[] delta) {
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

    private void learn(final NDArray[] delta, final DeltaSet buffer) {
      java.util.stream.IntStream.range(0, inObj.data.length).parallel().forEach(dataIndex->{
        final double[] deltaData = delta[dataIndex].getData();
        final double[] inputData = this.inObj.data[dataIndex].getData();
        final NDArray weightDelta = multiply(deltaData, inputData);
        buffer.get(DenseSynapseLayerOjAlgo.this, DenseSynapseLayerOjAlgo.this.getWeights()).feed(weightDelta.getData());
      });
    }

  }

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DenseSynapseLayerOjAlgo.class);

  /**
   * 
   */
  private static final long serialVersionUID = 3538627887600182889L;

  private static NDArray multiply(final double[] deltaData, final double[] inputData) {
    final NDArray weightDelta = new NDArray(inputData.length, deltaData.length);
    crossMultiply(deltaData, inputData, weightDelta.getData());
    return weightDelta;
  }

  public static void crossMultiply(final double[] rows, final double[] cols, double[] matrix) {
    int i = 0;
    for (final double c : cols) {
      for (final double r : rows) {
        matrix[i++] = r * c;
      }
    }
  }

  public static void multiply(final double[] matrix, final double[] in, double[] out) {
    org.ojalgo.matrix.store.RawStore matrixObj = new org.ojalgo.matrix.store.RawStore(matrix, out.length);
    PrimitiveArray inVec = org.ojalgo.array.PrimitiveArray.wrap(in);
    RawStore outVec = matrixObj.multiply(inVec).transpose();
    double[] ds = outVec.data[0]; for (int o = 0; o < out.length; o++) out[o] = ds[o];
  }

  public static void multiplyT(final double[] matrix, final double[] in, double[] out) {
    org.ojalgo.matrix.store.RawStore matrixObj = new org.ojalgo.matrix.store.RawStore(matrix, in.length).transpose();
    PrimitiveArray inVec = org.ojalgo.array.PrimitiveArray.wrap(in);
    RawStore outVec = matrixObj.multiply(inVec).transpose();
    double[] ds = outVec.data[0]; for (int o = 0; o < out.length; o++) out[o] = ds[o];
  }

  public final int[] outputDims;

  private final NDArray weights;

  protected DenseSynapseLayerOjAlgo() {
    super();
    this.outputDims = null;
    this.weights = null;
  }

  public DenseSynapseLayerOjAlgo(final int inputs, final int[] outputDims) {
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    this.weights = new NDArray(inputs, NDArray.dim(outputDims));
    int outs = NDArray.dim(outputDims);
    setWeights(() -> {
      double ratio = Math.sqrt(6. / (inputs + outs));
      double fate = Util.R.get().nextDouble();
      double v = (1 - 2 * fate) * ratio;
      return v;
    });
  }

  public DenseSynapseLayerOjAlgo addWeights(final DoubleSupplier f) {
    Util.add(f, this.getWeights().getData());
    return this;
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    NDArray[] outputA = java.util.stream.IntStream.range(0, inObj[0].data.length).parallel().mapToObj(dataIndex->{
      final NDArray input = inObj[0].data[dataIndex];
      return multiply2(this.getWeights().getData(), input.getData());
    }).toArray(i->new NDArray[i]);
    return new Result(outputA, inObj[0]);
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.addProperty("weights", this.getWeights().toString());
    return json;
  }

  protected double getMobility() {
    return 1;
  }

  private NDArray multiply2(final double[] wdata, final double[] indata) {
    final NDArray output = new NDArray(this.outputDims);
    multiply(wdata, indata, output.getData());
    return output;
  }

  public DenseSynapseLayerOjAlgo setWeights(final double[] data) {
    this.weights.set(data);
    return this;
  }

  public DenseSynapseLayerOjAlgo setWeights(final java.util.function.ToDoubleFunction<Coordinate> f) {
    weights.coordStream().parallel().forEach(c->{
      weights.set(c, f.applyAsDouble(c));
    });
    return this;
  }

  public DenseSynapseLayerOjAlgo setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.weights.getData(), i -> f.getAsDouble());
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList(this.getWeights().getData());
  }

  public NDArray getWeights() {
    return weights;
  }

}
