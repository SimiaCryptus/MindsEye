package com.simiacryptus.mindseye.net.dev;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;

import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.gson.JsonObject;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Coordinate;
import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

public class DenseSynapseLayerJBLAS extends NNLayer<DenseSynapseLayerJBLAS> {

  private final class Result extends NNResult {
    private final NNResult inObj;

    private Result(final Tensor[] outputA, final NNResult inObj) {
      super(outputA);
      this.inObj = inObj;
    }

    private Tensor[] backprop(final Tensor[] delta, final DeltaSet buffer) {
      Tensor[] passbackA = java.util.stream.IntStream.range(0, inObj.data.length).parallel().mapToObj(dataIndex->{
        final double[] deltaData = delta[dataIndex].getData();
        final Tensor r = DenseSynapseLayerJBLAS.this.getWeights();
        final Tensor passback = new Tensor(this.inObj.data[dataIndex].getDims());
        multiplyT(r.getData(), deltaData, passback.getData());
        return passback;
      }).toArray(i->new Tensor[i]);
      this.inObj.accumulate(buffer, passbackA);
      return passbackA;
    }

    @Override
    public void accumulate(final DeltaSet buffer, final Tensor[] delta) {
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

    private void learn(final Tensor[] delta, final DeltaSet buffer) {
      java.util.stream.IntStream.range(0, inObj.data.length).parallel().forEach(dataIndex->{
        final double[] deltaData = delta[dataIndex].getData();
        final double[] inputData = this.inObj.data[dataIndex].getData();
        final Tensor weightDelta = multiply(deltaData, inputData);
        buffer.get(DenseSynapseLayerJBLAS.this, DenseSynapseLayerJBLAS.this.getWeights()).accumulate(weightDelta.getData());
      });
    }

  }

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DenseSynapseLayerJBLAS.class);

  private static final long serialVersionUID = 3538627887600182889L;

  static Tensor multiply(final double[] deltaData, final double[] inputData) {
    final Tensor weightDelta = new Tensor(inputData.length, deltaData.length);
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
    org.jblas.DoubleMatrix matrixObj = new org.jblas.DoubleMatrix(out.length, in.length, matrix);
    double[] r = matrixObj.mmul(new org.jblas.DoubleMatrix(in.length, 1, in)).data; 
    for (int o = 0; o < out.length; o++) out[o] = r[o];
  }

  public static void multiplyT(final double[] matrix, final double[] in, double[] out) {
    org.jblas.DoubleMatrix matrixObj = new org.jblas.DoubleMatrix(in.length, out.length, matrix).transpose();
    double[] r = matrixObj.mmul(new org.jblas.DoubleMatrix(in.length, 1, in)).data; 
    for (int o = 0; o < out.length; o++) out[o] = r[o];
  }

  public final int[] outputDims;
  private final int[] inputDims;
  private final Tensor weights;

  protected DenseSynapseLayerJBLAS() {
    super();
    this.outputDims = null;
    this.weights = null;
    this.inputDims = null;
  }

  public DenseSynapseLayerJBLAS(final int[] inputDims, final int[] outputDims) {
    final int inputs = Tensor.dim(inputDims);
    this.inputDims = Arrays.copyOf(inputDims, inputDims.length);
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    int outs = Tensor.dim(outputDims);
    this.weights = new Tensor(inputs, outs);
    setWeights(() -> {
      double ratio = Math.sqrt(6. / (inputs + outs));
      double fate = Util.R.get().nextDouble();
      double v = (1 - 2 * fate) * ratio;
      return v;
    });
  }

  public DenseSynapseLayerJBLAS addWeights(final DoubleSupplier f) {
    Util.add(f, this.getWeights().getData());
    return this;
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    Tensor[] outputA = java.util.stream.IntStream.range(0, inObj[0].data.length).parallel().mapToObj(dataIndex->{
      final Tensor input = inObj[0].data[dataIndex];
      return multiply2(this.getWeights().getData(), input.getData());
    }).toArray(i->new Tensor[i]);
    return new Result(outputA, inObj[0]);
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.addProperty("weights", this.getWeights().toString());
    return json;
  }

  private Tensor multiply2(final double[] wdata, final double[] indata) {
    final Tensor output = new Tensor(this.outputDims);
    multiply(wdata, indata, output.getData());
    return output;
  }

  public DenseSynapseLayerJBLAS setWeights(final double[] data) {
    this.weights.set(data);
    return this;
  }

  public DenseSynapseLayerJBLAS setWeights(final java.util.function.ToDoubleFunction<Coordinate> f) {
    weights.coordStream().parallel().forEach(c->{
      weights.set(c, f.applyAsDouble(c));
    });
    return this;
  }

  public DenseSynapseLayerJBLAS setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.weights.getData(), i -> f.getAsDouble());
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList(this.getWeights().getData());
  }

  public Tensor getWeights() {
    return weights;
  }

}
