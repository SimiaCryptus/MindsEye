package com.simiacryptus.mindseye.net.dev;

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
import com.simiacryptus.mindseye.opencl.MatrixMultiplyKernel;

public class DenseSynapseLayerGPU extends NNLayer<DenseSynapseLayerGPU> {

  private final class Result extends NNResult {
    private final NNResult inObj;

    private Result(final NDArray[] outputA, final NNResult inObj) {
      super(outputA);
      this.inObj = inObj;
    }

    private NDArray[] backprop(final NDArray[] delta, final DeltaSet buffer) {
      NDArray[] passbackA = java.util.stream.IntStream.range(0, inObj.data.length).parallel().mapToObj(dataIndex->{
        final double[] deltaData = delta[dataIndex].getData();
        final NDArray r = DenseSynapseLayerGPU.this.weights;
        final DoubleMatrix matrix = new DoubleMatrix(r.getDims()[1], r.getDims()[0], r.getData());
        final NDArray passback = new NDArray(this.inObj.data[dataIndex].getDims());
        for (int i = 0; i < matrix.columns; i++) {
          for (int j = 0; j < matrix.rows; j++) {
            passback.add(i, deltaData[j] * matrix.get(j, i));
          }
        }
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
        final NDArray weightDelta = new NDArray(inputData.length, deltaData.length);
        double[] weightData = weightDelta.getData();
        gradientCrossMatrix(deltaData, inputData, weightData);
        buffer.get(DenseSynapseLayerGPU.this, DenseSynapseLayerGPU.this.weights).feed(weightData);
      });
    }

  }
  
  public static void gradientCrossMatrix(final double[] deltaData, final double[] inputData, double[] weightData) {
    gradientCrossMatrix(deltaData, inputData, weightData, 0, inputData.length);
  }

  private static void gradientCrossMatrix(final double[] deltaData, final double[] inputData, double[] weightData, int from, int to) {
    int k = from * deltaData.length;
    for (int i = from; i < to; i++) {
      final double element = inputData[i];
      for (int j = 0; j < deltaData.length; j++) {
        final double element2 = deltaData[j];
        weightData[k++]= element2 * element;
      }
    }
  }

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(DenseSynapseLayerGPU.class);

  /**
   * 
   */
  private static final long serialVersionUID = 3538627887600182889L;

  public final int[] outputDims;

  public final NDArray weights;

  protected DenseSynapseLayerGPU() {
    super();
    this.outputDims = null;
    this.weights = null;
  }

  public DenseSynapseLayerGPU(final int inputs, final int[] outputDims) {
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    this.weights = new NDArray(inputs, NDArray.dim(outputDims));
    setWeights(() -> (1 - 2 * Util.R.get().nextDouble()) * Math.sqrt(6 / (inputs + NDArray.dim(outputDims))));
  }

  public DenseSynapseLayerGPU addWeights(final DoubleSupplier f) {
    Util.add(f, this.weights.getData());
    return this;
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    
    NDArray[] inputA = java.util.stream.IntStream.range(0, inObj[0].data.length).parallel()
        .mapToObj(dataIndex->inObj[0].data[dataIndex]).toArray(i->new NDArray[i]);
    NDArray[] outputA = java.util.stream.IntStream.range(0, inObj[0].data.length).parallel()
        .mapToObj(dataIndex->new NDArray(this.outputDims)).toArray(i->new NDArray[i]);
    double[][] inputAD = java.util.Arrays.stream(inputA).parallel().map(x->x.getData()).toArray(ii->new double[ii][]);
    double[][] outputAD = java.util.Arrays.stream(outputA).parallel().map(x->x.getData()).toArray(ii->new double[ii][]);;
    MatrixMultiplyKernel.multiply(inputAD, this.weights.getData(), outputAD);
    return new Result(outputA, inObj[0]);
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

  public DenseSynapseLayerGPU setWeights(final double[] data) {
    this.weights.set(data);
    return this;
  }

  public DenseSynapseLayerGPU setWeights(final java.util.function.ToDoubleFunction<Coordinate> f) {
    weights.coordStream().parallel().forEach(c->{
      weights.set(c, f.applyAsDouble(c));
    });
    return this;
  }

  public DenseSynapseLayerGPU setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.weights.getData(), i -> f.getAsDouble());
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList(this.weights.getData());
  }

}
