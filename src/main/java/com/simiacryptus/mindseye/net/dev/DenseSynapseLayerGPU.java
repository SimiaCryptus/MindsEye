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
import com.simiacryptus.mindseye.net.basic.DenseSynapseLayer;
import com.simiacryptus.mindseye.opencl.MatrixMultiplyKernel;

public class DenseSynapseLayerGPU extends NNLayer {

  private final class Result extends NNResult {
    private final NNResult inObj;

    private Result(final Tensor[] outputA, final NNResult inObj) {
      super(outputA);
      this.inObj = inObj;
    }

    private Tensor[] backprop(final Tensor[] delta, final DeltaSet buffer) {
      Tensor[] passbackA = java.util.stream.IntStream.range(0, inObj.data.length).parallel().mapToObj(dataIndex->{
        final double[] deltaData = delta[dataIndex].getData();
        final Tensor r = DenseSynapseLayerGPU.this.getWeights();
        final Tensor passback = new Tensor(this.inObj.data[dataIndex].getDims());
        DenseSynapseLayer.multiplyT(r.getData(), deltaData, passback.getData());
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
        final Tensor weightDelta = new Tensor(inputData.length, deltaData.length);
        double[] weightData = weightDelta.getData();
        gradientCrossMatrix(deltaData, inputData, weightData);
        buffer.get(DenseSynapseLayerGPU.this, DenseSynapseLayerGPU.this.getWeights()).accumulate(weightData);
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

  private final Tensor weights;

  protected DenseSynapseLayerGPU() {
    super();
    this.outputDims = null;
    this.weights = null;
  }

  public DenseSynapseLayerGPU(final int inputs, final int[] outputDims) {
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    this.weights = new Tensor(inputs, Tensor.dim(outputDims));
    setWeights(() -> (1 - 2 * Util.R.get().nextDouble()) * Math.sqrt(6 / (inputs + Tensor.dim(outputDims))));
  }

  public DenseSynapseLayerGPU addWeights(final DoubleSupplier f) {
    Util.add(f, this.getWeights().getData());
    return this;
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    
    Tensor[] inputA = java.util.stream.IntStream.range(0, inObj[0].data.length).parallel()
        .mapToObj(dataIndex->inObj[0].data[dataIndex]).toArray(i->new Tensor[i]);
    Tensor[] outputA = java.util.stream.IntStream.range(0, inObj[0].data.length).parallel()
        .mapToObj(dataIndex->new Tensor(this.outputDims)).toArray(i->new Tensor[i]);
    double[][] inputAD = java.util.Arrays.stream(inputA).parallel().map(x->x.getData()).toArray(ii->new double[ii][]);
    double[][] outputAD = java.util.Arrays.stream(outputA).parallel().map(x->x.getData()).toArray(ii->new double[ii][]);
      MatrixMultiplyKernel.multiply(inputAD, this.getWeights().getData(), outputAD);
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
    return Arrays.asList(this.getWeights().getData());
  }

  public final Tensor getWeights() {
    return weights;
  }

}
