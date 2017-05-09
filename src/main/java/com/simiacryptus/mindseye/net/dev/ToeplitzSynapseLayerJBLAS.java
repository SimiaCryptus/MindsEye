package com.simiacryptus.mindseye.net.dev;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Coordinate;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;

public class ToeplitzSynapseLayerJBLAS extends NNLayer {



  private final class Result extends NNResult {
    private final NNResult result;

    private Result(final Tensor[] data, final NNResult result) {
      super(data);
      this.result = result;
    }

    private Tensor[] backprop(final Tensor[] delta, final DeltaSet buffer) {
      double[] expandedWeights = getExpandedWeights();
      Tensor[] passbackA = java.util.stream.IntStream.range(0, result.data.length).parallel().mapToObj(dataIndex->{
        final double[] deltaData = delta[dataIndex].getData();
        final Tensor passback = new Tensor(this.result.data[dataIndex].getDims());
        DenseSynapseLayerJBLAS.multiplyT(expandedWeights, deltaData, passback.getData());
        return passback;
      }).toArray(i->new Tensor[i]);
      this.result.accumulate(buffer, passbackA);
      return passbackA;
    }

    @Override
    public void accumulate(final DeltaSet buffer, final Tensor[] delta) {
      if (!isFrozen()) {
        learn(delta, buffer);
      }
      if (this.result.isAlive()) {
        backprop(delta, buffer);
      }
    }

    @Override
    public boolean isAlive() {
      return this.result.isAlive() || !isFrozen();
    }

    private void learn(final Tensor[] delta, final DeltaSet buffer) {
      final double[] deltaData0 = delta[0].getData();
      final double[] inputData0 = this.result.data[0].getData();
      java.util.stream.IntStream.range(0, result.data.length).parallel().forEach(dataIndex->{
        Tensor buffer1 = new Tensor(ToeplitzSynapseLayerJBLAS.this.weights.getDims());
        final Tensor buffer2 = new Tensor(inputData0.length, deltaData0.length);
        final double[] deltaData = delta[dataIndex].getData();
        final double[] inputData = this.result.data[dataIndex].getData();
        assert(deltaData0.length == deltaData.length);
        assert(inputData0.length == inputData.length);
        DenseSynapseLayerJBLAS.crossMultiply(deltaData, inputData, buffer2.getData());
        buffer1.setAll(0.0);
        getCompactedWeights(buffer2.getData(), buffer1);
        buffer.get(ToeplitzSynapseLayerJBLAS.this, weights).accumulate(buffer1.getData());
        try {
          buffer1.finalize();
          buffer2.finalize();
        } catch (Throwable e) {
          throw new RuntimeException(e);
        }
      });
    }

  }

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ToeplitzSynapseLayerJBLAS.class);

  /**
   *
   */
  private static final long serialVersionUID = 3538627887600182889L;

  public final int[] outputDims;
  public final int[] inputDims;
  private final Tensor weights;
  private int radius;
  private final int outputSize;
  private final int inputSize;
  private final int[] mappingMatrix;


  protected ToeplitzSynapseLayerJBLAS() {
    super();
    this.outputDims = null;
    this.weights = null;
    this.inputDims = null;
    this.outputSize = 0;
    this.inputSize = 0;
    this.radius = 0;
    this.mappingMatrix = null;
  }

  public ToeplitzSynapseLayerJBLAS(final int[] inputDims, final int[] outputDims) {
    this(inputDims,outputDims,Integer.MAX_VALUE);
  }

  public ToeplitzSynapseLayerJBLAS(final int[] inputDims, final int[] outputDims, int radius) {
    assert(inputDims.length <= outputDims.length);
    int[] weightDims = new int[outputDims.length];
    for(int i=0;i<weightDims.length;i++) weightDims[i] = (i<inputDims.length?inputDims[i]:1) + outputDims[i] - 1;
    this.outputDims = Arrays.copyOf(outputDims, outputDims.length);
    this.inputDims = Arrays.copyOf(inputDims, inputDims.length);
    this.weights = new Tensor(weightDims);
    final int inputs = Tensor.dim(inputDims);
    final int outs = Tensor.dim(outputDims);
    setWeights(() -> {
      double ratio = Math.sqrt(6. / (inputs + outs));
      double fate = Util.R.get().nextDouble();
      double v = (1 - 2 * fate) * ratio;
      return v;
    });
    Tensor inputPrototype = new Tensor(inputDims);
    Tensor outputPrototype = new Tensor(outputDims);
    this.inputSize = inputPrototype.dim();
    this.outputSize = outputPrototype.dim();
    this.mappingMatrix = new int[this.inputSize*this.outputSize];
    this.radius = radius;
    int[] coordVector = new int[inputDims.length];
    int[] spareVector = new int[outputDims.length-inputDims.length];
    inputPrototype.coordStream().forEach(inputCoord->{
      outputPrototype.coordStream().forEach(outputCoord->{
        for(int i=0;i<coordVector.length;i++) {
          coordVector[i] = inputCoord.coords[i] - outputCoord.coords[i] + (outputDims[i] - 1);
        }
        for(int i=0;i<spareVector.length;i++) {
          spareVector[i] = outputCoord.coords[i+coordVector.length];
        }
        int mappedIndex = allowVector(coordVector) ? weights.index(concat(coordVector, spareVector)) : -1;
        assert(mappedIndex < weights.dim());
        mappingMatrix[inputCoord.index + inputSize * outputCoord.index] = mappedIndex;
      });
    });
  }

  private int[] concat(int[] a, int[] b) {
    int[] c = new int[a.length + b.length];
    for(int i=0;i<a.length;i++) c[i] = a[i];
    for(int i=0;i<b.length;i++) c[i+a.length] = b[i];
    return c;
  }

  private boolean allowVector(int[] coordVector) {
    return Arrays.stream(coordVector).sum() < radius;
  }

  public ToeplitzSynapseLayerJBLAS addWeights(final DoubleSupplier f) {
    Util.add(f, weights.getData());
    return this;
  }

  @Override
  public NNResult eval(final NNResult... input) {
    double[] expandedWeights = getExpandedWeights();
    Tensor[] outputA = java.util.stream.IntStream.range(0, input[0].data.length).parallel().mapToObj(dataIndex->{
      final Tensor inputTensor = input[0].data[dataIndex];
      return multiply2(expandedWeights, inputTensor.getData());
    }).toArray(i->new Tensor[i]);
    return new Result(outputA, input[0]);
  }

  private double[] getExpandedWeights() {
    double[] matrix = new double[mappingMatrix.length];
    double[] data = weights.getData();
    for(int i=0;i<matrix.length;i++) {
      int mappedIndex = this.mappingMatrix[i];
      matrix[i] = (mappedIndex >= 0)? data[mappedIndex]:0;
    }
    return matrix;
  }

  private void getCompactedWeights(double[] source, Tensor target) {
    assert(source.length==this.mappingMatrix.length);
    double[] data = target.getData();
    for(int i=0;i<source.length;i++) {
      int mappedIndex = this.mappingMatrix[i];
      if(mappedIndex >= 0) {
        data[mappedIndex] += source[i];
      }
    }
  }

  private Tensor multiply2(final double[] wdata, final double[] indata) {
    final Tensor output = new Tensor(this.outputDims);
    DenseSynapseLayerJBLAS.multiply(wdata, indata, output.getData());
    return output;
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.addProperty("weights", weights.toString());
    return json;
  }

  protected double getMobility() {
    return 1;
  }

  public ToeplitzSynapseLayerJBLAS setWeights(final double[] data) {
    this.weights.set(data);
    return this;
  }

  public ToeplitzSynapseLayerJBLAS setWeights(final java.util.function.ToDoubleFunction<Coordinate> f) {
    weights.coordStream().parallel().forEach(c->{
      weights.set(c, f.applyAsDouble(c));
    });
    return this;
  }

  public ToeplitzSynapseLayerJBLAS setWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.weights.getData(), i -> f.getAsDouble());
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList(weights.getData());
  }

}
