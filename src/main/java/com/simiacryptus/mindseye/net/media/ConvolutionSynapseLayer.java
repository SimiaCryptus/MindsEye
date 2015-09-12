package com.simiacryptus.mindseye.net.media;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.Coordinate;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.dag.EvaluationContext;
import com.simiacryptus.mindseye.util.Util;

public class ConvolutionSynapseLayer extends NNLayer {
  public static final class IndexMapKey {
    int[] input;
    int[] kernel;
    int[] output;

    public IndexMapKey(final int[] kernel, final int[] input, final int[] output) {
      super();
      this.kernel = kernel;
      this.input = input;
      this.output = output;
    }

    public IndexMapKey(final NDArray kernel, final NDArray input, final NDArray output) {
      super();
      this.kernel = kernel.getDims();
      this.input = input.getDims();
      this.output = output.getDims();
    }

    @Override
    public boolean equals(final Object obj) {
      if (this == obj)
        return true;
      if (obj == null)
        return false;
      if (getClass() != obj.getClass())
        return false;
      final IndexMapKey other = (IndexMapKey) obj;
      if (!Arrays.equals(this.input, other.input))
        return false;
      if (!Arrays.equals(this.kernel, other.kernel))
        return false;
      if (!Arrays.equals(this.output, other.output))
        return false;
      return true;
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + Arrays.hashCode(this.input);
      result = prime * result + Arrays.hashCode(this.kernel);
      result = prime * result + Arrays.hashCode(this.output);
      return result;
    }
  }

  public static final LoadingCache<IndexMapKey, int[][]> indexMapCache = CacheBuilder.newBuilder().build(new CacheLoader<IndexMapKey, int[][]>() {
    @Override
    public int[][] load(final IndexMapKey key) throws Exception {
      final NDArray kernel = new NDArray(key.kernel);
      return kernel.coordStream(false).flatMap(k -> {
        final NDArray output = new NDArray(key.output);
        return output.coordStream(false).map(o -> {
          final NDArray input = new NDArray(key.input);
          final int[] inputCoords = Coordinate.add(k.coords, o.coords);
          for (int d = 0; d < input.getDims().length; d++) {
            if (inputCoords[d] < 0)
              return null;
            if (inputCoords[d] >= input.getDims()[d])
              return null;
          }
          final int input_index = input.index(inputCoords);
          return new int[] { k.index, input_index, o.index };
        });
      }).filter(x -> null != x).toArray(i -> new int[i][]);
    }
  });
  private static final Logger log = LoggerFactory.getLogger(ConvolutionSynapseLayer.class);

  public static int[][] getIndexMap(final NDArray kernel, final NDArray input, final NDArray output) {
    try {
      return ConvolutionSynapseLayer.indexMapCache.get(new IndexMapKey(kernel, input, output));
    } catch (final ExecutionException e) {
      throw new RuntimeException(e);
    }
  }

  private boolean frozen = false;
  public final NDArray kernel;
  private boolean paralell = false;
  private boolean verbose = false;

  protected ConvolutionSynapseLayer() {
    super();
    this.kernel = null;
  }

  public ConvolutionSynapseLayer(final int[] kernelDims, final int bandwidth) {

    final int[] kernelDims2 = Arrays.copyOf(kernelDims, kernelDims.length + 1);
    kernelDims2[kernelDims2.length - 1] = bandwidth;
    this.kernel = new NDArray(kernelDims2);
  }

  public ConvolutionSynapseLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.kernel.getData());
    return this;
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj) {
    assert 1 == inObj.length;
    final NDArray input = inObj[0].data;
    final int[] inputDims = input.getDims();
    final int[] kernelDims = this.kernel.getDims();
    final int[] newDims = IntStream.range(0, kernelDims.length).map(i -> i == kernelDims.length - 1 ? kernelDims[i] : inputDims[i] - kernelDims[i] + 1).toArray();
    final NDArray output = new NDArray(newDims);
    final int[][] indexMap = ConvolutionSynapseLayer.getIndexMap(this.kernel, input, output);
    Arrays.stream(indexMap).forEach(array -> {
      final int k = array[0];
      final int i = array[1];
      final int o = array[2];
      assert Double.isFinite(this.kernel.getData()[k]);
      assert Double.isFinite(input.getData()[i]);
      output.add(o, input.getData()[i] * this.kernel.getData()[k]);
    });
    if (isVerbose()) {
      ConvolutionSynapseLayer.log.debug(String.format("Feed forward: %s * %s %n\t=> %s", inObj[0].data, this.kernel, output));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray errorSignal, final DeltaBuffer buffer) {
        if (!isFrozen()) {
          final NDArray weightGradient = new NDArray(ConvolutionSynapseLayer.this.kernel.getDims());
          Arrays.stream(indexMap).forEach(array -> {
            final int i = array[1];
            final int o = array[2];
            final int k = array[0];
            final double in = input.getData()[i];
            final double err = errorSignal.getData()[o];
            weightGradient.add(k, err*(in));
          });
          buffer.get(ConvolutionSynapseLayer.this, ConvolutionSynapseLayer.this.kernel).feed(weightGradient.getData());
        }
        if (inObj[0].isAlive()) {
          final NDArray klog = ConvolutionSynapseLayer.this.kernel;
          final NDArray backprop = new NDArray(inputDims);

          Arrays.stream(indexMap).forEach(array -> {
            final int k = array[0];
            final int o = array[2];
            final int i = array[1];
            final double kernelValue = klog.get(k);
            if (Double.isFinite(kernelValue)) {
              final double errorValue = errorSignal.get(o);
              backprop.add(i, errorValue*(kernelValue));
            }
          });
          if (isVerbose()) {
            ConvolutionSynapseLayer.log.debug(String.format("Feed back: %s * -1 %n\t=> %s", errorSignal, backprop));
          }
          inObj[0].feedback(backprop, buffer);
        }
      }

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || !isFrozen();
      }
    };
  }

  public ConvolutionSynapseLayer fillWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.kernel.getData(), i -> f.getAsDouble());
    return this;
  }

  public ConvolutionSynapseLayer freeze() {
    return freeze(true);
  }

  public ConvolutionSynapseLayer freeze(final boolean b) {
    this.frozen = b;
    return this;
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.addProperty("kernel", this.kernel.toString());
    return json;
  }

  protected double getMobility() {
    return 1;
  }

  public boolean isFrozen() {
    return this.frozen;
  }

  public boolean isParalell() {
    return this.paralell;
  }

  @Override
  public boolean isVerbose() {
    return this.verbose;
  }

  public void setFrozen(final boolean frozen) {
    this.frozen = frozen;
  }

  public ConvolutionSynapseLayer setParallel(final boolean parallel) {
    this.paralell = parallel;
    return this;
  }

  public ConvolutionSynapseLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList(this.kernel.getData());
  }
}
