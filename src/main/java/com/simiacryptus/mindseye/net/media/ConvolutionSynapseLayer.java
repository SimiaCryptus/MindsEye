package com.simiacryptus.mindseye.net.media;

import java.util.Arrays;
import java.util.List;
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
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.dag.EvaluationContext;
import com.simiacryptus.mindseye.util.Util;

public class ConvolutionSynapseLayer extends NNLayer<ConvolutionSynapseLayer> {

  private static final Logger log = LoggerFactory.getLogger(ConvolutionSynapseLayer.class);

  public static final class IndexMapKey {
    int[] input;
    int[] kernel;
    int[] output;

    public IndexMapKey(final int[] kernel, final int[] input, final int[] output) {
      super();
      this.kernel = kernel;
      this.input = input;
      this.output = output;
      assert (3 == input.length);
      assert (3 == kernel.length);
    }

    public IndexMapKey(final NDArray kernel, final NDArray input, final NDArray output) {
      super();
      this.kernel = kernel.getDims();
      this.input = input.getDims();
      this.output = output.getDims();
      assert (3 == this.input.length);
      assert (3 == this.kernel.length);
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

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      builder.append("IndexMapKey [input=");
      builder.append(Arrays.toString(input));
      builder.append(", kernel=");
      builder.append(Arrays.toString(kernel));
      builder.append(", output=");
      builder.append(Arrays.toString(output));
      builder.append("]");
      return builder.toString();
    }

  }

  @SuppressWarnings("deprecation")
  public static <F, T> java.util.function.Function<F, T> cache(java.util.function.Function<F, T> inner) {
    LoadingCache<F, T> cache = CacheBuilder.newBuilder().build(new CacheLoader<F, T>() {
      @Override
      public T load(F key) throws Exception {
        return inner.apply(key);
      }
    });
    return cache::apply;
  }

  public static final java.util.function.Function<IndexMapKey, ConvolutionController> indexMapCache = cache((IndexMapKey key) -> {

    int outDim = new NDArray(key.output).dim();
    int inDim = new NDArray(key.input).dim();
    log.debug(String.format("%s ins * %s bands => %s outs", inDim, Arrays.toString(key.kernel), outDim));

    assert (3 == key.input.length);
    assert (3 == key.kernel.length);
    ConvolutionController kernels = new ConvolutionController(key.input, key.kernel);
    log.debug("Commputed kernels for " + key + ": " + kernels.toString());
    return kernels;

  });

  public final NDArray kernel;
  private boolean paralell = false;

  protected ConvolutionSynapseLayer() {
    super();
    this.kernel = null;
  }

  public ConvolutionSynapseLayer(final int[] kernelDims, final int bandwidth) {
    final int[] kernelDims2 = Arrays.copyOf(kernelDims, kernelDims.length + 1);
    kernelDims2[kernelDims2.length - 1] = bandwidth;
    this.kernel = new NDArray(kernelDims2);
    assert (3 == this.kernel.getDims().length);
  }

  public ConvolutionSynapseLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.kernel.getData());
    return this;
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final int[] inputDims = input.getDims();
    final int[] kernelDims = this.kernel.getDims();
    final NDArray output = new NDArray(getOutputDims(inputDims, kernelDims));
    final ConvolutionController indexMap = ConvolutionSynapseLayer.indexMapCache.apply(new IndexMapKey(this.kernel, input, output));
    indexMap.convolve(input.getData(), this.kernel.getData(), output.getData());
    if (isVerbose()) {
      //ConvolutionSynapseLayer.log.debug(String.format("Feed forward: %s * %s %n\t=> %s", inObj[0].data, this.kernel, output));
    }
    return new NNResult(evaluationContext, output) {
      @Override
      public void feedback(final NDArray errorSignal, final DeltaBuffer buffer) {
        if (!isFrozen()) {
          NDArray kernel = ConvolutionSynapseLayer.this.kernel;
          final NDArray weightGradient = new NDArray(kernel.getDims());
          indexMap.gradient(input.getData(), weightGradient.getData(), errorSignal.getData());
          buffer.get(ConvolutionSynapseLayer.this, kernel).feed(weightGradient.getData());
        }
        if (inObj[0].isAlive()) {
          final NDArray backprop = new NDArray(inputDims);
          indexMap.backprop(backprop.getData(), kernel.getData(), errorSignal.getData());
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

  public static int[] getOutputDims(final int[] inputSize, final int[] kernelSize) {
    return IntStream.range(0, kernelSize.length).map(i -> {
      int x;
      if (i == (kernelSize.length - 1)) {        
        x = kernelSize[i] / inputSize[i];
      } else {
        x = 1 + inputSize[i] - kernelSize[i];
      }
      if (0 >= x) {
        assert (false);
      }
      return x;
    }).toArray();
  }

  public ConvolutionSynapseLayer fillWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.kernel.getData(), i -> f.getAsDouble());
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

  public boolean isParalell() {
    return this.paralell;
  }

  public ConvolutionSynapseLayer setParallel(final boolean parallel) {
    this.paralell = parallel;
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList(this.kernel.getData());
  }
}
