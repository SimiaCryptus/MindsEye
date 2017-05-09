package com.simiacryptus.mindseye.net.media;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

import com.simiacryptus.util.ml.Coordinate;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.gson.JsonObject;
import com.simiacryptus.util.Util;
import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;
import com.simiacryptus.mindseye.opencl.ConvolutionController;

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
      assert 3 == input.length;
      assert 3 == kernel.length;
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
        return Arrays.equals(this.output, other.output);
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
      final StringBuilder builder = new StringBuilder();
      builder.append("IndexMapKey [input=");
      builder.append(Arrays.toString(this.input));
      builder.append(", kernel=");
      builder.append(Arrays.toString(this.kernel));
      builder.append(", output=");
      builder.append(Arrays.toString(this.output));
      builder.append("]");
      return builder.toString();
    }

  }

  private static final Logger log = LoggerFactory.getLogger(ConvolutionSynapseLayer.class);
  
  public static final java.util.function.Function<IndexMapKey, ConvolutionController> indexMapCache = cache((final IndexMapKey key) -> {

    final int outDim = new Tensor(key.output).dim();
    final int inDim = new Tensor(key.input).dim();
    log.debug(String.format("%s ins * %s bands => %s outs", inDim, Arrays.toString(key.kernel), outDim));

    assert 3 == key.input.length;
    assert 3 == key.kernel.length;
    final ConvolutionController kernels = new ConvolutionController(key.input, key.kernel);
    log.debug("Commputed kernels for " + key + ": " + kernels.toString());
    return kernels;

  });


  private static final long serialVersionUID = -139062498597441290L;

  @SuppressWarnings("deprecation")
  public static <F, T> java.util.function.Function<F, T> cache(final java.util.function.Function<F, T> inner) {
    final LoadingCache<F, T> cache = CacheBuilder.newBuilder().build(new CacheLoader<F, T>() {
      @Override
      public T load(final F key) throws Exception {
        return inner.apply(key);
      }
    });
    return cache::apply;
  }

  public static int[] getOutputDims(final int[] inputSize, final int[] kernelSize) {
    return IntStream.range(0, kernelSize.length).map(i -> {
      int x;
      if (i == kernelSize.length - 1) {
        x = kernelSize[i] / inputSize[i];
      } else {
        x = 1 + inputSize[i] - kernelSize[i];
      }
      if (0 >= x) {
        assert false;
      }
      return x;
    }).toArray();
  }

  public final Tensor kernel;

  private boolean paralell = false;

  protected ConvolutionSynapseLayer() {
    super();
    this.kernel = null;
  }

  public ConvolutionSynapseLayer(final int[] kernelDims, final int bandwidth) {
    final int[] kernelDims2 = Arrays.copyOf(kernelDims, kernelDims.length + 1);
    kernelDims2[kernelDims2.length - 1] = bandwidth;
    this.kernel = new Tensor(kernelDims2);
    assert 3 == this.kernel.getDims().length;
  }

  public ConvolutionSynapseLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.kernel.getData());
    return this;
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    final ConvolutionController indexMap;
    Tensor head = inObj[0].data[0];
    final int[] inputDims = head.getDims();
    {
      final int[] kernelDims = this.kernel.getDims();
      indexMap = ConvolutionSynapseLayer.indexMapCache.apply(new IndexMapKey(kernelDims, inputDims, getOutputDims(inputDims, kernelDims)));
    }
    Tensor[] outputA = java.util.stream.IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex->{
      final Tensor input = inObj[0].data[dataIndex];
      final int[] kernelDims = this.kernel.getDims();
      final Tensor output = new Tensor(getOutputDims(inputDims, kernelDims));
      indexMap.convolve(input.getData(), this.kernel.getData(), output.getData());
      return output;
    }).toArray(i->new Tensor[i]);

    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] errorSignal) {
        if (!isFrozen()) {
          java.util.stream.IntStream.range(0, inObj[0].data.length).forEach(dataIndex->{
            final Tensor input = inObj[0].data[dataIndex];
            final Tensor kernel = ConvolutionSynapseLayer.this.kernel;
            final Tensor weightGradient = new Tensor(kernel.getDims());
            indexMap.gradient(input.getData(), weightGradient.getData(), errorSignal[dataIndex].getData());
            buffer.get(ConvolutionSynapseLayer.this, kernel).accumulate(weightGradient.getData());
          });
        }
        if (inObj[0].isAlive()) {
          Tensor[] passbackA = java.util.stream.IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex->{
            final Tensor backprop = new Tensor(inputDims);
            indexMap.backprop(backprop.getData(), ConvolutionSynapseLayer.this.kernel.getData(), errorSignal[dataIndex].getData());
            return backprop;
          }).toArray(i->new Tensor[i]);
          inObj[0].accumulate(buffer, passbackA);
        }
      }

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || !isFrozen();
      }
    };
  }

  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJson();
    json.addProperty("kernel", this.kernel.toString());
    return json;
  }

  public boolean isParalell() {
    return this.paralell;
  }

  public ConvolutionSynapseLayer setParallel(final boolean parallel) {
    this.paralell = parallel;
    return this;
  }

  public ConvolutionSynapseLayer setWeights(final ToDoubleFunction<Coordinate> f) {
    this.kernel.coordStream().parallel().forEach(c->{
      this.kernel.set(c, f.applyAsDouble(c));
    });
    return this;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList(this.kernel.getData());
  }
}
