package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.learning.DeltaBuffer;
import com.simiacryptus.mindseye.learning.DeltaFlushBuffer;
import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.math.Coordinate;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.LogNumber;
import com.simiacryptus.mindseye.math.NDArray;

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
      if (this == obj) return true;
      if (obj == null) return false;
      if (getClass() != obj.getClass()) return false;
      final IndexMapKey other = (IndexMapKey) obj;
      if (!Arrays.equals(this.input, other.input)) return false;
      if (!Arrays.equals(this.kernel, other.kernel)) return false;
      if (!Arrays.equals(this.output, other.output)) return false;
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
            if (inputCoords[d] < 0) return null;
            if (inputCoords[d] >= input.getDims()[d]) return null;
          }
          final int input_index = input.index(inputCoords);
          return new int[] {
              k.index,
              input_index,
              o.index
          };
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

  private DeltaFlushBuffer writer;
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
    this.writer = new DeltaFlushBuffer(this.kernel);
  }
  
  public ConvolutionSynapseLayer addWeights(final DoubleSupplier f) {
    Util.add(f, this.kernel.getData());
    return this;
  }
  
  @Override
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final int[] inputDims = input.getDims();
    final int[] kernelDims = this.kernel.getDims();
    final int[] newDims = IntStream.range(0, kernelDims.length).map(
        i -> i == kernelDims.length - 1 ? kernelDims[i] : inputDims[i] - kernelDims[i] + 1).toArray();
    final NDArray output = new NDArray(newDims);
    Arrays.stream(ConvolutionSynapseLayer.getIndexMap(this.kernel, input, output)).forEach(array -> {
      assert Double.isFinite(this.kernel.getData()[array[0]]);
      assert Double.isFinite(input.getData()[array[1]]);
    });
    Arrays.stream(ConvolutionSynapseLayer.getIndexMap(this.kernel, input, output)).forEach(array -> {
      output.add(array[2], input.getData()[array[1]] * this.kernel.getData()[array[0]]);
    });
    if (isVerbose()) {
      ConvolutionSynapseLayer.log.debug(String.format("Feed forward: %s * %s %n\t=> %s", inObj.data, this.kernel, output));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final LogNDArray errorSignal, DeltaBuffer buffer) {
        if (!ConvolutionSynapseLayer.this.frozen) {
          final LogNDArray weightGradient = new LogNDArray(ConvolutionSynapseLayer.this.kernel.getDims());
          Arrays.stream(ConvolutionSynapseLayer.getIndexMap(ConvolutionSynapseLayer.this.kernel, input, output)).forEach(array -> {
            double in = input.getData()[array[1]];
            LogNumber err = errorSignal.getData()[array[2]];
            weightGradient.add(array[0], err.multiply(in));
          });
          ConvolutionSynapseLayer.this.writer.feed(weightGradient.exp().getData());
        }
        if (inObj.isAlive()) {
          LogNDArray klog = ConvolutionSynapseLayer.this.kernel.log();
          final LogNDArray backprop = new LogNDArray(inputDims);
          
          Arrays.stream(ConvolutionSynapseLayer.getIndexMap(ConvolutionSynapseLayer.this.kernel, input, output)).forEach(array -> {
            final LogNumber kernelValue = klog.get(array[0]);
            if (kernelValue.isFinite())
            {
              final LogNumber errorValue = errorSignal.get(array[2]);
              backprop.add(array[1], errorValue.multiply(kernelValue));
            }
          });
          if (isVerbose()) {
            ConvolutionSynapseLayer.log.debug(String.format("Feed back: %s * -1 %n\t=> %s", errorSignal, backprop));
          }
          inObj.feedback(backprop, buffer);
        }
      }
      
      @Override
      public boolean isAlive() {
        return inObj.isAlive() || !isFrozen();
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

  public boolean isFrozen() {
    return this.frozen;
  }

  public boolean isParalell() {
    return this.paralell;
  }
  
  public boolean isVerbose() {
    return this.verbose;
  }

  public void setFrozen(final boolean frozen) {
    this.frozen = frozen;
  }
  
  public ConvolutionSynapseLayer setParalell(final boolean parallel) {
    this.paralell = parallel;
    return this;
  }

  public ConvolutionSynapseLayer setVerbose(final boolean verbose) {
    this.verbose = verbose;
    return this;
  }

  protected double getMobility() {
    return 1;
  }
}
