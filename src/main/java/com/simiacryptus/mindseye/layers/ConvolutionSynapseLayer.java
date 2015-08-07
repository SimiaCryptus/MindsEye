package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.concurrent.ExecutionException;
import java.util.function.DoubleSupplier;
import java.util.stream.IntStream;
import java.util.stream.StreamSupport;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.simiacryptus.mindseye.Coordinate;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.Util;
import com.simiacryptus.mindseye.learning.DeltaInversionBuffer;
import com.simiacryptus.mindseye.learning.DeltaMassMomentum;
import com.simiacryptus.mindseye.learning.DeltaFlushBuffer;
import com.simiacryptus.mindseye.learning.DeltaMemoryWriter;
import com.simiacryptus.mindseye.learning.DeltaTransaction;
import com.simiacryptus.mindseye.learning.GradientDescentAccumulator;
import com.simiacryptus.mindseye.learning.MassParameters;
import com.simiacryptus.mindseye.learning.NNResult;

public class ConvolutionSynapseLayer extends NNLayer implements MassParameters<ConvolutionSynapseLayer>, DeltaTransaction {
  private static final Logger log = LoggerFactory.getLogger(ConvolutionSynapseLayer.class);
  
  public final NDArray kernel;
  private DeltaMassMomentum massMomentum;
  private GradientDescentAccumulator deltaBuffer;
  private boolean verbose = false;
  private boolean frozen = false;
  private DeltaFlushBuffer flush;
//  NDArray _inputGradient;
  private boolean paralell = false;
  
  protected ConvolutionSynapseLayer() {
    super();
    kernel = null;
  }

  public ConvolutionSynapseLayer(final int[] kernelDims, final int bandwidth) {
    
    final int[] kernelDims2 = Arrays.copyOf(kernelDims, kernelDims.length + 1);
    kernelDims2[kernelDims2.length - 1] = bandwidth;
    this.kernel = new NDArray(kernelDims2);
    DeltaMemoryWriter writer = new DeltaMemoryWriter(this.kernel);
    this.massMomentum = new DeltaMassMomentum(writer);
    this.flush = new DeltaFlushBuffer(this.massMomentum);
    this.deltaBuffer = new GradientDescentAccumulator(this.flush);
  }

  @Override
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final int[] inputDims = input.getDims();
    final int[] kernelDims = this.kernel.getDims();
    final int[] newDims = IntStream.range(0, kernelDims.length).map(
        i -> (i == kernelDims.length - 1) ? kernelDims[i] : (inputDims[i] - kernelDims[i] + 1)).toArray();
    final NDArray output = new NDArray(newDims);
    Arrays.stream(getIndexMap(kernel, input, output)).forEach(array->{
      assert Double.isFinite(this.kernel.getData()[array[0]]);
      assert Double.isFinite(input.getData()[array[1]]);
    });
    Arrays.stream(getIndexMap(kernel, input, output)).forEach(array->{
      output.add(array[2], input.getData()[array[1]] * this.kernel.getData()[array[0]]);
    });
    if (isVerbose()) {
      log.debug(String.format("Feed forward: %s * %s %n\t=> %s", inObj.data, this.kernel, output));
    }
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray errorSignal) {
        if (!frozen) {
          final NDArray weightGradient = new NDArray(kernel.dim(), output.dim());
          Arrays.stream(getIndexMap(kernel, input, output)).forEach(array->{
            weightGradient.add(array[3], input.getData()[array[1]]);
          });
          ConvolutionSynapseLayer.this.deltaBuffer.feed(weightGradient, errorSignal.getData());
        }
        if (inObj.isAlive()) {
          NDArray backprop = new NDArray(inputDims);
          
          Arrays.stream(getIndexMap(kernel, input, output)).forEach(array->{
            final double kernelValue = kernel.get(array[0]);
            if(0. != kernelValue)
            {
              double errorValue = errorSignal.get(array[2]);
              backprop.add(array[1], errorValue*kernelValue);
            }
          });
          if (isVerbose()) {
            log.debug(String.format("Feed back: %s * -1 %n\t=> %s", errorSignal, backprop));
          }
          inObj.feedback(backprop);
        }
      }
      
      @Override
      public boolean isAlive() {
        return !frozen || inObj.isAlive();
      }
    };
  }

  public static final class IndexMapKey {
    int[] kernel;
    int[] input;
    int[] output;
    public IndexMapKey(int[] kernel, int[] input, int[] output) {
      super();
      this.kernel = kernel;
      this.input = input;
      this.output = output;
    }
    public IndexMapKey(NDArray kernel, NDArray input, NDArray output) {
      super();
      this.kernel = kernel.getDims();
      this.input = input.getDims();
      this.output = output.getDims();
    }
    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + Arrays.hashCode(input);
      result = prime * result + Arrays.hashCode(kernel);
      result = prime * result + Arrays.hashCode(output);
      return result;
    }
    @Override
    public boolean equals(Object obj) {
      if (this == obj) return true;
      if (obj == null) return false;
      if (getClass() != obj.getClass()) return false;
      IndexMapKey other = (IndexMapKey) obj;
      if (!Arrays.equals(input, other.input)) return false;
      if (!Arrays.equals(kernel, other.kernel)) return false;
      if (!Arrays.equals(output, other.output)) return false;
      return true;
    }
  }
  
  public static final LoadingCache<IndexMapKey, int[][]> indexMapCache = CacheBuilder.newBuilder().build(new CacheLoader<IndexMapKey, int[][]>(){
    @Override
    public int[][] load(IndexMapKey key) throws Exception {
      NDArray kernel = new NDArray(key.kernel);
      return kernel.coordStream(false).flatMap(k -> {
        NDArray output = new NDArray(key.output);
        return output.coordStream(false).map(o -> {
          NDArray input = new NDArray(key.input);
          NDArray gradientSize = new NDArray(kernel.dim(), output.dim());
          int[] inputCoords = Coordinate.add(k.coords, o.coords);
          for(int d=0;d<input.getDims().length;d++){
            if(inputCoords[d]<0) return null;
            if(inputCoords[d]>=input.getDims()[d]) return null;
          }
          int input_index = input.index(inputCoords);
          return new int[]{ 
              k.index, 
              input_index, 
              o.index,
              gradientSize.index(k.index, o.index)
            };
        });
      }).filter(x->null!=x).toArray(i->new int[i][]);
    }});
  
  public static int[][] getIndexMap(NDArray kernel, final NDArray input, final NDArray output) {
    try {
      return indexMapCache.get(new IndexMapKey(kernel, input, output));
    } catch (ExecutionException e) {
      throw new RuntimeException(e);
    }
  }
  
  public ConvolutionSynapseLayer fillWeights(final DoubleSupplier f) {
    Arrays.parallelSetAll(this.kernel.getData(), i -> f.getAsDouble());
    return this;
  }
  
  public ConvolutionSynapseLayer addWeights(final DoubleSupplier f) {
    Util.add(f, kernel.getData());
    return this;
  }
  

  @Override
  public double getMass() {
    return this.massMomentum.getMass();
  }
  
  @Override
  public double getMomentumDecay() {
    return this.massMomentum.getMomentumDecay();
  }
  
  @Override
  public ConvolutionSynapseLayer setMass(final double mass) {
    this.massMomentum.setMass(mass);
    return this;
  }

  @Override
  public ConvolutionSynapseLayer setMomentumDecay(final double momentumDecay) {
    this.massMomentum.setMomentumDecay(momentumDecay);
    return this;
  }

  public boolean isVerbose() {
    return verbose;
  }

  public ConvolutionSynapseLayer setVerbose(boolean verbose) {
    this.verbose = verbose;
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
    return frozen;
  }

  public void setFrozen(boolean frozen) {
    this.frozen = frozen;
  }
  
  @Override
  public void write(double factor) {
    if(isFrozen()) return;
    flush.write(factor);
  }

  public ConvolutionSynapseLayer setHalflife(final double halflife) {
    return setMomentumDecay(Math.exp(2 * Math.log(0.5) / halflife));
  }

  public boolean isParalell() {
    return paralell;
  }

  public ConvolutionSynapseLayer setParalell(boolean parallel) {
    this.paralell = parallel;
    return this;
  }

  @Override
  public void setRate(double rate) {
    this.massMomentum.setMass(1./rate);
  }

  @Override
  public double getRate() {
    return 1./this.massMomentum.getMass();
  }
}
