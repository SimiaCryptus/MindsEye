package com.simiacryptus.mindseye.net.media;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.function.DoubleSupplier;
import java.util.function.Function;
import java.util.stream.IntStream;
import java.util.stream.Stream;

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

public class ConvolutionSynapseLayer extends NNLayer<ConvolutionSynapseLayer> {

  private static final Logger log = LoggerFactory.getLogger(ConvolutionSynapseLayer.class);

  
  public static class Convolve {
    
    public final int[] outputs;
    public final int[] script;

    public Convolve(int[] outputIndexes, int[] scripts) {
      this.outputs = outputIndexes;
      this.script = scripts;
    }

    public void convolve1(double[] input, double[] weights, double[] output) {
      for(int o=0;o<this.outputs.length;o++){
        int oo = this.outputs[o];
        double sum = 0;
        for(int s=0;s<this.script.length/2;s++){
          sum += input[oo+this.script[s*2]] * weights[this.script[s*2+1]];
        } 
        output[oo] = sum;
      }
    }

    public void calGradient(double[] input, double[] output, double[] weights) {
      for(int o=0;o<this.outputs.length;o++){
        for(int s=0;s<this.script.length/2;s++){
          int oo = this.outputs[o];
          weights[this.script[s*2+1]] += output[oo] * input[oo+this.script[s*2]];
        } 
      }
    }

    public void backprop(double[] output, double[] weights, double[] input) {
      for(int o=0;o<this.outputs.length;o++){
        for(int s=0;s<this.script.length/2;s++){
          int oo = this.outputs[o];
          input[oo+this.script[s*2]] += weights[this.script[s*2+1]] * output[oo];
        } 
      }
    }

    @Override
    public String toString() {
      StringBuilder builder = new StringBuilder();
      builder.append(String.format("Convolve (%sx%s) [outputs=", outputs.length, script.length));
      builder.append(Arrays.toString(outputs));
      builder.append(", script=");
      builder.append(Arrays.toString(script));
      builder.append("]");
      return builder.toString();
    }
    
    
  }
  
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
  public static <F,T> java.util.function.Function<F,T> cache(java.util.function.Function<F,T> inner) {
    LoadingCache<F,T> cache = CacheBuilder.newBuilder().build(new CacheLoader<F, T>(){
      @Override
      public T load(F key) throws Exception {
        return inner.apply(key);
      }});
    return cache::apply;
  }

  public static final java.util.function.Function<IndexMapKey, List<Convolve>> indexMapCache = cache((IndexMapKey key) -> {
    
    int outDim = new NDArray(key.output).dim();
    int inDim = new NDArray(key.input).dim();
    log.debug(String.format("%s ins * %s bands => %s outs", inDim, Arrays.toString(key.kernel), outDim));
    
    final NDArray kernel = new NDArray(key.kernel);
    Stream<int[]> productTuples = kernel.coordStream(false).flatMap(k -> {
      final NDArray output = new NDArray(key.output);
      return output.coordStream(false).map(o -> {
        final NDArray input = new NDArray(key.input);
        if(o.coords[o.coords.length-1]!=o.coords[o.coords.length-1]) {
          return null;
        }else {
          final int[] add = Coordinate.add(k.coords, o.coords);
          final int[] inputCoords = Arrays.copyOfRange(add, 0, add.length-1);
          for (int d = 0; d < input.getDims().length; d++) {
            if (inputCoords[d] < 0)
              return null;
            if (inputCoords[d] >= input.getDims()[d])
              return null;
          }
          final Coordinate input_index = new Coordinate(input.index(inputCoords), inputCoords);
          int[] arr = new int[] { k.index, input_index.index, o.index };
          //log.debug(String.format("key %s: i=%s k=%s o=%s", key, input_index, k, o));
          return arr;
        }
      }).filter(x->null!=x);
    });

    Comparator<int[]> orderBy = new Comparator<int[]>() {
      @Override
      public int compare(int[] o1, int[] o2) {
        int r = 0;
        for (int idx = 0; 0 == r && idx < o1.length; idx++)
          r = Integer.compare(o1[idx], o2[idx]);
        return r;
      }
    };

    int[][] array = productTuples.filter(x -> null != x).sorted(orderBy).toArray(i -> new int[i][]);
    
    Map<Set<List<Integer>>, Set<Integer>> collect = Stream.of(array) //
        .map(a->new int[]{a[2],a[1]-a[2], a[0]}) //
        .collect(java.util.stream.Collectors.groupingBy(a->a[0], // 
            java.util.stream.Collectors.mapping(a->Arrays.<Integer>asList(a[1],a[2]),java.util.stream.Collectors.toSet()))) //
        .entrySet().stream().collect(java.util.stream.Collectors.groupingBy(e->e.getValue(), //
            java.util.stream.Collectors.mapping(e->e.getKey(), java.util.stream.Collectors.toSet())));
    
    Function<Entry<Set<List<Integer>>, Set<Integer>>, Convolve> mapper = e->{
      int[] outputIndexes = e.getValue().stream().mapToInt(x->x).sorted().toArray();
      // tuples of (in_idx-out_idx,k_idx)
      int[] script = e.getKey().stream().sorted(Comparator.comparing(a->a.get(0))).flatMap(x->{
        assert(x.size()==2);
        return x.stream();
      }).mapToInt(x->x).toArray();
      Convolve k = new Convolve(outputIndexes, script);
      return k;
    };
    
    List<Convolve> kernels = collect.entrySet().stream().map(mapper).collect(java.util.stream.Collectors.toList());
    log.debug("Commputed kernels for " + key + ": " + kernels.stream().map(x->x.toString()).reduce((a,b)->a+"\n\t"+b).get());
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
    final int[] newDims = IntStream.range(0, kernelDims.length).map(i -> i == kernelDims.length - 1 ? kernelDims[i] : inputDims[i] - kernelDims[i] + 1).toArray();
    final NDArray output = new NDArray(newDims);
    final List<Convolve> indexMap = ConvolutionSynapseLayer.indexMapCache.apply(new IndexMapKey(this.kernel, input, output));
    double[] indata = input.getData();
    double[] kdata = this.kernel.getData();
    final double[] indata1 = indata;
    final double[] kdata1 = kdata;
    double[] outdata = output.getData();
    for(Convolve k : indexMap) {
      k.convolve1(indata1, kdata1, outdata);
    };
    if (isVerbose()) {
      ConvolutionSynapseLayer.log.debug(String.format("Feed forward: %s * %s %n\t=> %s", inObj[0].data, this.kernel, output));
    }
    return new NNResult(evaluationContext, output) {
      @Override
      public void feedback(final NDArray errorSignal, final DeltaBuffer buffer) {
        if (!isFrozen()) {
          final NDArray weightGradient = new NDArray(ConvolutionSynapseLayer.this.kernel.getDims());
          indexMap.stream().forEach(array -> {
            array.calGradient(indata,errorSignal.getData(),weightGradient.getData());
          });
          buffer.get(ConvolutionSynapseLayer.this, ConvolutionSynapseLayer.this.kernel).feed(weightGradient.getData());
        }
        if (inObj[0].isAlive()) {
          final NDArray klog = ConvolutionSynapseLayer.this.kernel;
          final NDArray backprop = new NDArray(inputDims);

          indexMap.stream().forEach(array -> {
            array.backprop(errorSignal.getData(), kernel.getData(), backprop.getData());
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
