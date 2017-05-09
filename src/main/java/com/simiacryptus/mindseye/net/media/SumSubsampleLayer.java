package com.simiacryptus.mindseye.net.media;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ExecutionException;
import java.util.stream.IntStream;

import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.simiacryptus.util.ml.Coordinate;
import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

public class SumSubsampleLayer extends NNLayer {

  public static final class IndexMapKey {
    int[] kernel;
    int[] output;

    public IndexMapKey(final int[] kernel, final int[] output) {
      super();
      this.kernel = kernel;
      this.output = output;
    }

    public IndexMapKey(final Tensor kernel, final Tensor input, final Tensor output) {
      super();
      this.kernel = kernel.getDims();
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
      if (!Arrays.equals(this.kernel, other.kernel))
        return false;
        return Arrays.equals(this.output, other.output);
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + Arrays.hashCode(this.kernel);
      result = prime * result + Arrays.hashCode(this.output);
      return result;
    }
  }

  public static final LoadingCache<IndexMapKey, java.util.Map<Coordinate, List<int[]>>> indexMapCache = CacheBuilder.newBuilder()
      .build(new CacheLoader<IndexMapKey, java.util.Map<Coordinate, List<int[]>>>() {
        @Override
        public java.util.Map<Coordinate, List<int[]>> load(final IndexMapKey key) throws Exception {
          final int[] ksize = key.kernel;
          final java.util.Map<Coordinate, List<int[]>> coordMap = new Tensor(key.output).coordStream(false).collect(java.util.stream.Collectors.toMap(o -> o, o -> {
            return new Tensor(ksize).coordStream(false).map(kernelCoord -> {
              final int[] r = new int[o.coords.length];
              for (int i = 0; i < o.coords.length; i++) {
                r[i] = o.coords[i] * ksize[i] + kernelCoord.coords[i];
              }
              return r;
            }).collect(java.util.stream.Collectors.toList());
          }));
          return coordMap;
        }
      });

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SumSubsampleLayer.class);

  /**
   * 
   */
  private static final long serialVersionUID = 7441695931197085499L;

  private static java.util.Map<Coordinate, List<int[]>> getCoordMap(final int[] kernelDims, final int[] outDims) {
    try {
      return indexMapCache.get(new IndexMapKey(kernelDims, outDims));
    } catch (final ExecutionException e) {
      throw new RuntimeException(e);
    }
  }

  private int[] kernelDims;

  protected SumSubsampleLayer() {
    super();
  }

  public SumSubsampleLayer(final int... kernelDims) {

    this.kernelDims = Arrays.copyOf(kernelDims, kernelDims.length);
  }

  @SuppressWarnings("unchecked")
  @Override
  public NNResult eval(final NNResult... inObj) {
    final int kernelSize = new Tensor(this.kernelDims).dim();
    final int[] inputDims = inObj[0].data[0].getDims();
    int itemCnt = inObj[0].data.length;
    final java.util.Map<Coordinate, List<int[]>> coordMapA[] = new Map[itemCnt];
    Tensor[] outputA = java.util.stream.IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex->{
      final Tensor input = inObj[0].data[dataIndex];
      final int[] newDims = IntStream.range(0, inputDims.length).map(i -> {
        if(!(0 == inputDims[i] % this.kernelDims[i])){
          assert(false);
        }
        return inputDims[i] / this.kernelDims[i];
      }).toArray();
      final Tensor output = new Tensor(newDims);
      final java.util.Map<Coordinate, List<int[]>> coordMap = getCoordMap(this.kernelDims, output.getDims());
      for (final Entry<Coordinate, List<int[]>> outputMapping : coordMap.entrySet()) {
        double sum = 0;
        for (final int[] inputCoord : outputMapping.getValue()) {
          sum += input.get(inputCoord);
        }
        if (Double.isFinite(sum)) {
          output.add(outputMapping.getKey(), sum / kernelSize);
        }
      }
      coordMapA[dataIndex] = coordMap;
      return output;
    }).toArray(i->new Tensor[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (inObj[0].isAlive()) {
          Tensor[] passbackA = java.util.stream.IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex->{
            final Tensor backSignal = new Tensor(inputDims);
            for (final Entry<Coordinate, List<int[]>> outputMapping : coordMapA[dataIndex].entrySet()) {
              final double outputValue = data[dataIndex].get(outputMapping.getKey());
              for (final int[] inputCoord : outputMapping.getValue()) {
                backSignal.add(inputCoord, outputValue / kernelSize);
              }
            }
            return backSignal;
          }).toArray(i->new Tensor[i]);
          inObj[0].accumulate(buffer, passbackA);
        }
      }

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
    };
  }

  // public Stream<int[]> getKernelInputCoords(Coordinate outCoord) {
  // List<int[]> kernelCoords = new
  // Tensor(this.kernelDims).coordStream(false).map(kernelCoord -> {
  // final int[] r = new int[outCoord.coords.length];
  // for (int i1 = 0; i1 < outCoord.coords.length; i1++) {
  // r[i1] = outCoord.coords[i1] * this.kernelDims[i1] + kernelCoord.coords[i1];
  // }
  // return r;
  // }).collect(java.util.stream.Collectors.toList());
  // Stream<int[]> stream = kernelCoords.stream();
  // return stream;
  // }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
