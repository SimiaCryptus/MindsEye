package com.simiacryptus.mindseye.net.media;

import java.util.Arrays;
import java.util.List;
import java.util.Map.Entry;
import java.util.concurrent.ExecutionException;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.simiacryptus.mindseye.core.Coordinate;
import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

public class SumSubsampleLayer extends NNLayer<SumSubsampleLayer> {

  public static final class IndexMapKey {
    int[] kernel;
    int[] output;

    public IndexMapKey(final int[] kernel, final int[] output) {
      super();
      this.kernel = kernel;
      this.output = output;
    }

    public IndexMapKey(final NDArray kernel, final NDArray input, final NDArray output) {
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
      if (!Arrays.equals(this.output, other.output))
        return false;
      return true;
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
          final java.util.Map<Coordinate, List<int[]>> coordMap = new NDArray(key.output).coordStream(false).collect(java.util.stream.Collectors.toMap(o -> o, o -> {
            return new NDArray(ksize).coordStream(false).map(kernelCoord -> {
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

  @Override
  public NNResult eval(final NNResult... inObj) {
    final int kernelSize = new NDArray(this.kernelDims).dim();
    final NDArray input = inObj[0].data;
    final int[] inputDims = input.getDims();
    final int[] newDims = IntStream.range(0, inputDims.length).map(i -> {
      assert 0 == inputDims[i] % this.kernelDims[i];
      return inputDims[i] / this.kernelDims[i];
    }).toArray();
    final NDArray output = new NDArray(newDims);
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
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data, final DeltaSet buffer) {
        if (inObj[0].isAlive()) {
          final NDArray backSignal = new NDArray(inputDims);
          for (final Entry<Coordinate, List<int[]>> outputMapping : coordMap.entrySet()) {
            final double outputValue = data.get(outputMapping.getKey());
            for (final int[] inputCoord : outputMapping.getValue()) {
              backSignal.add(inputCoord, outputValue / kernelSize);
            }
          }
          // output.coordStream(false).forEach(o -> {
          // double outV = output.get(o);
          // getKernelInputCoords(o).forEach(i->{
          // backSignal.add(i, outV/kernelSize);
          // });
          // });
          inObj[0].feedback(backSignal, buffer);
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
  // NDArray(this.kernelDims).coordStream(false).map(kernelCoord -> {
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
