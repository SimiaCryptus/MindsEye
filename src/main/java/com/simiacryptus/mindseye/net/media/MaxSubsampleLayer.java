package com.simiacryptus.mindseye.net.media;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.core.Coordinate;
import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

import groovy.lang.Tuple2;

public class MaxSubsampleLayer extends NNLayer<MaxSubsampleLayer> {
  public static class CalcRegionsParameter {
    public int[] inputDims;
    public int[] kernelDims;

    public CalcRegionsParameter(final int[] inputDims, final int[] kernelDims) {
      this.inputDims = inputDims;
      this.kernelDims = kernelDims;
    }

    @Override
    public boolean equals(final Object obj) {
      if (this == obj)
        return true;
      if (obj == null)
        return false;
      if (getClass() != obj.getClass())
        return false;
      final CalcRegionsParameter other = (CalcRegionsParameter) obj;
      if (!Arrays.equals(this.inputDims, other.inputDims))
        return false;
      if (!Arrays.equals(this.kernelDims, other.kernelDims))
        return false;
      return true;
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + Arrays.hashCode(this.inputDims);
      result = prime * result + Arrays.hashCode(this.kernelDims);
      return result;
    }

  }

  private static final java.util.function.Function<CalcRegionsParameter, List<Tuple2<Coordinate, List<Coordinate>>>> calcRegionsCache = ConvolutionSynapseLayer
      .cache(MaxSubsampleLayer::calcRegions);

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxSubsampleLayer.class);

  /**
   * 
   */
  private static final long serialVersionUID = -4486788592198117530L;

  private static List<Tuple2<Coordinate, List<Coordinate>>> calcRegions(final CalcRegionsParameter parameterObject) {
    final NDArray input = new NDArray(parameterObject.inputDims);
    final int[] newDims = IntStream.range(0, parameterObject.inputDims.length).map(i -> {
      assert 0 == parameterObject.inputDims[i] % parameterObject.kernelDims[i];
      return parameterObject.inputDims[i] / parameterObject.kernelDims[i];
    }).toArray();
    final NDArray output = new NDArray(newDims);

    final List<Tuple2<Coordinate, List<Coordinate>>> regions = output.coordStream(false).map(o -> {
      final List<Coordinate> inCoords = new NDArray(parameterObject.kernelDims).coordStream(false).map(kernelCoord -> {
        final int[] r = new int[o.coords.length];
        for (int i1 = 0; i1 < o.coords.length; i1++) {
          r[i1] = o.coords[i1] * parameterObject.kernelDims[i1] + kernelCoord.coords[i1];
        }
        return new Coordinate(input.index(r), r);
      }).collect(java.util.stream.Collectors.toList());
      return new groovy.lang.Tuple2<>(o, inCoords);
    }).collect(java.util.stream.Collectors.toList());
    return regions;
  }

  private int[] kernelDims;

  protected MaxSubsampleLayer() {
    super();
  }

  public MaxSubsampleLayer(final int... kernelDims) {

    this.kernelDims = Arrays.copyOf(kernelDims, kernelDims.length);
  }

  @Override
  public NNResult eval(final NNResult... inObj) {

    int itemCnt = inObj[0].data.length;
    final HashMap<Coordinate, Coordinate> gradientMapA[] = new HashMap[itemCnt];

    final int[] inputDims = inObj[0].data[0].getDims();
    NDArray[] outputA = java.util.stream.IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex->{
      final NDArray input = inObj[0].data[dataIndex];
      final int[] newDims = IntStream.range(0, inputDims.length).map(i -> {
        assert 0 == inputDims[i] % this.kernelDims[i];
        return inputDims[i] / this.kernelDims[i];
      }).toArray();
      final NDArray output = new NDArray(newDims);
      final HashMap<Coordinate, Coordinate> gradientMap = new HashMap<Coordinate, Coordinate>();
      final List<Tuple2<Coordinate, List<Coordinate>>> regions = calcRegionsCache.apply(new CalcRegionsParameter(inputDims, this.kernelDims));
      final ToDoubleFunction<? super Coordinate> keyExtractor = inputCoords -> input.get(inputCoords);
      regions.stream().parallel().forEach(tuple -> {
        final Coordinate inputCoord = tuple.getSecond().stream().max(Comparator.comparingDouble(keyExtractor)).get();
        final Coordinate o = tuple.getFirst();
        synchronized (gradientMap) {
          gradientMap.put(o, inputCoord);
        }
        output.set(o, input.get(inputCoord));
      });
      gradientMapA[dataIndex] = gradientMap;
      return output;
    }).toArray(i->new NDArray[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final NDArray[] data) {
        if (inObj[0].isAlive()) {
          NDArray[] passbackA = java.util.stream.IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex->{
            final NDArray backSignal = new NDArray(inputDims);
            gradientMapA[dataIndex].entrySet().forEach(e -> backSignal.add(e.getValue().index, data[dataIndex].get(e.getKey().index)));
            return backSignal;
          }).toArray(i->new NDArray[i]);
          inObj[0].accumulate(buffer, passbackA);
        }
      }

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
    };
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
