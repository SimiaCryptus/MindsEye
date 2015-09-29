package com.simiacryptus.mindseye.net.media;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.Coordinate;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.dag.EvaluationContext;

import groovy.lang.Tuple2;

public class MaxSubsampleLayer extends NNLayer<MaxSubsampleLayer> {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxSubsampleLayer.class);

  private int[] kernelDims;

  protected MaxSubsampleLayer() {
    super();
  }

  public MaxSubsampleLayer(final int... kernelDims) {

    this.kernelDims = Arrays.copyOf(kernelDims, kernelDims.length);
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj) {
    final NDArray input = inObj[0].data;
    final int[] inputDims = input.getDims();
    final int[] newDims = IntStream.range(0, inputDims.length).map(i -> {
      assert 0 == inputDims[i] % this.kernelDims[i];
      return inputDims[i] / this.kernelDims[i];
    }).toArray();
    final NDArray output = new NDArray(newDims);
    final HashMap<Coordinate, Coordinate> gradientMap = new HashMap<Coordinate, Coordinate>();
    List<Tuple2<Coordinate, List<Coordinate>>> regions = calcRegionsCache.apply(new CalcRegionsParameter(inputDims, kernelDims));
    regions.stream().forEach(tuple -> {
      final Coordinate inputCoord = tuple.getSecond().stream().max(Comparator.comparing(inputCoords -> input.get(inputCoords))).get();
      Coordinate o = tuple.getFirst();
      gradientMap.put(o, inputCoord);
      output.add(o, input.get(inputCoord));
    });
    return new NNResult(evaluationContext, output) {
      @Override
      public void feedback(final NDArray data, final DeltaBuffer buffer) {
        if (inObj[0].isAlive()) {
          final NDArray backSignal = new NDArray(inputDims);
          gradientMap.entrySet().forEach(e -> backSignal.add(e.getValue().index, data.get(e.getKey().index)));
          inObj[0].feedback(backSignal, buffer);
        }
      }

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
    };
  }

  public static class CalcRegionsParameter {
    public int[] inputDims;
    public int[] kernelDims;

    public CalcRegionsParameter(int[] inputDims, int[] kernelDims) {
      this.inputDims = inputDims;
      this.kernelDims = kernelDims;
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + Arrays.hashCode(inputDims);
      result = prime * result + Arrays.hashCode(kernelDims);
      return result;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj)
        return true;
      if (obj == null)
        return false;
      if (getClass() != obj.getClass())
        return false;
      CalcRegionsParameter other = (CalcRegionsParameter) obj;
      if (!Arrays.equals(inputDims, other.inputDims))
        return false;
      if (!Arrays.equals(kernelDims, other.kernelDims))
        return false;
      return true;
    }

  }

  private static final java.util.function.Function<CalcRegionsParameter, List<Tuple2<Coordinate, List<Coordinate>>>> calcRegionsCache = ConvolutionSynapseLayer
      .cache(MaxSubsampleLayer::calcRegions);

  private static List<Tuple2<Coordinate, List<Coordinate>>> calcRegions(CalcRegionsParameter parameterObject) {
    final NDArray input = new NDArray(parameterObject.inputDims);
    final int[] newDims = IntStream.range(0, parameterObject.inputDims.length).map(i -> {
      assert 0 == parameterObject.inputDims[i] % parameterObject.kernelDims[i];
      return parameterObject.inputDims[i] / parameterObject.kernelDims[i];
    }).toArray();
    final NDArray output = new NDArray(newDims);

    List<Tuple2<Coordinate, List<Coordinate>>> regions = output.coordStream(false).map(o -> {
      List<Coordinate> inCoords = new NDArray(parameterObject.kernelDims).coordStream(false).map(kernelCoord -> {
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

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
