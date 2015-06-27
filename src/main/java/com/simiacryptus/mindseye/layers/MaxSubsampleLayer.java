package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.Coordinate;
import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.NNLayer;
import com.simiacryptus.mindseye.learning.NNResult;

public class MaxSubsampleLayer extends NNLayer {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxSubsampleLayer.class);
  
  private int[] kernelDims;
  
  public MaxSubsampleLayer(int... kernelDims) {
    
    this.kernelDims = Arrays.copyOf(kernelDims, kernelDims.length);
  }
  
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final int[] inputDims = input.getDims();
    int[] newDims = IntStream.range(0, inputDims.length).map(
        i -> inputDims[i] / kernelDims[i]
        ).toArray();
    final NDArray output = new NDArray(newDims);
    HashMap<Coordinate, Coordinate> gradientMap = new HashMap<Coordinate, Coordinate>();
    output.coordStream().forEach(o -> {
      int[] i = new NDArray(kernelDims).coordStream()
          .map(k -> IntStream.range(0, k.coords.length).map(idx -> k.coords[idx] + kernelDims[idx] * o.coords[idx]).toArray())
          .sorted(Comparator.comparing(inputCoords -> input.get(inputCoords))).findFirst().get();
      Coordinate inputCoord = new Coordinate(input.index(i), i);
      gradientMap.put(o, inputCoord);
      output.add(o, input.get(inputCoord));
    });
    return new NNResult(output) {
      @Override
      public void feedback(NDArray data) {
        if (inObj.isAlive()) {
          NDArray backSignal = new NDArray(inputDims);
          gradientMap.entrySet().forEach(e->backSignal.add(e.getValue().coords, data.get(e.getKey().coords)));
          inObj.feedback(backSignal);
        }
      }
      
      public boolean isAlive() {
        return true;
      }
    };
  }
}
