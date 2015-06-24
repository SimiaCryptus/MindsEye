package com.simiacryptus.mindseye;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Optional;
import java.util.stream.IntStream;

import org.jblas.DoubleMatrix;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray.Coords;

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
    HashMap<Coords, Coords> gradientMap = new HashMap<NDArray.Coords, NDArray.Coords>();
    output.coordStream().forEach(o -> {
      int[] i = new NDArray(kernelDims).coordStream()
          .map(k -> IntStream.range(0, k.length).map(idx -> k[idx] + kernelDims[idx] * o[idx]).toArray())
          .sorted(Comparator.comparing(inputCoords -> input.get(inputCoords))).findFirst().get();
      gradientMap.put(new Coords(o), new Coords(i));
      output.add(o, input.get(i));
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
