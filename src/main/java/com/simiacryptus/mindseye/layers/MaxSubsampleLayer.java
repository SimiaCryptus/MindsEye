package com.simiacryptus.mindseye.layers;

import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.learning.NNResult;
import com.simiacryptus.mindseye.math.Coordinate;
import com.simiacryptus.mindseye.math.LogNDArray;
import com.simiacryptus.mindseye.math.NDArray;

public class MaxSubsampleLayer extends NNLayer {
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
  public NNResult eval(final NNResult inObj) {
    final NDArray input = inObj.data;
    final int[] inputDims = input.getDims();
    final int[] newDims = IntStream.range(0, inputDims.length).map(
        i -> inputDims[i] / this.kernelDims[i]
        ).toArray();
    final NDArray output = new NDArray(newDims);
    final HashMap<Coordinate, Coordinate> gradientMap = new HashMap<Coordinate, Coordinate>();
    output.coordStream(false).forEach(o -> {
      final int[] i = new NDArray(this.kernelDims).coordStream(false)
          .map(kernelCoord -> Coordinate.add(o.coords, kernelCoord.coords))
          .sorted(Comparator.comparing(inputCoords -> input.get(inputCoords))).findFirst().get();
      final Coordinate inputCoord = new Coordinate(input.index(i), i);
      gradientMap.put(o, inputCoord);
      output.add(o, input.get(inputCoord));
    });
    return new NNResult(output) {
      @Override
      public void feedback(final LogNDArray data) {
        if (inObj.isAlive()) {
          final LogNDArray backSignal = new LogNDArray(inputDims);
          gradientMap.entrySet().forEach(e -> backSignal.add(e.getValue().coords, data.get(e.getKey().coords)));
          inObj.feedback(backSignal);
        }
      }
      
      @Override
      public boolean isAlive() {
        return inObj.isAlive();
      }
    };
  }
}
