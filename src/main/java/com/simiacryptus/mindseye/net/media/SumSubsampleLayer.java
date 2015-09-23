package com.simiacryptus.mindseye.net.media;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.deltas.DeltaBuffer;
import com.simiacryptus.mindseye.deltas.NNResult;
import com.simiacryptus.mindseye.math.Coordinate;
import com.simiacryptus.mindseye.math.NDArray;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.dag.EvaluationContext;

public class SumSubsampleLayer extends NNLayer<SumSubsampleLayer> {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SumSubsampleLayer.class);

  private int[] kernelDims;

  protected SumSubsampleLayer() {
    super();
  }

  public SumSubsampleLayer(final int... kernelDims) {

    this.kernelDims = Arrays.copyOf(kernelDims, kernelDims.length);
  }

  @Override
  public NNResult eval(final EvaluationContext evaluationContext, final NNResult... inObj) {
    int kernelSize = new NDArray(this.kernelDims).dim();
    final NDArray input = inObj[0].data;
    final int[] inputDims = input.getDims();
    final int[] newDims = IntStream.range(0, inputDims.length).map(i -> {
      assert 0 == inputDims[i] % this.kernelDims[i];
      return inputDims[i] / this.kernelDims[i];
    }).toArray();
    final NDArray output = new NDArray(newDims);
    output.coordStream(false).forEach(o -> {
      Stream<int[]> kernelCoords = getKernelCoords(o);
      final double x = kernelCoords
          .mapToDouble(i->input.get(i))
          .sum();
      if(Double.isFinite(x) && kernelSize>0) output.add(o, x/kernelSize);
    });
    return new NNResult(evaluationContext, output) {
      @Override
      public void feedback(final NDArray data, final DeltaBuffer buffer) {
        if (inObj[0].isAlive()) {
          final NDArray backSignal = new NDArray(inputDims);
          output.coordStream(false).forEach(o -> {
            double outV = output.get(o);
            getKernelCoords(o).forEach(i->{
              backSignal.add(i, outV/kernelSize);
            });
          });
          inObj[0].feedback(backSignal, buffer);
        }
      }

      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
    };
  }

  public Stream<int[]> getKernelCoords(Coordinate o) {
    Stream<int[]> kernelCoords = new NDArray(this.kernelDims).coordStream(false).map(kernelCoord -> {
      final int[] r = new int[o.coords.length];
      for (int i1 = 0; i1 < o.coords.length; i1++) {
        r[i1] = o.coords[i1] * this.kernelDims[i1] + kernelCoord.coords[i1];
      }
      return r;
    });
    return kernelCoords;
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
