package com.simiacryptus.mindseye.layers;

import java.util.stream.DoubleStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.NDArray;
import com.simiacryptus.mindseye.learning.NNResult;

public class NormalizerLayer extends NNLayer {
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(NormalizerLayer.class);

  private int count = 0;
  private final double[] sum;
  private double sumMag = 0;

  public NormalizerLayer(final int[] outputDims) {
    this.sum = new double[NDArray.dim(outputDims)];
  }

  @Override
  public NNResult eval(final NNResult inObj) {
    this.count++;
    final NDArray translated = 0 == this.count ? inObj.data : inObj.data.map((v, i) -> {
      this.sum[i.index] += v;
      final double avgi = this.sum[i.index] / this.count;
      return v - avgi;
    });
    this.sumMag += Math.sqrt(DoubleStream.of(translated.data).map(x -> x * x).sum());
    final double avg = this.sumMag / this.count;

    final NDArray scaled = translated.map((v, i) -> {
      return v / avg;
    });
    return new NNResult(scaled) {
      @Override
      public void feedback(final NDArray data) {
        if (inObj.isAlive())
        {
          inObj.feedback(data.map((v, i) -> v * avg));
        }
      }

      @Override
      public boolean isAlive() {
        return inObj.isAlive();
      }
    };
  }

}
