package com.simiacryptus.mindseye.net.loss;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

public class SqLossLayer extends NNLayer<SqLossLayer> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SqLossLayer.class);
  /**
   * 
   */
  private static final long serialVersionUID = 7589211270512485408L;

  public SqLossLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    NDArray rA[] = new NDArray[inObj[0].data.length];
    NDArray[] outputA = java.util.stream.IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex->{
      final NDArray a = inObj[0].data[dataIndex];
      final NDArray b = inObj[1].data[dataIndex];
      final NDArray r = new NDArray(a.getDims());
      double total = 0;
      for (int i = 0; i < a.dim(); i++) {
        final double x = a.getData()[i] - b.getData()[i];
        r.getData()[i] = x;
        total += x * x;
      }
      rA[dataIndex] = r;
      final double rms = total / a.dim();
      return new NDArray(new int[] { 1 }, new double[] { rms });
    }).toArray(i->new NDArray[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final NDArray[] data) {
        if (inObj[0].isAlive() || inObj[1].isAlive()) {
          NDArray[] passbackA = java.util.stream.IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex->{
            final NDArray passback = new NDArray(inObj[0].data[0].getDims());
            final int adim = passback.dim();
            final double data0 = data[dataIndex].get(0);
            for (int i = 0; i < adim; i++) {
              passback.set(i, data0 * rA[dataIndex].get(i) * 2 / adim);
            }
            return passback;
          }).toArray(i->new NDArray[i]);
          if (inObj[0].isAlive()) {
            inObj[0].accumulate(buffer, passbackA);
          }
          if (inObj[1].isAlive()) {
            inObj[1].accumulate(buffer, java.util.Arrays.stream(passbackA).map(x->x.scale(-1)).toArray(i->new NDArray[i]));
          }
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
