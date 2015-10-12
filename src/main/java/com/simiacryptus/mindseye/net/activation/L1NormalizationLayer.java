package com.simiacryptus.mindseye.net.activation;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

public class L1NormalizationLayer extends NNLayer<L1NormalizationLayer> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(L1NormalizationLayer.class);
  private static final long serialVersionUID = -8028442822064680557L;

  public L1NormalizationLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    int itemCnt = inObj[0].data.length;
    double[] sum_A = new double[itemCnt];
    final NDArray inputA[] = new NDArray[itemCnt];
    final boolean isZeroInputA[] = new boolean[itemCnt];
    NDArray[] outputA = java.util.stream.IntStream.range(0, itemCnt).mapToObj(dataIndex->{
      final NDArray input = inObj[0].data[dataIndex];
      final double sum = input.sum();
      sum_A[dataIndex] = sum;
      final boolean isZeroInput = sum == 0.;
      isZeroInputA[dataIndex] = isZeroInput;
      return input.map(x -> isZeroInput ? x : x / sum);
    }).toArray(i->new NDArray[i]);

    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final NDArray[] data) {
        if (inObj[0].isAlive()) {
          NDArray[] passbackA = java.util.stream.IntStream.range(0, itemCnt).mapToObj(dataIndex->{
            final double[] delta = Arrays.copyOf(data[dataIndex].getData(), data[dataIndex].getData().length);
            final double[] indata = inputA[dataIndex].getData();
            final NDArray passback = new NDArray(data[dataIndex].getDims());
            double dot = 0;
            for (int i = 0; i < indata.length; i++) {
              dot += delta[i] * indata[i];
            }
            for (int i = 0; i < indata.length; i++) {
              final double d = delta[i];
              double sum = sum_A[dataIndex];
              passback.set(i, isZeroInputA[dataIndex] ? d : (d * sum - dot) / (sum * sum));
            }
            return passback;
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
