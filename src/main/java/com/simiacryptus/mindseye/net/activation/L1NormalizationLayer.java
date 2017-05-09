package com.simiacryptus.mindseye.net.activation;

import java.util.Arrays;
import java.util.List;

import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

public class L1NormalizationLayer extends NNLayer {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(L1NormalizationLayer.class);
  private static final long serialVersionUID = -8028442822064680557L;

  public L1NormalizationLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    int itemCnt = inObj[0].data.length;
    double[] sum_A = new double[itemCnt];
    final Tensor inputA[] = new Tensor[itemCnt];
    final boolean isZeroInputA[] = new boolean[itemCnt];
    Tensor[] outputA = java.util.stream.IntStream.range(0, itemCnt).mapToObj(dataIndex->{
      final Tensor input = inObj[0].data[dataIndex];
      final double sum = input.sum();
      sum_A[dataIndex] = sum;
      final boolean isZeroInput = sum == 0.;
      isZeroInputA[dataIndex] = isZeroInput;
      inputA[dataIndex] = input;
      return input.map(x -> isZeroInput ? x : x / sum);
    }).toArray(i->new Tensor[i]);

    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (inObj[0].isAlive()) {
          Tensor[] passbackA = java.util.stream.IntStream.range(0, itemCnt).mapToObj(dataIndex->{
            final double[] delta = Arrays.copyOf(data[dataIndex].getData(), data[dataIndex].getData().length);
            final double[] indata = inputA[dataIndex].getData();
            final Tensor passback = new Tensor(data[dataIndex].getDims());
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
          }).toArray(i->new Tensor[i]);
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
