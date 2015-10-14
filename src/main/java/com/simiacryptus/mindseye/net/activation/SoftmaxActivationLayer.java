package com.simiacryptus.mindseye.net.activation;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

public class SoftmaxActivationLayer extends NNLayer<SoftmaxActivationLayer> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SoftmaxActivationLayer.class);

  /**
   * 
   */
  private static final long serialVersionUID = 2373420906380031927L;

  double maxInput = 50;

  public SoftmaxActivationLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    int itemCnt = inObj[0].data.length;
    double[] sumA = new double[itemCnt];
    final NDArray expA[] = new NDArray[itemCnt];
    NDArray[] outputA = java.util.stream.IntStream.range(0, itemCnt).mapToObj(dataIndex->{
      final NDArray input = inObj[0].data[dataIndex];
      assert 1 < input.dim();
      
      final NDArray exp;
      {
        final DoubleSummaryStatistics summaryStatistics = java.util.stream.DoubleStream.of(input.getData()).filter(x -> Double.isFinite(x)).summaryStatistics();
        final double max = summaryStatistics.getMax();
        //final double min = summaryStatistics.getMin();
        exp = inObj[0].data[dataIndex].map(x -> {
          return Double.isFinite(x) ? x : Double.NaN;
        }).map(x -> Math.exp(x - max));
      }
      
      final double sum = exp.sum();
      assert 0. < sum;
      expA[dataIndex] = exp;
      sumA[dataIndex] = sum;
      return exp.map(x -> x / sum);
    }).toArray(i->new NDArray[i]);

    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final NDArray[] data) {
        if (inObj[0].isAlive()) {
          NDArray[] passbackA = java.util.stream.IntStream.range(0, itemCnt).mapToObj(dataIndex->{
            final double[] delta = data[dataIndex].getData();
            final double[] expdata = expA[dataIndex].getData();
            final NDArray passback = new NDArray(data[dataIndex].getDims());
            final int dim = expdata.length;
            double dot = 0;
            for (int i = 0; i < expdata.length; i++) {
              dot += delta[i] * expdata[i];
            }
            double sum = sumA[dataIndex];
            for (int i = 0; i < dim; i++) {
              double value = 0;
              value = ((sum * delta[i] - dot) * expdata[i]) / (sum * sum);
              passback.set(i, value);
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
