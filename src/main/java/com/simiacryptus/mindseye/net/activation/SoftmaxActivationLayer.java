package com.simiacryptus.mindseye.net.activation;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;

import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

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
    final Tensor expA[] = new Tensor[itemCnt];
    Tensor[] outputA = java.util.stream.IntStream.range(0, itemCnt).mapToObj(dataIndex->{
      final Tensor input = inObj[0].data[dataIndex];
      assert 1 < input.dim();
      
      final Tensor exp;
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
    }).toArray(i->new Tensor[i]);

    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (inObj[0].isAlive()) {
          Tensor[] passbackA = java.util.stream.IntStream.range(0, itemCnt).mapToObj(dataIndex->{
            final double[] delta = data[dataIndex].getData();
            final double[] expdata = expA[dataIndex].getData();
            final Tensor passback = new Tensor(data[dataIndex].getDims());
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
