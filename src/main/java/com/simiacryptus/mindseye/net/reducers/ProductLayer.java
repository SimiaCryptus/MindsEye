package com.simiacryptus.mindseye.net.reducers;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.util.ml.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

public class ProductLayer extends NNLayer<ProductLayer> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ProductLayer.class);
  /**
   * 
   */
  private static final long serialVersionUID = -5171545060770814729L;

  public ProductLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    double[] sum_A = new double[inObj[0].data.length];
    NDArray[] outputA = java.util.stream.IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex->{
      double sum = 1;
      for (final NNResult element : inObj) {
        final double[] input = element.data[dataIndex].getData();
        for (final double element2 : input) {
          sum *= element2;
        }
      }
      sum_A[dataIndex] = sum;
      return new NDArray(new int[] { 1 }, new double[] { sum });
    }).toArray(i->new NDArray[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final NDArray[] data) {
        for (final NNResult in_l : inObj) {
          if (in_l.isAlive()) {
            NDArray[] passbackA = java.util.stream.IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex->{
              final double delta = data[dataIndex].get(0);
              final NDArray passback = new NDArray(in_l.data[dataIndex].getDims());
              for (int i = 0; i < in_l.data[dataIndex].dim(); i++) {
                passback.set(i, delta * sum_A[dataIndex] / in_l.data[dataIndex].getData()[i]);
              }
              return passback;
            }).toArray(i->new NDArray[i]);
            in_l.accumulate(buffer, passbackA);
          }
        }
      }

      @Override
      public boolean isAlive() {
        for (final NNResult element : inObj)
          if (element.isAlive())
            return true;
        return false;
      }

    };
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
