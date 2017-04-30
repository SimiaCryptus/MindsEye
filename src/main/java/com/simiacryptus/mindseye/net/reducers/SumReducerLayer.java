package com.simiacryptus.mindseye.net.reducers;

import java.util.Arrays;
import java.util.List;

import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

public class SumReducerLayer extends NNLayer<SumReducerLayer> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SumReducerLayer.class);
  /**
   * 
   */
  private static final long serialVersionUID = -5171545060770814729L;

  public SumReducerLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    double outputA = java.util.stream.IntStream.range(0, inObj[0].data.length).mapToDouble(dataIndex->{
      double sum = 0;
      for (final NNResult element : inObj) {
        final double[] input = element.data[dataIndex].getData();
        for (final double element2 : input) {
          sum += element2;
        }
      }
      return sum;
    }).sum();
    return new NNResult(new Tensor(new int[]{1},new double[]{outputA})) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        for (final NNResult in_l : inObj) {
          if (in_l.isAlive()) {
            Tensor[] passbackA = java.util.stream.IntStream.range(0, in_l.data.length).mapToObj(dataIndex->{
              final double delta = data[0].get(0);
              final Tensor passback = new Tensor(in_l.data[dataIndex].getDims());
              for (int i = 0; i < in_l.data[dataIndex].dim(); i++) {
                passback.set(i, delta);
              }
              return passback;
            }).toArray(i->new Tensor[i]);
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
