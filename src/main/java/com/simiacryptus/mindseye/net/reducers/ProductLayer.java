package com.simiacryptus.mindseye.net.reducers;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.NNLayer;
import com.simiacryptus.mindseye.core.NNResult;
import com.simiacryptus.mindseye.core.delta.DeltaSet;

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
    double sum = 1;
    for (final NNResult element : inObj) {
      final double[] input = element.data.getData();
      for (final double element2 : input) {
        sum *= element2;
      }
    }
    final double sum_ = sum;
    final NDArray output = new NDArray(new int[] { 1 }, new double[] { sum });
    return new NNResult(output) {
      @Override
      public void feedback(final NDArray data, final DeltaSet buffer) {
        final double delta = data.get(0);
        for (final NNResult in_l : inObj) {
          if (in_l.isAlive()) {
            final NDArray passback = new NDArray(in_l.data.getDims());
            for (int i = 0; i < in_l.data.dim(); i++) {
              passback.set(i, delta * sum_ / in_l.data.getData()[i]);
            }
            in_l.feedback(passback, buffer);
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
