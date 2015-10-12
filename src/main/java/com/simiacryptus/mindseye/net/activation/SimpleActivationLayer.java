package com.simiacryptus.mindseye.net.activation;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

public abstract class SimpleActivationLayer<T extends SimpleActivationLayer<T>> extends NNLayer<T> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SigmoidActivationLayer.class);
  /**
   * 
   */
  private static final long serialVersionUID = -5439874559292833041L;

  public SimpleActivationLayer() {
    super();
  }

  protected abstract void eval(final double x, double[] results);

  @Override
  public NNResult eval(final NNResult... inObj) {
    int itemCnt = inObj[0].data.length;
    NDArray inputGradientA[] = new NDArray[itemCnt];
    NDArray[] outputA = java.util.stream.IntStream.range(0, itemCnt).mapToObj(dataIndex->{
      final NDArray input = inObj[0].data[dataIndex];
      final NDArray output = new NDArray(inObj[0].data[dataIndex].getDims());
      final NDArray inputGradient = new NDArray(input.dim());
      inputGradientA[dataIndex] = inputGradient;
      final double[] results = new double[2];
      for (int i = 0; i < input.dim(); i++) {
        eval(input.getData()[i], results);
        inputGradient.set(i, results[1]);
        output.set(i, results[0]);
      }
      return output;
    }).toArray(i->new NDArray[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final NDArray[] data) {
        if (inObj[0].isAlive()) {
          NDArray[] passbackA = java.util.stream.IntStream.range(0, itemCnt).mapToObj(dataIndex->{
            final NDArray passback = new NDArray(data[dataIndex].getDims());
            final double[] gradientData = inputGradientA[dataIndex].getData();
            IntStream.range(0, passback.dim()).forEach(i -> {
              final double v = gradientData[i];
              if (Double.isFinite(v)) {
                passback.set(i, data[dataIndex].getData()[i] * v);
              }
            });
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
