package com.simiacryptus.mindseye.net.activation;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

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
    Tensor inputGradientA[] = new Tensor[itemCnt];
    Tensor[] outputA = java.util.stream.IntStream.range(0, itemCnt).mapToObj(dataIndex->{
      final Tensor input = inObj[0].data[dataIndex];
      final Tensor output = new Tensor(inObj[0].data[dataIndex].getDims());
      final Tensor inputGradient = new Tensor(input.dim());
      inputGradientA[dataIndex] = inputGradient;
      final double[] results = new double[2];
      for (int i = 0; i < input.dim(); i++) {
        eval(input.getData()[i], results);
        inputGradient.set(i, results[1]);
        output.set(i, results[0]);
      }
      return output;
    }).toArray(i->new Tensor[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (inObj[0].isAlive()) {
          Tensor[] passbackA = java.util.stream.IntStream.range(0, itemCnt).mapToObj(dataIndex->{
            final Tensor passback = new Tensor(data[dataIndex].getDims());
            final double[] gradientData = inputGradientA[dataIndex].getData();
            IntStream.range(0, passback.dim()).forEach(i -> {
              final double v = gradientData[i];
              if (Double.isFinite(v)) {
                passback.set(i, data[dataIndex].getData()[i] * v);
              }
            });
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
