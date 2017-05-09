package com.simiacryptus.mindseye.net.meta;

import java.util.Arrays;
import java.util.List;

import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.net.DeltaSet;
import com.simiacryptus.mindseye.net.NNLayer;
import com.simiacryptus.mindseye.net.NNResult;

@SuppressWarnings("serial")
public class AvgMetaLayer extends NNLayer {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AvgMetaLayer.class);

  double sparsity = 0.05;

  public AvgMetaLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    NNResult input = inObj[0];
    int itemCnt = input.data.length;
    Tensor avgActivationArray = input.data[0].map((v, c)->
      java.util.stream.IntStream.range(0, itemCnt)
        .mapToDouble(dataIndex->input.data[dataIndex].get(c))
        .average().getAsDouble());
    return new NNResult(avgActivationArray) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (input.isAlive()) {
          Tensor delta = data[0];
          Tensor feedback[] = new Tensor[itemCnt];
          java.util.Arrays.parallelSetAll(feedback, i->new Tensor(delta.getDims()));
          avgActivationArray.map((rho,inputCoord)->{
            for (int inputItem = 0; inputItem < itemCnt; inputItem++) {
              feedback[inputItem].add(inputCoord, delta.get(inputCoord) / itemCnt);
            }
            return 0;
          });
          input.accumulate(buffer, feedback);
        }
      }

      @Override
      public boolean isAlive() {
        return input.isAlive();
      }

    };
  }

  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
