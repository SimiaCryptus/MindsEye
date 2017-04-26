package com.simiacryptus.mindseye.net.meta;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.util.ml.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

@SuppressWarnings("serial")
public class AvgMetaLayer extends NNLayer<AvgMetaLayer> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AvgMetaLayer.class);

  double sparsity = 0.05;

  public AvgMetaLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    NNResult input = inObj[0];
    int itemCnt = input.data.length;
    NDArray avgActivationArray = input.data[0].map((v,c)->
      java.util.stream.IntStream.range(0, itemCnt)
        .mapToDouble(dataIndex->input.data[dataIndex].get(c))
        .average().getAsDouble());
    return new NNResult(new NDArray[]{avgActivationArray}) {
      @Override
      public void accumulate(final DeltaSet buffer, final NDArray[] data) {
        if (input.isAlive()) {
          NDArray delta = data[0];
          NDArray feedback[] = new NDArray[itemCnt];
          java.util.Arrays.parallelSetAll(feedback, i->new NDArray(delta.getDims()));
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
