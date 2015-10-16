package com.simiacryptus.mindseye.net.meta;

import java.util.Arrays;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.simiacryptus.mindseye.core.NDArray;
import com.simiacryptus.mindseye.core.delta.DeltaSet;
import com.simiacryptus.mindseye.core.delta.NNLayer;
import com.simiacryptus.mindseye.core.delta.NNResult;

@SuppressWarnings("serial")
public class Sparse01MetaLayer extends NNLayer<Sparse01MetaLayer> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(Sparse01MetaLayer.class);

  double sparsity = 0.05;

  public Sparse01MetaLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    NNResult input = inObj[0];
    int itemCnt = input.data.length;
    NDArray avgActivationArray = input.data[0].map((v,c)->
      java.util.stream.IntStream.range(0, itemCnt)
        .mapToDouble(dataIndex->input.data[dataIndex].get(c))
        .average().getAsDouble());
    NDArray divergenceArray = avgActivationArray.map((avgActivation,c)->{
      return sparsity * Math.log(sparsity * avgActivation) + (1-sparsity) * Math.log((1-sparsity)/(1-avgActivation));
    });
    return new NNResult(new NDArray[]{divergenceArray}) {
      @Override
      public void accumulate(final DeltaSet buffer, final NDArray[] data) {
        if (input.isAlive()) {
          NDArray delta = data[0];
          NDArray feedback[] = new NDArray[itemCnt];
          java.util.Arrays.parallelSetAll(feedback, i->new NDArray(delta.getDims()));
          avgActivationArray.map((rho,inputCoord)->{
            double d = delta.get(inputCoord);
            for (int inputItem = 0; inputItem < itemCnt; inputItem++) {
              double in = input.data[inputItem].get(inputCoord);
              double log2 = Math.log((in-1)/(rho-1));
              double log3 = Math.log(in/rho);
              double value = d * (log3-log2);
              feedback[inputItem].add(inputCoord, value);
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
