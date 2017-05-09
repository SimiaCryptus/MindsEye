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
public class Sparse01MetaLayer extends NNLayer {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(Sparse01MetaLayer.class);

  double sparsity = 0.05;

  public Sparse01MetaLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    NNResult input = inObj[0];
    int itemCnt = input.data.length;
    Tensor avgActivationArray = input.data[0].map((v, c)->
      java.util.stream.IntStream.range(0, itemCnt)
        .mapToDouble(dataIndex->input.data[dataIndex].get(c))
        .average().getAsDouble());
    Tensor divergenceArray = avgActivationArray.map((avgActivation, c)->{
      assert(Double.isFinite(avgActivation));
      if(avgActivation > 0 && avgActivation < 1)
        return sparsity * Math.log(sparsity / avgActivation) + (1-sparsity) * Math.log((1-sparsity)/(1-avgActivation));
      else
        return 0;
    });
    return new NNResult(divergenceArray) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (input.isAlive()) {
          Tensor delta = data[0];
          Tensor feedback[] = new Tensor[itemCnt];
          java.util.Arrays.parallelSetAll(feedback, i->new Tensor(delta.getDims()));
          avgActivationArray.map((rho,inputCoord)->{
            double d = delta.get(inputCoord);
            double log2 = (1-sparsity)/(1-rho);
            double log3 = sparsity/rho;
            double value = d * (log2-log3) / itemCnt;
            if(Double.isFinite(value))
              for (int inputItem = 0; inputItem < itemCnt; inputItem++) {
                //double in = input.data[inputItem].get(inputCoord);
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
