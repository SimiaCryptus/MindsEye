package com.simiacryptus.mindseye.net.meta;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
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
    int itemCnt = inObj[0].data.length;
    NDArray avgActivationArray = inObj[0].data[0].map((v,c)->{
      double avgActivation = java.util.stream.IntStream.range(0, itemCnt).mapToDouble(dataIndex->inObj[0].data[dataIndex].get(c)).average().getAsDouble();
      return avgActivation;
    });
    NDArray divergenceArray = avgActivationArray.map((avgActivation,c)->{
      double divergence = sparsity * Math.log(sparsity * avgActivation) + (1-sparsity) * Math.log((1-sparsity)/(1-avgActivation)); 
      return divergence;
    });
    return new NNResult(new NDArray[]{divergenceArray}) {
      @Override
      public void accumulate(final DeltaSet buffer, final NDArray[] data) {
        if (inObj[0].isAlive()) {

          
          NDArray results[] = new NDArray[itemCnt];
          java.util.Arrays.parallelSetAll(results, i->new NDArray(data[0].getDims()));
          inObj[0].data[0].map((v,c)->{
            double rho = avgActivationArray.get(c);
            double d = data[0].get(c);
            for (int i = 0; i < itemCnt; i++) {
              double in = inObj[0].data[i].get(c);
              results[i].set(c, d * (Math.log(in/rho)-Math.log((in-1)/(rho-1))));
            }
            return 0;
          });
          inObj[0].accumulate(buffer, results);
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
