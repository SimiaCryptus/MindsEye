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
public class CrossDotMetaLayer extends NNLayer<CrossDotMetaLayer> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(CrossDotMetaLayer.class);

  double sparsity = 0.05;

  public CrossDotMetaLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    NNResult input = inObj[0];
    int itemCnt = input.data.length;
    
    int dim = input.data[0].dim();
    NDArray results = new NDArray(dim,dim);
    for(int i=0;i<dim;i++){
      for(int j=0;j<dim;j++){
        if(i==j) continue;
        double v = 0;
        for(int k=0;k<itemCnt;k++){
          double[] kk = input.data[k].getData();
          v += kk[i] * kk[j];
        }
        results.set(new int[]{i,j}, v);
      }
    }
    return new NNResult(new NDArray[]{results}) {
      @Override
      public void accumulate(final DeltaSet buffer, final NDArray[] data) {
        if (input.isAlive()) {
          NDArray delta = data[0];
          NDArray feedback[] = new NDArray[itemCnt];
          java.util.Arrays.parallelSetAll(feedback, i->new NDArray(dim));

          for(int i=0;i<dim;i++){
            for(int j=0;j<dim;j++){
              if(i==j) continue;
              for(int k=0;k<itemCnt;k++){
                double[] kk = input.data[k].getData();
                feedback[k].set(i, delta.get(i,j) * kk[j]);
              }
            }
          }

          
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
