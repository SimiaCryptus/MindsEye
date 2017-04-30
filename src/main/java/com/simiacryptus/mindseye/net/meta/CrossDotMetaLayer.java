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
public class CrossDotMetaLayer extends NNLayer<CrossDotMetaLayer> {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(CrossDotMetaLayer.class);

  public CrossDotMetaLayer() {
  }

  @Override
  public NNResult eval(final NNResult... inObj) {
    NNResult input = inObj[0];
    int itemCnt = input.data.length;
    int dim = input.data[0].dim();
    Tensor results = new Tensor(dim,dim);
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
    return new NNResult(new Tensor[]{results}) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (input.isAlive()) {
          Tensor delta = data[0];
          Tensor feedback[] = new Tensor[itemCnt];
          java.util.Arrays.parallelSetAll(feedback, i->new Tensor(dim));

          for(int i=0;i<dim;i++){
            for(int j=0;j<dim;j++){
              if(i==j) continue;
              double v = delta.get(i,j);
              for(int k=0;k<itemCnt;k++){
                double[] kk = input.data[k].getData();
                feedback[k].add(i, v * kk[j]);
                feedback[k].add(j, v * kk[i]);
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
