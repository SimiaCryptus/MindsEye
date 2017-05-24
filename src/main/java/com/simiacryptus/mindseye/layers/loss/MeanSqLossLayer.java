/*
 * Copyright (c) 2017 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.loss;

import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class MeanSqLossLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MeanSqLossLayer.class);
  /**
   *
   */
  private static final long serialVersionUID = 7589211270512485408L;
  
  public MeanSqLossLayer() {
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    Tensor rA[] = new Tensor[inObj[0].data.length];
    Tensor[] outputA = IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex -> {
      final Tensor a = inObj[0].data[dataIndex];
      final Tensor b = inObj[1].data[dataIndex];
      final Tensor r = new Tensor(a.getDims());
      double total = 0;
      for (int i = 0; i < a.dim(); i++) {
        final double x = a.getData()[i] - b.getData()[i];
        r.getData()[i] = x;
        total += x * x;
      }
      rA[dataIndex] = r;
      final double rms = total / a.dim();
      return new Tensor(new int[]{1}, new double[]{rms});
    }).toArray(i -> new Tensor[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (inObj[0].isAlive() || inObj[1].isAlive()) {
          Tensor[] passbackA = IntStream.range(0, inObj[0].data.length).mapToObj(dataIndex -> {
            final Tensor passback = new Tensor(inObj[0].data[0].getDims());
            final int adim = passback.dim();
            final double data0 = data[dataIndex].get(0);
            for (int i = 0; i < adim; i++) {
              passback.set(i, data0 * rA[dataIndex].get(i) * 2 / adim);
            }
            return passback;
          }).toArray(i -> new Tensor[i]);
          if (inObj[0].isAlive()) {
            inObj[0].accumulate(buffer, passbackA);
          }
          if (inObj[1].isAlive()) {
            inObj[1].accumulate(buffer, Arrays.stream(passbackA).map(x -> x.scale(-1)).toArray(i -> new Tensor[i]));
          }
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
