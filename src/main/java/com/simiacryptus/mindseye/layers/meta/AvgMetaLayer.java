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

package com.simiacryptus.mindseye.layers.meta;

import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

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
    Tensor avgActivationArray = input.data[0].map((v, c) ->
                                                      IntStream.range(0, itemCnt)
                                                          .mapToDouble(dataIndex -> input.data[dataIndex].get(c))
                                                          .average().getAsDouble());
    return new NNResult(avgActivationArray) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (input.isAlive()) {
          Tensor delta = data[0];
          Tensor feedback[] = new Tensor[itemCnt];
          Arrays.parallelSetAll(feedback, i -> new Tensor(delta.getDims()));
          avgActivationArray.map((rho, inputCoord) -> {
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