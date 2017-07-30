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

package com.simiacryptus.mindseye.layers.activation;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.*;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

public class L1NormalizationLayer extends NNLayer {
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  public static L1NormalizationLayer fromJson(JsonObject json) {
    return new L1NormalizationLayer(json);
  }
  protected L1NormalizationLayer(JsonObject id) {
    super(id);
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(L1NormalizationLayer.class);
  private static final long serialVersionUID = -8028442822064680557L;
  
  public L1NormalizationLayer() {
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    int itemCnt = inObj[0].data.length();
    double[] sum_A = new double[itemCnt];
    final Tensor inputA[] = new Tensor[itemCnt];
    final boolean isZeroInputA[] = new boolean[itemCnt];
    Tensor[] outputA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final Tensor input = inObj[0].data.get(dataIndex);
      final double sum = input.sum();
      sum_A[dataIndex] = sum;
      final boolean isZeroInput = sum == 0.;
      isZeroInputA[dataIndex] = isZeroInput;
      inputA[dataIndex] = input;
      return input.map(x -> isZeroInput ? x : x / sum);
    }).toArray(i -> new Tensor[i]);
    
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        if (inObj[0].isAlive()) {
          Tensor[] passbackA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
            final double[] delta = Arrays.copyOf(data.get(dataIndex).getData(), data.get(dataIndex).getData().length);
            final double[] indata = inputA[dataIndex].getData();
            final Tensor passback = new Tensor(data.get(dataIndex).getDimensions());
            double dot = 0;
            for (int i = 0; i < indata.length; i++) {
              dot += delta[i] * indata[i];
            }
            for (int i = 0; i < indata.length; i++) {
              final double d = delta[i];
              double sum = sum_A[dataIndex];
              passback.set(i, isZeroInputA[dataIndex] ? d : (d * sum - dot) / (sum * sum));
            }
            return passback;
          }).toArray(i -> new Tensor[i]);
          inObj[0].accumulate(buffer, new TensorArray(passbackA));
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
