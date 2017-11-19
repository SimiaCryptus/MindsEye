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

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.ArrayUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * The type Softmax activation layer.
 */
public class L1NormalizationLayer extends NNLayer {
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  
  /**
   * From json softmax activation layer.
   *
   * @param json the json
   * @return the softmax activation layer
   */
  public static L1NormalizationLayer fromJson(JsonObject json) {
    return new L1NormalizationLayer(json);
  }
  
  /**
   * Instantiates a new Softmax activation layer.
   *
   * @param id the id
   */
  protected L1NormalizationLayer(JsonObject id) {
    super(id);
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(L1NormalizationLayer.class);

  /**
   * The Max input.
   */
  double maxInput = 50;
  
  /**
   * Instantiates a new Softmax activation layer.
   */
  public L1NormalizationLayer() {
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... input) {
    final NNResult in = input[0];
    final TensorList inData = in.getData();
    Tensor[] output = IntStream.range(0, inData.length()).mapToObj(dataIndex -> {
      final Tensor value = inData.get(dataIndex);
      double sum = value.sum();
      if(!Double.isFinite(sum) || 0 == sum) return value;
      return value.scale(1.0 / sum);
    }).toArray(i -> new Tensor[i]);
    return new NNResult(output) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList outDelta) {
        if (in.isAlive()) {
          Tensor[] passbackArray = IntStream.range(0, outDelta.length()).mapToObj(dataIndex -> {
            final double[] value = inData.get(dataIndex).getData();
            final double[] delta = outDelta.get(dataIndex).getData();
            double dot = ArrayUtil.dot(value, delta);
            double sum = Arrays.stream(value).sum();
            final Tensor passback = new Tensor(outDelta.get(dataIndex).getDimensions());
            double[] passbackData = passback.getData();
            if(0 != sum || Double.isFinite(sum)) for (int i = 0; i < value.length; i++) {
              passbackData[i] = (delta[i] - dot / sum) / sum;
            }
            return passback;
          }).toArray(i -> new Tensor[i]);
          assert Arrays.stream(passbackArray).flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
          in.accumulate(buffer, new TensorArray(passbackArray));
        }
      }
      
      @Override
      public boolean isAlive() {
        return in.isAlive();
      }
      
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
