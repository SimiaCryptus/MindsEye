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
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.UUID;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

public class SoftmaxActivationLayer extends NNLayer {
  
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  public static SoftmaxActivationLayer fromJson(JsonObject json) {
    return new SoftmaxActivationLayer(UUID.fromString(json.get("id").getAsString()));
  }
  protected SoftmaxActivationLayer(UUID id) {
    super(id);
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SoftmaxActivationLayer.class);
  
  /**
   *
   */
  private static final long serialVersionUID = 2373420906380031927L;
  
  double maxInput = 50;
  
  public SoftmaxActivationLayer() {
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
    int itemCnt = inObj[0].data.length;
    double[] sumA = new double[itemCnt];
    final Tensor expA[] = new Tensor[itemCnt];
    Tensor[] outputA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final Tensor input = inObj[0].data[dataIndex];
      assert 1 < input.dim();
      
      final Tensor exp;
      final DoubleSummaryStatistics summaryStatistics = DoubleStream.of(input.getData()).filter(x -> Double.isFinite(x)).summaryStatistics();
      final double max = summaryStatistics.getMax();
      //final double min = summaryStatistics.getMin();
      exp = inObj[0].data[dataIndex].map(x -> {
        return Double.isFinite(x) ? x : 0;
      }).map(x -> Math.exp(x - max));
      
      final double sum = exp.sum();
      assert (Double.isFinite(sum));
      assert (0.0 != sum);
      expA[dataIndex] = exp;
      sumA[dataIndex] = sum;
      return exp.map(x -> x / sum);
    }).toArray(i -> new Tensor[i]);
    
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final Tensor[] data) {
        if (inObj[0].isAlive()) {
          Tensor[] passbackA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
            final double[] delta = data[dataIndex].getData();
            final double[] expdata = expA[dataIndex].getData();
            final Tensor passback = new Tensor(data[dataIndex].getDims());
            final int dim = expdata.length;
            double dot = 0;
            for (int i = 0; i < expdata.length; i++) {
              dot += delta[i] * expdata[i];
            }
            double sum = sumA[dataIndex];
            for (int i = 0; i < dim; i++) {
              double value = 0;
              value = ((sum * delta[i] - dot) * expdata[i]) / (sum * sum);
              passback.set(i, value);
            }
            return passback;
          }).toArray(i -> new Tensor[i]);
          inObj[0].accumulate(buffer, passbackA);
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
