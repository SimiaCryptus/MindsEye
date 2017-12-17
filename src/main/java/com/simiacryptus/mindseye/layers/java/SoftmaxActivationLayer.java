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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * The type Softmax activation layer.
 */
@SuppressWarnings("serial")
public class SoftmaxActivationLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(SoftmaxActivationLayer.class);
  /**
   * The Max input.
   */
  double maxInput = 50;
  
  /**
   * Instantiates a new Softmax activation layer.
   */
  public SoftmaxActivationLayer() {
  }
  
  /**
   * Instantiates a new Softmax activation layer.
   *
   * @param id the id
   */
  protected SoftmaxActivationLayer(final JsonObject id) {
    super(id);
  }
  
  /**
   * From json softmax activation layer.
   *
   * @param json the json
   * @return the softmax activation layer
   */
  public static SoftmaxActivationLayer fromJson(final JsonObject json) {
    return new SoftmaxActivationLayer(json);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    final int itemCnt = inObj[0].getData().length();
    final double[] sumA = new double[itemCnt];
    final Tensor expA[] = new Tensor[itemCnt];
    final Tensor[] outputA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final Tensor input = inObj[0].getData().get(dataIndex);
      assert 1 < input.dim() : "input.dim() = " + input.dim();
      
      final Tensor exp;
      final DoubleSummaryStatistics summaryStatistics = DoubleStream.of(input.getData()).filter(x -> Double.isFinite(x)).summaryStatistics();
      final double max = summaryStatistics.getMax();
      //final double min = summaryStatistics.getMin();
      exp = inObj[0].getData().get(dataIndex).map(x -> Math.exp(x - max)).map(x -> {
        return Double.isFinite(x) ? x : 0;
      });
      assert Arrays.stream(exp.getData()).allMatch(Double::isFinite);
      assert Arrays.stream(exp.getData()).allMatch(v -> v >= 0);
      //assert exp.sum() > 0;
      final double sum = 0 < exp.sum() ? exp.sum() : 1;
      assert Double.isFinite(sum);
      expA[dataIndex] = exp;
      sumA[dataIndex] = sum;
      return exp.map(x -> x / sum);
    }).toArray(i -> new Tensor[i]);
    assert Arrays.stream(outputA).flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        if (inObj[0].isAlive()) {
          final Tensor[] passbackA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
            final double[] delta = data.get(dataIndex).getData();
            final double[] expdata = expA[dataIndex].getData();
            final Tensor passback = new Tensor(data.get(dataIndex).getDimensions());
            final int dim = expdata.length;
            double dot = 0;
            for (int i = 0; i < expdata.length; i++) {
              dot += delta[i] * expdata[i];
            }
            final double sum = sumA[dataIndex];
            for (int i = 0; i < dim; i++) {
              double value = 0;
              value = (sum * delta[i] - dot) * expdata[i] / (sum * sum);
              passback.set(i, value);
            }
            return passback;
          }).toArray(i -> new Tensor[i]);
          assert Arrays.stream(passbackA).flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
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
  public JsonObject getJson() {
    return super.getJsonStub();
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
