/*
 * Copyright (c) 2018 by Andrew Charneski.
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
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.ReferenceCounting;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.DoubleSummaryStatistics;
import java.util.List;
import java.util.Map;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * The classic "softmax" layer. All outputs will sum to 1 and be proportional to the log of the input.
 */
@SuppressWarnings("serial")
public class SoftmaxActivationLayer extends LayerBase {
  
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
  protected SoftmaxActivationLayer(@Nonnull final JsonObject id) {
    super(id);
  }
  
  /**
   * From json softmax activation layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the softmax activation layer
   */
  public static SoftmaxActivationLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new SoftmaxActivationLayer(json);
  }
  
  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    final int itemCnt = inObj[0].getData().length();
    @Nonnull final double[] sumA = new double[itemCnt];
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    @Nonnull final Tensor expA[] = new Tensor[itemCnt];
    final Tensor[] outputA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      @Nullable final Tensor input = inObj[0].getData().get(dataIndex);
      assert 1 < input.length() : "input.length() = " + input.length();
  
      @Nullable final Tensor exp;
      final DoubleSummaryStatistics summaryStatistics = DoubleStream.of(input.getData()).filter(x -> Double.isFinite(x)).summaryStatistics();
      final double max = summaryStatistics.getMax();
      //final double min = summaryStatistics.getMin();
      exp = input.map(x -> {
        double xx = Math.exp(x - max);
        return Double.isFinite(xx) ? xx : 0;
      });
      input.freeRef();
      assert Arrays.stream(exp.getData()).allMatch(Double::isFinite);
      assert Arrays.stream(exp.getData()).allMatch(v -> v >= 0);
      //assert exp.sum() > 0;
      final double sum = 0 < exp.sum() ? exp.sum() : 1;
      assert Double.isFinite(sum);
      expA[dataIndex] = exp;
      sumA[dataIndex] = sum;
      @Nullable Tensor result = exp.map(x -> x / sum);
      return result;
    }).toArray(i -> new Tensor[i]);
    assert Arrays.stream(outputA).flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
    return new Result(TensorArray.wrap(outputA), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList data) -> {
      if (inObj[0].isAlive()) {
        final Tensor[] passbackA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
          Tensor deltaTensor = data.get(dataIndex);
          @Nullable final double[] delta = deltaTensor.getData();
          @Nullable final double[] expdata = expA[dataIndex].getData();
          @Nonnull final Tensor passback = new Tensor(data.getDimensions());
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
          deltaTensor.freeRef();
          return passback;
        }).toArray(i -> new Tensor[i]);
        assert Arrays.stream(passbackA).flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
        @Nonnull TensorArray tensorArray = TensorArray.wrap(passbackA);
        inObj[0].accumulate(buffer, tensorArray);
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(expA).forEach(ReferenceCounting::freeRef);
        Arrays.stream(inObj).forEach(ReferenceCounting::freeRef);
      }
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
      
    };
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
