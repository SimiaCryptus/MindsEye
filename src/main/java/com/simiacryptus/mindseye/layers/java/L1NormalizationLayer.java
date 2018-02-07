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
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.ArrayUtil;
import org.jetbrains.annotations.NotNull;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Normalizes the input so that the L1 magnitude (ie sum of abs) is 1.
 */
@SuppressWarnings("serial")
public class L1NormalizationLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(L1NormalizationLayer.class);
  /**
   * The Max input.
   */
  double maxInput = 50;
  
  /**
   * Instantiates a new L 1 normalization layer.
   */
  public L1NormalizationLayer() {
  }
  
  /**
   * Instantiates a new L 1 normalization layer.
   *
   * @param id the id
   */
  protected L1NormalizationLayer(final @NotNull JsonObject id) {
    super(id);
  }
  
  /**
   * From json l 1 normalization layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the l 1 normalization layer
   */
  public static L1NormalizationLayer fromJson(final @NotNull JsonObject json, Map<String, byte[]> rs) {
    return new L1NormalizationLayer(json);
  }
  
  @Override
  public @NotNull NNResult eval(final @NotNull NNResult... input) {
    Arrays.stream(input).forEach(nnResult -> nnResult.addRef());
    final NNResult in = input[0];
    final TensorList inData = in.getData();
    inData.addRef();
    return new NNResult(TensorArray.wrap(IntStream.range(0, inData.length()).mapToObj(dataIndex -> {
      final Tensor value = inData.get(dataIndex);
      final double sum = value.sum();
      if (!Double.isFinite(sum) || 0 == sum) return value;
      return value.scale(1.0 / sum);
    }).toArray(i -> new Tensor[i])), (final @NotNull DeltaSet<NNLayer> buffer, final @NotNull TensorList outDelta) -> {
      if (in.isAlive()) {
        final Tensor[] passbackArray = IntStream.range(0, outDelta.length()).mapToObj(dataIndex -> {
          final @Nullable double[] value = inData.get(dataIndex).getData();
          final @Nullable double[] delta = outDelta.get(dataIndex).getData();
          final double dot = ArrayUtil.dot(value, delta);
          final double sum = Arrays.stream(value).sum();
          final @NotNull Tensor passback = new Tensor(outDelta.get(dataIndex).getDimensions());
          final @Nullable double[] passbackData = passback.getData();
          if (0 != sum || Double.isFinite(sum)) {
            for (int i = 0; i < value.length; i++) {
              passbackData[i] = (delta[i] - dot / sum) / sum;
            }
          }
          return passback;
        }).toArray(i -> new Tensor[i]);
        assert Arrays.stream(passbackArray).flatMapToDouble(x -> Arrays.stream(x.getData())).allMatch(v -> Double.isFinite(v));
        @NotNull TensorArray tensorArray = TensorArray.wrap(passbackArray);
        in.accumulate(buffer, tensorArray);
        tensorArray.freeRef();
      }
    }) {
  
      @Override
      protected void _free() {
        inData.freeRef();
        Arrays.stream(input).forEach(nnResult -> nnResult.freeRef());
      }
  
  
      @Override
      public boolean isAlive() {
        return in.isAlive();
      }
      
    };
  }
  
  @Override
  public @NotNull JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }
  
  @Override
  public @NotNull List<double[]> state() {
    return Arrays.asList();
  }
}
