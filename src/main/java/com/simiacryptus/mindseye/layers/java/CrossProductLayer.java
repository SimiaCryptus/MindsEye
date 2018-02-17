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

import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * The type Cross product layer.
 */
@SuppressWarnings("serial")
public class CrossProductLayer extends LayerBase {
  
  /**
   * Instantiates a new Cross product layer.
   */
  public CrossProductLayer() {
  }
  
  /**
   * Instantiates a new Cross product layer.
   *
   * @param id the id
   */
  protected CrossProductLayer(@javax.annotation.Nonnull final JsonObject id) {
    super(id);
  }
  
  /**
   * From json cross product layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the cross product layer
   */
  public static CrossProductLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new CrossProductLayer(json);
  }
  
  /**
   * Index int.
   *
   * @param x   the x
   * @param y   the y
   * @param max the max
   * @return the int
   */
  public static int index(final int x, final int y, final int max) {
    return max * (max - 1) / 2 - (max - x) * (max - x - 1) / 2 + y - x - 1;
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNResult eval(@javax.annotation.Nonnull final NNResult... inObj) {
    assert 1 == inObj.length;
    final NNResult in = inObj[0];
    TensorList indata = in.getData();
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    indata.addRef();
    return new NNResult(TensorArray.wrap(indata.stream().parallel().map(tensor -> {
      final int inputDim = tensor.dim();
      final int outputDim = (inputDim * inputDim - inputDim) / 2;
      @javax.annotation.Nonnull final Tensor result = new Tensor(outputDim);
      @Nullable final double[] inputData = tensor.getData();
      @Nullable final double[] resultData = result.getData();
      IntStream.range(0, inputDim).forEach(x -> {
        IntStream.range(x + 1, inputDim).forEach(y -> {
          resultData[CrossProductLayer.index(x, y, inputDim)] = inputData[x] * inputData[y];
        });
      });
      tensor.freeRef();
      return result;
    }).toArray(i -> new Tensor[i])), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList delta) -> {
      if (in.isAlive()) {
        assert delta.length() == delta.length();
        @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).parallel().mapToObj(batchIndex -> {
          @javax.annotation.Nullable final Tensor deltaTensor = delta.get(batchIndex);
          final int outputDim = deltaTensor.dim();
          final int inputDim = (1 + (int) Math.sqrt(1 + 8 * outputDim)) / 2;
          @javax.annotation.Nonnull final Tensor passback = new Tensor(inputDim);
          @Nullable final double[] passbackData = passback.getData();
          @Nullable final double[] tensorData = deltaTensor.getData();
          Tensor inputTensor = indata.get(batchIndex);
          @Nullable final double[] inputData = inputTensor.getData();
          IntStream.range(0, inputDim).forEach(x -> {
            IntStream.range(x + 1, inputDim).forEach(y -> {
              passbackData[x] += tensorData[CrossProductLayer.index(x, y, inputDim)] * inputData[y];
              passbackData[y] += tensorData[CrossProductLayer.index(x, y, inputDim)] * inputData[x];
            });
          });
          deltaTensor.freeRef();
          inputTensor.freeRef();
          return passback;
        }).toArray(i -> new Tensor[i]));
        in.accumulate(buffer, tensorArray);
        tensorArray.freeRef();
      }
    }) {
      
      @Override
      protected void _free() {
        indata.freeRef();
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }
      
      @Override
      public boolean isAlive() {
        for (@javax.annotation.Nonnull final NNResult element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
      }
      
    };
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
}
