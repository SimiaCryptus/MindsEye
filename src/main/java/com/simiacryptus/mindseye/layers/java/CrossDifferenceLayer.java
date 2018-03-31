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
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * The type Cross difference layer.
 */
@SuppressWarnings("serial")
public class CrossDifferenceLayer extends LayerBase {
  
  /**
   * Instantiates a new Cross difference layer.
   */
  public CrossDifferenceLayer() {
  }
  
  /**
   * Instantiates a new Cross difference layer.
   *
   * @param id the id
   */
  protected CrossDifferenceLayer(@Nonnull final JsonObject id) {
    super(id);
  }
  
  /**
   * From json cross difference layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the cross difference layer
   */
  public static CrossDifferenceLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new CrossDifferenceLayer(json);
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
  
  @Nonnull
  @Override
  public Result eval(@Nonnull final Result... inObj) {
    assert 1 == inObj.length;
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    return new Result(TensorArray.wrap(inObj[0].getData().stream().parallel().map(tensor -> {
      final int inputDim = tensor.length();
      final int outputDim = (inputDim * inputDim - inputDim) / 2;
      @Nonnull final Tensor result = new Tensor(outputDim);
      @Nullable final double[] inputData = tensor.getData();
      @Nullable final double[] resultData = result.getData();
      IntStream.range(0, inputDim).forEach(x -> {
        IntStream.range(x + 1, inputDim).forEach(y -> {
          resultData[CrossDifferenceLayer.index(x, y, inputDim)] = inputData[x] - inputData[y];
        });
      });
      tensor.freeRef();
      return result;
    }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList data) -> {
      final Result input = inObj[0];
      if (input.isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(data.stream().parallel().map(tensor -> {
          final int outputDim = tensor.length();
          final int inputDim = (1 + (int) Math.sqrt(1 + 8 * outputDim)) / 2;
          @Nonnull final Tensor passback = new Tensor(inputDim);
          @Nullable final double[] passbackData = passback.getData();
          @Nullable final double[] tensorData = tensor.getData();
          IntStream.range(0, inputDim).forEach(x -> {
            IntStream.range(x + 1, inputDim).forEach(y -> {
              passbackData[x] += tensorData[CrossDifferenceLayer.index(x, y, inputDim)];
              passbackData[y] += -tensorData[CrossDifferenceLayer.index(x, y, inputDim)];
            });
          });
          tensor.freeRef();
          return passback;
        }).toArray(i -> new Tensor[i]));
        input.accumulate(buffer, tensorArray);
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }
      
      @Override
      public boolean isAlive() {
        for (@Nonnull final Result element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
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
