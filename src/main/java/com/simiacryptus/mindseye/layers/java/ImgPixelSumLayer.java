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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * Scales the input using per-color-band coefficients
 */
@SuppressWarnings("serial")
public class ImgPixelSumLayer extends LayerBase {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgPixelSumLayer.class);
  
  /**
   * Instantiates a new Img band scale layer.
   */
  protected ImgPixelSumLayer() {
    super();
  }
  
  
  /**
   * Instantiates a new Img band scale layer.
   *
   * @param json the json
   */
  protected ImgPixelSumLayer(@Nonnull final JsonObject json) {
    super(json);
  }
  
  /**
   * From json img band scale layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img band scale layer
   */
  public static ImgPixelSumLayer fromJson(@Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new ImgPixelSumLayer(json);
  }
  
  @Nonnull
  @Override
  public Result evalAndFree(final Result... inObj) {
    assert 1 == inObj.length;
    return evalAndFree(inObj[0]);
  }
  
  /**
   * Eval nn result.
   *
   * @param input the input
   * @return the nn result
   */
  @Nonnull
  public Result evalAndFree(@Nonnull final Result input) {
    final TensorList inputData = input.getData();
    int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    return new Result(TensorArray.wrap(inputData.stream().map(tensor -> {
      Tensor result = new Tensor(inputDims[0], inputDims[1], 1).setByCoord(c -> {
        return IntStream.range(0, inputDims[2]).mapToDouble(i -> {
          int[] coords = c.getCoords();
          return tensor.get(coords[0], coords[1], i);
        }).sum();
      });
      tensor.freeRef();
      return result;
    }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      if (input.isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(delta.stream().map(deltaTensor -> {
          int[] deltaDims = deltaTensor.getDimensions();
          Tensor result = new Tensor(deltaDims[0], deltaDims[1], inputDims[2])
            .setByCoord(c -> {
              int[] coords = c.getCoords();
              return deltaTensor.get(coords[0], coords[1], 0);
            });
          deltaTensor.freeRef();
          return result;
        }).toArray(i -> new Tensor[i]));
        input.accumulate(buffer, tensorArray);
      }
    }) {
      
      @Override
      protected void _free() {
        inputData.freeRef();
        input.freeRef();
      }
      
      
      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    return json;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
