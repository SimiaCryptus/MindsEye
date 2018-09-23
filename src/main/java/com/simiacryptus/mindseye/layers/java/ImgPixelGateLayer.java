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
public class ImgPixelGateLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgPixelGateLayer.class);

  /**
   * Instantiates a new Img band scale layer.
   */
  public ImgPixelGateLayer() {
    super();
  }


  /**
   * Instantiates a new Img band scale layer.
   *
   * @param json the json
   */
  protected ImgPixelGateLayer(@Nonnull final JsonObject json) {
    super(json);
  }

  /**
   * From json img band scale layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img band scale layer
   */
  public static ImgPixelGateLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgPixelGateLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    assert 2 == inObj.length;
    return eval(inObj[0], inObj[1]);
  }

  /**
   * Eval nn result.
   *
   * @param input the input
   * @param gate  the gate
   * @return the nn result
   */
  @Nonnull
  public Result eval(@Nonnull final Result input, @Nonnull final Result gate) {
    final TensorList inputData = input.getData();
    final TensorList gateData = gate.getData();
    inputData.addRef();
    input.addRef();
    gate.addRef();
    gateData.addRef();
    int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    return new Result(TensorArray.wrap(IntStream.range(0, inputData.length()).mapToObj(i -> {
      Tensor inputTensor = inputData.get(i);
      Tensor gateTensor = gateData.get(i);
      Tensor result = new Tensor(inputDims[0], inputDims[1], 1).setByCoord(c -> {
        return IntStream.range(0, inputDims[2]).mapToDouble(b -> {
          int[] coords = c.getCoords();
          return inputTensor.get(coords[0], coords[1], b) * gateTensor.get(coords[0], coords[1], 0);
        }).sum();
      });
      inputTensor.freeRef();
      gateTensor.freeRef();
      return result;
    }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      if (input.isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).mapToObj(i -> {
          Tensor deltaTensor = delta.get(i);
          Tensor gateTensor = gateData.get(i);
          Tensor result = new Tensor(input.getData().getDimensions())
              .setByCoord(c -> {
                int[] coords = c.getCoords();
                return deltaTensor.get(coords[0], coords[1], 0) * gateTensor.get(coords[0], coords[1], 0);
              });
          deltaTensor.freeRef();
          gateTensor.freeRef();
          return result;
        }).toArray(i -> new Tensor[i]));
        input.accumulate(buffer, tensorArray);
      }
      if (gate.isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).mapToObj(i -> {
          Tensor deltaTensor = delta.get(i);
          Tensor inputTensor = inputData.get(i);
          Tensor result = new Tensor(gateData.getDimensions())
              .setByCoord(c -> IntStream.range(0, inputDims[2]).mapToDouble(b -> {
                int[] coords = c.getCoords();
                return deltaTensor.get(coords[0], coords[1], 0) * inputTensor.get(coords[0], coords[1], b);
              }).sum());
          deltaTensor.freeRef();
          inputTensor.freeRef();
          return result;
        }).toArray(i -> new Tensor[i]));
        gate.accumulate(buffer, tensorArray);
      }
    }) {

      @Override
      protected void _free() {
        inputData.freeRef();
        input.freeRef();
        gate.freeRef();
        gateData.freeRef();
      }


      @Override
      public boolean isAlive() {
        return input.isAlive() || !isFrozen();
      }
    };
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    return json;
  }

  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
}
