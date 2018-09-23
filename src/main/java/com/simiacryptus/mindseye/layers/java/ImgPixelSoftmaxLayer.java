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
public class ImgPixelSoftmaxLayer extends LayerBase {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(ImgPixelSoftmaxLayer.class);

  /**
   * Instantiates a new Img band scale layer.
   */
  public ImgPixelSoftmaxLayer() {
    super();
  }


  /**
   * Instantiates a new Img band scale layer.
   *
   * @param json the json
   */
  protected ImgPixelSoftmaxLayer(@Nonnull final JsonObject json) {
    super(json);
  }

  /**
   * From json img band scale layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the img band scale layer
   */
  public static ImgPixelSoftmaxLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new ImgPixelSoftmaxLayer(json);
  }

  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    assert 1 == inObj.length;
    return eval(inObj[0]);
  }

  /**
   * Eval nn result.
   *
   * @param input the input
   * @return the nn result
   */
  @Nonnull
  public Result eval(@Nonnull final Result input) {
    final TensorList inputData = input.getData();
    inputData.addRef();
    input.addRef();
    int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    final int inputBands = inputDims[2];
    final int width = inputDims[0];
    final int height = inputDims[1];
    TensorArray maxima = TensorArray.wrap(inputData.stream().map(inputTensor -> {
      try {
        return new Tensor(width, height, 1).setByCoord(c -> {
          return IntStream.range(0, inputBands).mapToDouble(band -> {
            int[] coords = c.getCoords();
            return inputTensor.get(coords[0], coords[1], band);
          }).max().getAsDouble();
        });
      } finally {
        inputTensor.freeRef();
      }
    }).toArray(i -> new Tensor[i]));
    TensorArray exps = TensorArray.wrap(IntStream.range(0, inputData.length()).mapToObj(index -> {
      final Tensor inputTensor = inputData.get(index);
      Tensor maxTensor = maxima.get(index);
      try {
        return new Tensor(inputDims).setByCoord(c -> {
          int[] coords = c.getCoords();
          return Math.exp(inputTensor.get(c) - maxTensor.get(coords[0], coords[1], 0));
        });
      } finally {
        inputTensor.freeRef();
        maxTensor.freeRef();
      }
    }).toArray(i -> new Tensor[i]));
    maxima.freeRef();
    TensorArray sums = TensorArray.wrap(exps.stream().map(expTensor -> {
      try {
        return new Tensor(width, height, 1).setByCoord(c -> {
          return IntStream.range(0, inputBands).mapToDouble(band -> {
            int[] coords = c.getCoords();
            return expTensor.get(coords[0], coords[1], band);
          }).sum();
        });
      } finally {
        expTensor.freeRef();
      }
    }).toArray(i -> new Tensor[i]));
    TensorArray output = TensorArray.wrap(IntStream.range(0, inputData.length()).mapToObj(index -> {
      Tensor sumTensor = sums.get(index);
      Tensor expTensor = exps.get(index);
      try {
        return new Tensor(inputDims).setByCoord(c -> {
          int[] coords = c.getCoords();
          return (expTensor.get(c) / sumTensor.get(coords[0], coords[1], 0));
        });
      } finally {
        sumTensor.freeRef();
        expTensor.freeRef();
      }
    }).toArray(i -> new Tensor[i]));
    return new Result(output, (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      if (input.isAlive()) {

        TensorArray dots = TensorArray.wrap(IntStream.range(0, inputData.length()).mapToObj(index -> {
          final Tensor deltaTensor = delta.get(index);
          Tensor expTensor = exps.get(index);
          try {
            return new Tensor(width, height, 1).setByCoord(c -> {
              return IntStream.range(0, inputBands).mapToDouble(band -> {
                int[] coords = c.getCoords();
                return expTensor.get(coords[0], coords[1], band) * deltaTensor.get(coords[0], coords[1], band);
              }).sum();
            });
          } finally {
            expTensor.freeRef();
            deltaTensor.freeRef();
          }
        }).toArray(i -> new Tensor[i]));

        TensorArray passback = TensorArray.wrap(IntStream.range(0, inputData.length()).mapToObj(index -> {
          final Tensor deltaTensor = delta.get(index);
          final Tensor expTensor = exps.get(index);
          Tensor sumTensor = sums.get(index);
          Tensor dotTensor = dots.get(index);
          try {
            return new Tensor(inputDims).setByCoord(c -> {
              int[] coords = c.getCoords();
              double sum = sumTensor.get(coords[0], coords[1], 0);
              double dot = dotTensor.get(coords[0], coords[1], 0);
              double deltaValue = deltaTensor.get(c);
              double expValue = expTensor.get(c);
              return (sum * deltaValue - dot) * expValue / (sum * sum);
            });
          } finally {
            deltaTensor.freeRef();
            expTensor.freeRef();
            sumTensor.freeRef();
            dotTensor.freeRef();
          }
        }).toArray(i -> new Tensor[i]));

        input.accumulate(buffer, passback);
        dots.freeRef();
      }
    }) {

      @Override
      protected void _free() {
        inputData.freeRef();
        input.freeRef();
        sums.freeRef();
        exps.freeRef();
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
