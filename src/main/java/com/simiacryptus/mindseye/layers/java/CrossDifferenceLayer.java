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

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.IntStream;

/**
 * The type Cross difference layer.
 */
@SuppressWarnings("serial")
public class CrossDifferenceLayer extends NNLayer {
  
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
  protected CrossDifferenceLayer(final JsonObject id) {
    super(id);
  }
  
  /**
   * From json cross difference layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the cross difference layer
   */
  public static CrossDifferenceLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
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
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    assert 1 == inObj.length;
    return new NNResult(inObj[0].getData().stream().parallel().map(tensor -> {
      final int inputDim = tensor.dim();
      final int outputDim = (inputDim * inputDim - inputDim) / 2;
      final Tensor result = new Tensor(outputDim);
      final double[] inputData = tensor.getData();
      final double[] resultData = result.getData();
      IntStream.range(0, inputDim).forEach(x -> {
        IntStream.range(x + 1, inputDim).forEach(y -> {
          resultData[CrossDifferenceLayer.index(x, y, inputDim)] = inputData[x] - inputData[y];
        });
      });
      return result;
    }).toArray(i -> new Tensor[i])) {
  
      @Override
      public void free() {
        Arrays.stream(inObj).forEach(NNResult::free);
      }
  
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        final NNResult input = inObj[0];
        if (input.isAlive()) {
          input.accumulate(buffer, new TensorArray(data.stream().parallel().map(tensor -> {
            final int outputDim = tensor.dim();
            final int inputDim = (1 + (int) Math.sqrt(1 + 8 * outputDim)) / 2;
            final Tensor passback = new Tensor(inputDim);
            final double[] passbackData = passback.getData();
            final double[] tensorData = tensor.getData();
            IntStream.range(0, inputDim).forEach(x -> {
              IntStream.range(x + 1, inputDim).forEach(y -> {
                passbackData[x] += tensorData[CrossDifferenceLayer.index(x, y, inputDim)];
                passbackData[y] += -tensorData[CrossDifferenceLayer.index(x, y, inputDim)];
              });
            });
            return passback;
          }).toArray(i -> new Tensor[i])));
        }
      }
      
      @Override
      public boolean isAlive() {
        for (final NNResult element : inObj)
          if (element.isAlive()) {
            return true;
          }
        return false;
      }
      
    };
  }
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    return super.getJsonStub();
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
}
