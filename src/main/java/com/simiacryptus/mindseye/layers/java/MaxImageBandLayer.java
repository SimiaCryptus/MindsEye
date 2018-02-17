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
import com.simiacryptus.util.io.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * Selects the highest value in each color band, emitting a 1x1xN tensor.
 */
@SuppressWarnings("serial")
public class MaxImageBandLayer extends LayerBase {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxImageBandLayer.class);
  
  /**
   * Instantiates a new Max image band layer.
   */
  public MaxImageBandLayer() {
    super();
  }
  
  /**
   * Instantiates a new Max image band layer.
   *
   * @param id         the id
   * @param kernelDims the kernel dims
   */
  protected MaxImageBandLayer(@javax.annotation.Nonnull final JsonObject id, final int... kernelDims) {
    super(id);
  }
  
  /**
   * From json max image band layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the max image band layer
   */
  public static MaxImageBandLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new MaxImageBandLayer(json,
      JsonUtil.getIntArray(json.getAsJsonArray("heapCopy")));
  }
  
  @javax.annotation.Nonnull
  @Override
  public Result eval(@javax.annotation.Nonnull final Result... inObj) {
    
    assert 1 == inObj.length;
    final TensorList inputData = inObj[0].getData();
    inputData.addRef();
    inputData.length();
    @javax.annotation.Nonnull final int[] inputDims = inputData.getDimensions();
    assert 3 == inputDims.length;
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
  
    final Coordinate[][] maxCoords = inputData.stream().map(data -> {
      Coordinate[] coordinates = IntStream.range(0, inputDims[2]).mapToObj(band -> {
        return data.coordStream(true).filter(e -> e.getCoords()[2] == band).max(Comparator.comparing(c -> data.get(c))).get();
      }).toArray(i -> new Coordinate[i]);
      data.freeRef();
      return coordinates;
    }).toArray(i -> new Coordinate[i][]);
  
    return new Result(TensorArray.wrap(IntStream.range(0, inputData.length()).mapToObj(dataIndex -> {
      Tensor tensor = inputData.get(dataIndex);
      final DoubleStream doubleStream = IntStream.range(0, inputDims[2]).mapToDouble(band -> {
        final int[] maxCoord = maxCoords[dataIndex][band].getCoords();
        double v = tensor.get(maxCoord[0], maxCoord[1], band);
        return v;
      });
      Tensor tensor1 = new Tensor(1, 1, inputDims[2]).set(Tensor.getDoubles(doubleStream, inputDims[2]));
      tensor.freeRef();
      return tensor1;
    }).toArray(i -> new Tensor[i])), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList delta) -> {
      if (inObj[0].isAlive()) {
        @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).parallel().mapToObj(dataIndex -> {
          Tensor deltaTensor = delta.get(dataIndex);
          @javax.annotation.Nonnull final Tensor passback = new Tensor(inputData.getDimensions());
          IntStream.range(0, inputDims[2]).forEach(b -> {
            final int[] maxCoord = maxCoords[dataIndex][b].getCoords();
            passback.set(new int[]{maxCoord[0], maxCoord[1], b}, deltaTensor.get(0, 0, b));
          });
          deltaTensor.freeRef();
          return passback;
        }).toArray(i -> new Tensor[i]));
        inObj[0].accumulate(buffer, tensorArray);
        tensorArray.freeRef();
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
        inputData.freeRef();
      }
      
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
    };
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    return json;
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  /**
   * The type Calc regions parameter.
   */
  public static class CalcRegionsParameter {
    /**
     * The Input dims.
     */
    public int[] inputDims;
    /**
     * The Kernel dims.
     */
    public int[] kernelDims;
  
    /**
     * Instantiates a new Calc regions parameter.
     *
     * @param inputDims  the input dims
     * @param kernelDims the kernel dims
     */
    public CalcRegionsParameter(final int[] inputDims, final int[] kernelDims) {
      this.inputDims = inputDims;
      this.kernelDims = kernelDims;
    }
    
    @Override
    public boolean equals(@Nullable final Object obj) {
      if (this == obj) {
        return true;
      }
      if (obj == null) {
        return false;
      }
      if (getClass() != obj.getClass()) {
        return false;
      }
      @javax.annotation.Nonnull final MaxImageBandLayer.CalcRegionsParameter other = (MaxImageBandLayer.CalcRegionsParameter) obj;
      if (!Arrays.equals(inputDims, other.inputDims)) {
        return false;
      }
      return Arrays.equals(kernelDims, other.kernelDims);
    }
    
    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + Arrays.hashCode(inputDims);
      result = prime * result + Arrays.hashCode(kernelDims);
      return result;
    }
    
  }
}
