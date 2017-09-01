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

package com.simiacryptus.mindseye.layers.media;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.data.Coordinate;
import com.simiacryptus.mindseye.data.Tensor;
import com.simiacryptus.mindseye.data.TensorArray;
import com.simiacryptus.mindseye.data.TensorList;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.io.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;

/**
 * The type Max image band layer.
 */
public class MaxImageBandLayer extends NNLayer {
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    return json;
  }
  
  /**
   * From json max image band layer.
   *
   * @param json the json
   * @return the max image band layer
   */
  public static MaxImageBandLayer fromJson(JsonObject json) {
    return new MaxImageBandLayer(json,
                                  JsonUtil.getIntArray(json.getAsJsonArray("inner")));
  }
  
  /**
   * Instantiates a new Max image band layer.
   *
   * @param id         the id
   * @param kernelDims the kernel dims
   */
  protected MaxImageBandLayer(JsonObject id, int... kernelDims) {
    super(id);
  }
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxImageBandLayer.class);
  
  /**
   * Instantiates a new Max image band layer.
   */
  public MaxImageBandLayer() {
    super();
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {

    assert (1 == inObj.length);
    final NNResult in = inObj[0];
    int itemCnt = in.getData().length();
    final int[] inputDims = in.getData().get(0).getDimensions();
    assert (3 == inputDims.length);

    Coordinate[][] maxCoords = in.getData().stream().map(data -> {
      return IntStream.range(0, inputDims[2]).mapToObj(band -> {
        return data.coordStream().filter(e -> e.coords[2] == band).max(Comparator.comparing(c -> data.get(c))).get();
      }).toArray(i -> new Coordinate[i]);
    }).toArray(i -> new Coordinate[i][]);

    Tensor[] results = IntStream.range(0, in.getData().length()).mapToObj(dataIndex -> {
      return new Tensor(1, 1, inputDims[2]).set(IntStream.range(0, inputDims[2]).mapToDouble(band -> {
        int[] maxCoord = maxCoords[dataIndex][band].coords;
        return in.getData().get(dataIndex).get(maxCoord[0], maxCoord[1], band);
      }).toArray());
    }).toArray(i -> new Tensor[i]);

    return new NNResult(results) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        if (in.isAlive()) {
          final Tensor[] data1 = IntStream.range(0, in.getData().length()).parallel().mapToObj(dataIndex -> {
            Tensor passback = new Tensor(in.getData().get(dataIndex).getDimensions());
            IntStream.range(0, inputDims[2]).forEach(b -> {
              int[] maxCoord = maxCoords[dataIndex][b].coords;
              passback.set(new int[]{maxCoord[0], maxCoord[1], b}, data.get(dataIndex).get(0, 0, b));
            });
            return passback;
          }).toArray(i -> new Tensor[i]);
          in.accumulate(buffer, new TensorArray(data1));
        }
      }
      
      @Override
      public boolean isAlive() {
        return in.isAlive();
      }
    };
  }
  
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
    public boolean equals(final Object obj) {
      if (this == obj) {
        return true;
      }
      if (obj == null) {
        return false;
      }
      if (getClass() != obj.getClass()) {
        return false;
      }
      final MaxImageBandLayer.CalcRegionsParameter other = (MaxImageBandLayer.CalcRegionsParameter) obj;
      if (!Arrays.equals(this.inputDims, other.inputDims)) {
        return false;
      }
      return Arrays.equals(this.kernelDims, other.kernelDims);
    }
    
    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + Arrays.hashCode(this.inputDims);
      result = prime * result + Arrays.hashCode(this.kernelDims);
      return result;
    }
    
  }
}
