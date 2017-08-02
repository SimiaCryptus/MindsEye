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

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.*;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.ml.Tensor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * The type Avg image band layer.
 */
public class AvgImageBandLayer extends NNLayer {
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    return json;
  }
  
  /**
   * From json avg image band layer.
   *
   * @param json the json
   * @return the avg image band layer
   */
  public static AvgImageBandLayer fromJson(JsonObject json) {
    JsonArray jsonArray = json.getAsJsonArray("inner");
    return new AvgImageBandLayer(json, JsonUtil.getIntArray(jsonArray));
  }

  /**
   * Instantiates a new Avg image band layer.
   *
   * @param id         the id
   * @param kernelDims the kernel dims
   */
  protected AvgImageBandLayer(JsonObject id, int... kernelDims) {
    super(id);
  }
  
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AvgImageBandLayer.class);
  
  /**
   * Instantiates a new Avg image band layer.
   */
  public AvgImageBandLayer() {
    super();
  }
  
  
  @SuppressWarnings("unchecked")
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
  
    assert(1 == inObj.length);
    final NNResult in = inObj[0];
    int itemCnt = in.data.length();
    final int[] inputDims = in.data.get(0).getDimensions();
    assert(3 == inputDims.length);
  
    Tensor[] results = in.data.stream().map(data -> {
      return new Tensor(1, 1, inputDims[2]).set(IntStream.range(0, inputDims[2]).mapToDouble(band -> {
        int pixels = data.getDimensions()[0] * data.getDimensions()[1];
        return data.coordStream().filter(e->e.coords[2]==band).mapToDouble(c -> data.get(c)).sum() / pixels;
      }).toArray());
    }).toArray(i -> new Tensor[i]);
  
    return new NNResult(results) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        if (in.isAlive()) {
          final Tensor[] data1 = IntStream.range(0, in.data.length()).parallel().mapToObj(dataIndex -> {
            return new Tensor(in.data.get(dataIndex).getDimensions()).map((v, c)->{
              int pixels = in.data.get(dataIndex).getDimensions()[0] * in.data.get(dataIndex).getDimensions()[1];
              return data.get(dataIndex).get(0,0,c.coords[2]) / pixels;
            });
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
   * The type Index map key.
   */
  public static final class IndexMapKey {
    /**
     * The Kernel.
     */
    int[] kernel;
    /**
     * The Output.
     */
    int[] output;

    /**
     * Instantiates a new Index map key.
     *
     * @param kernel the kernel
     * @param output the output
     */
    public IndexMapKey(final int[] kernel, final int[] output) {
      super();
      this.kernel = kernel;
      this.output = output;
    }

    /**
     * Instantiates a new Index map key.
     *
     * @param kernel the kernel
     * @param input  the input
     * @param output the output
     */
    public IndexMapKey(final Tensor kernel, final Tensor input, final Tensor output) {
      super();
      this.kernel = kernel.getDimensions();
      this.output = output.getDimensions();
    }
    
    @Override
    public boolean equals(final Object obj) {
      if (this == obj)
        return true;
      if (obj == null)
        return false;
      if (getClass() != obj.getClass())
        return false;
      final AvgImageBandLayer.IndexMapKey other = (AvgImageBandLayer.IndexMapKey) obj;
      if (!Arrays.equals(this.kernel, other.kernel))
        return false;
      return Arrays.equals(this.output, other.output);
    }
    
    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + Arrays.hashCode(this.kernel);
      result = prime * result + Arrays.hashCode(this.output);
      return result;
    }
  }
}
