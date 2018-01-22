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

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * Reduces each band of an image to its average (arithmetic mean) value.
 */
@SuppressWarnings("serial")
public class AvgImageBandLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AvgImageBandLayer.class);
  
  /**
   * Instantiates a new Avg image band layer.
   */
  public AvgImageBandLayer() {
    super();
  }
  
  /**
   * Instantiates a new Avg image band layer.
   *
   * @param id the id
   */
  protected AvgImageBandLayer(final JsonObject id) {
    super(id);
  }
  
  /**
   * From json avg image band layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the avg image band layer
   */
  public static AvgImageBandLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new AvgImageBandLayer(json);
  }
  
  @Override
  public NNResult eval(final NNResult... inObj) {
  
    assert 1 == inObj.length;
    final NNResult in = inObj[0];
    final TensorList inData = in.getData();
    final int[] inputDims = inData.get(0).getDimensions();
    assert 3 == inputDims.length;
  
    final Tensor[] results = inData.stream().map(data -> {
      final DoubleStream doubleStream = IntStream.range(0, inputDims[2]).parallel().mapToDouble(band -> {
        return data.coordStream(true).filter(e -> e.getCoords()[2] == band).mapToDouble(c -> data.get(c)).average().getAsDouble();
      });
      return new Tensor(1, 1, inputDims[2]).set(Tensor.getDoubles(doubleStream, inputDims[2]));
    }).toArray(i -> new Tensor[i]);
    
    return new NNResult(results) {
  
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(NNResult::free);
      }
  
      @Override
      protected void _accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        if (in.isAlive()) {
          in.accumulate(buffer, new TensorArray(IntStream.range(0, data.length()).parallel().mapToObj(dataIndex -> {
            final Tensor tensor = inData.get(dataIndex);
            final int[] inputDim = tensor.getDimensions();
            final Tensor backprop = data.get(dataIndex);
            return new Tensor(inputDim).mapCoords((c) -> {
              return backprop.get(0, 0, c.getCoords()[2]) / (inputDim[0] * inputDim[1]);
            });
          }).toArray(i -> new Tensor[i])));
        }
      }
      
      @Override
      public boolean isAlive() {
        return in.isAlive();
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
      if (this == obj) {
        return true;
      }
      if (obj == null) {
        return false;
      }
      if (getClass() != obj.getClass()) {
        return false;
      }
      final AvgImageBandLayer.IndexMapKey other = (AvgImageBandLayer.IndexMapKey) obj;
      if (!Arrays.equals(kernel, other.kernel)) {
        return false;
      }
      return Arrays.equals(output, other.output);
    }
    
    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + Arrays.hashCode(kernel);
      result = prime * result + Arrays.hashCode(output);
      return result;
    }
  }
}
