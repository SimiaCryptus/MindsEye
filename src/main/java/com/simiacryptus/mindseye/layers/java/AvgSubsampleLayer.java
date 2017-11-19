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

package com.simiacryptus.mindseye.layers.java;

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.io.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Avg subsample layer.
 */
public class AvgSubsampleLayer extends NNLayer {
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("inner", JsonUtil.getJson(kernelDims));
    return json;
  }
  
  /**
   * From json avg subsample layer.
   *
   * @param json the json
   * @return the avg subsample layer
   */
  public static AvgSubsampleLayer fromJson(JsonObject json) {
    return new AvgSubsampleLayer(json,
                                  JsonUtil.getIntArray(json.getAsJsonArray("inner")));
  }
  
  /**
   * Instantiates a new Avg subsample layer.
   *
   * @param id         the id
   * @param kernelDims the kernel dims
   */
  protected AvgSubsampleLayer(JsonObject id, int... kernelDims) {
    super(id);
    this.kernelDims = Arrays.copyOf(kernelDims, kernelDims.length);
  }
  
  
  /**
   * The constant indexMapCache.
   */
  public static final LoadingCache<AvgSubsampleLayer.IndexMapKey, Map<Coordinate, List<int[]>>> indexMapCache = CacheBuilder.newBuilder()
                                                                                                                  .build(new CacheLoader<AvgSubsampleLayer.IndexMapKey, Map<Coordinate, List<int[]>>>() {
                                                                                                                    @Override
                                                                                                                    public Map<Coordinate, List<int[]>> load(final AvgSubsampleLayer.IndexMapKey key) throws Exception {
                                                                                                                      final int[] ksize = key.kernel;
                                                                                                                      final Map<Coordinate, List<int[]>> coordMap = new Tensor(key.output).coordStream().collect(Collectors.toMap(o -> o, o -> {
                                                                                                                        return new Tensor(ksize).coordStream().map(kernelCoord -> {
                                                                                                                          final int[] r = new int[o.coords.length];
                                                                                                                          for (int i = 0; i < o.coords.length; i++) {
                                                                                                                            r[i] = o.coords[i] * ksize[i] + kernelCoord.coords[i];
                                                                                                                          }
                                                                                                                          return r;
                                                                                                                        }).collect(Collectors.toList());
                                                                                                                      }));
                                                                                                                      return coordMap;
                                                                                                                    }
                                                                                                                  });
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AvgSubsampleLayer.class);
  /**
   *
   */
  private static final long serialVersionUID = 7441695931197085499L;
  private int[] kernelDims;
  
  /**
   * Instantiates a new Avg subsample layer.
   */
  protected AvgSubsampleLayer() {
    super();
  }
  
  /**
   * Instantiates a new Avg subsample layer.
   *
   * @param kernelDims the kernel dims
   */
  public AvgSubsampleLayer(final int... kernelDims) {
    
    this.kernelDims = Arrays.copyOf(kernelDims, kernelDims.length);
  }
  
  private static Map<Coordinate, List<int[]>> getCoordMap(final int[] kernelDims, final int[] outDims) {
    try {
      return indexMapCache.get(new AvgSubsampleLayer.IndexMapKey(kernelDims, outDims));
    } catch (final ExecutionException e) {
      throw new RuntimeException(e);
    }
  }
  
  @SuppressWarnings("unchecked")
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    final int kernelSize = new Tensor(this.kernelDims).dim();
    final int[] inputDims = inObj[0].getData().get(0).getDimensions();
    int itemCnt = inObj[0].getData().length();
    final Map<Coordinate, List<int[]>> coordMapA[] = new Map[itemCnt];
    Tensor[] outputA = IntStream.range(0, inObj[0].getData().length()).mapToObj(dataIndex -> {
      final Tensor input = inObj[0].getData().get(dataIndex);
      final int[] newDims = IntStream.range(0, inputDims.length).map(i -> {
        if (!(0 == inputDims[i] % this.kernelDims[i])) {
          assert (false) : inputDims[i] + ":" + this.kernelDims[i];
        }
        return inputDims[i] / this.kernelDims[i];
      }).toArray();
      final Tensor output = new Tensor(newDims);
      final Map<Coordinate, List<int[]>> coordMap = getCoordMap(this.kernelDims, output.getDimensions());
      for (final Entry<Coordinate, List<int[]>> outputMapping : coordMap.entrySet()) {
        double sum = 0;
        for (final int[] inputCoord : outputMapping.getValue()) {
          sum += input.get(inputCoord);
        }
        if (Double.isFinite(sum)) {
          output.add(outputMapping.getKey(), sum / kernelSize);
        }
      }
      coordMapA[dataIndex] = coordMap;
      return output;
    }).toArray(i -> new Tensor[i]);
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        if (inObj[0].isAlive()) {
          Tensor[] passbackA = IntStream.range(0, inObj[0].getData().length()).mapToObj(dataIndex -> {
            final Tensor backSignal = new Tensor(inputDims);
            for (final Entry<Coordinate, List<int[]>> outputMapping : coordMapA[dataIndex].entrySet()) {
              final double outputValue = data.get(dataIndex).get(outputMapping.getKey());
              for (final int[] inputCoord : outputMapping.getValue()) {
                backSignal.add(inputCoord, outputValue / kernelSize);
              }
            }
            return backSignal;
          }).toArray(i -> new Tensor[i]);
          inObj[0].accumulate(buffer, new TensorArray(passbackA));
        }
      }
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive();
      }
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  /**
   * The type Index mapCoords key.
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
     * Instantiates a new Index mapCoords key.
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
     * Instantiates a new Index mapCoords key.
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
      final AvgSubsampleLayer.IndexMapKey other = (AvgSubsampleLayer.IndexMapKey) obj;
      if (!Arrays.equals(this.kernel, other.kernel)) {
        return false;
      }
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
