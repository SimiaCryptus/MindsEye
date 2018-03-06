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

import com.google.common.cache.CacheBuilder;
import com.google.common.cache.CacheLoader;
import com.google.common.cache.LoadingCache;
import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.Coordinate;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.util.io.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ExecutionException;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * A local-pooling layer which sets all elements to the average value.
 */
@SuppressWarnings("serial")
public class AvgPoolingLayer extends LayerBase {
  
  /**
   * The constant indexMapCache.
   */
  public static final LoadingCache<AvgPoolingLayer.IndexMapKey, Map<Coordinate, List<int[]>>> indexMapCache = CacheBuilder.newBuilder()
    .build(new LayerCacheLoader());
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AvgPoolingLayer.class);
  private int[] kernelDims;
  
  
  /**
   * Instantiates a new Avg subsample layer.
   */
  protected AvgPoolingLayer() {
    super();
  }
  
  /**
   * Instantiates a new Avg subsample layer.
   *
   * @param kernelDims the kernel dims
   */
  public AvgPoolingLayer(@javax.annotation.Nonnull final int... kernelDims) {
    
    this.kernelDims = Arrays.copyOf(kernelDims, kernelDims.length);
  }
  
  /**
   * Instantiates a new Avg subsample layer.
   *
   * @param id         the id
   * @param kernelDims the kernel dims
   */
  protected AvgPoolingLayer(@javax.annotation.Nonnull final JsonObject id, @javax.annotation.Nonnull final int... kernelDims) {
    super(id);
    this.kernelDims = Arrays.copyOf(kernelDims, kernelDims.length);
  }
  
  /**
   * From json avg subsample layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the avg subsample layer
   */
  public static AvgPoolingLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new AvgPoolingLayer(json,
      JsonUtil.getIntArray(json.getAsJsonArray("heapCopy")));
  }
  
  private static synchronized Map<Coordinate, List<int[]>> getCoordMap(final int[] kernelDims, final int[] outDims) {
    try {
      return AvgPoolingLayer.indexMapCache.get(new AvgPoolingLayer.IndexMapKey(kernelDims, outDims));
    } catch (@javax.annotation.Nonnull final ExecutionException e) {
      throw new RuntimeException(e);
    }
  }
  
  @javax.annotation.Nonnull
  @SuppressWarnings("unchecked")
  @Override
  public Result eval(@javax.annotation.Nonnull final Result... inObj) {
    final int kernelSize = Tensor.length(kernelDims);
    final TensorList data = inObj[0].getData();
    @javax.annotation.Nonnull final int[] inputDims = data.getDimensions();
    final int[] newDims = IntStream.range(0, inputDims.length).map(i -> {
      assert 0 == inputDims[i] % kernelDims[i] : inputDims[i] + ":" + kernelDims[i];
      return inputDims[i] / kernelDims[i];
    }).toArray();
    final Map<Coordinate, List<int[]>> coordMap = AvgPoolingLayer.getCoordMap(kernelDims, newDims);
    final Tensor[] outputValues = IntStream.range(0, data.length()).mapToObj(dataIndex -> {
      @javax.annotation.Nullable final Tensor input = data.get(dataIndex);
      @javax.annotation.Nonnull final Tensor output = new Tensor(newDims);
      for (@javax.annotation.Nonnull final Entry<Coordinate, List<int[]>> entry : coordMap.entrySet()) {
        double sum = entry.getValue().stream().mapToDouble(inputCoord -> input.get(inputCoord)).sum();
        if (Double.isFinite(sum)) {
          output.add(entry.getKey(), sum / kernelSize);
        }
      }
      input.freeRef();
      return output;
    }).toArray(i -> new Tensor[i]);
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    return new Result(TensorArray.wrap(outputValues), (@javax.annotation.Nonnull final DeltaSet<Layer> buffer, @javax.annotation.Nonnull final TensorList delta) -> {
      if (inObj[0].isAlive()) {
        final Tensor[] passback = IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          @javax.annotation.Nullable Tensor tensor = delta.get(dataIndex);
          @javax.annotation.Nonnull final Tensor backSignal = new Tensor(inputDims);
          for (@javax.annotation.Nonnull final Entry<Coordinate, List<int[]>> outputMapping : coordMap.entrySet()) {
            final double outputValue = tensor.get(outputMapping.getKey());
            for (@javax.annotation.Nonnull final int[] inputCoord : outputMapping.getValue()) {
              backSignal.add(inputCoord, outputValue / kernelSize);
            }
          }
          tensor.freeRef();
          return backSignal;
        }).toArray(i -> new Tensor[i]);
        @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(passback);
        inObj[0].accumulate(buffer, tensorArray);
      }
    }) {
  
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
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
    json.add("heapCopy", JsonUtil.getJson(kernelDims));
    return json;
  }
  
  @javax.annotation.Nonnull
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
    public IndexMapKey(@javax.annotation.Nonnull final Tensor kernel, final Tensor input, @javax.annotation.Nonnull final Tensor output) {
      super();
      this.kernel = kernel.getDimensions();
      this.output = output.getDimensions();
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
      @Nullable final AvgPoolingLayer.IndexMapKey other = (AvgPoolingLayer.IndexMapKey) obj;
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
  
  private static class LayerCacheLoader extends CacheLoader<IndexMapKey, Map<Coordinate, List<int[]>>> {
    @Override
    public Map<Coordinate, List<int[]>> load(final IndexMapKey key) throws Exception {
      final int[] ksize = key.kernel;
      Tensor tensor = new Tensor(key.output);
      final Map<Coordinate, List<int[]>> coordMap = tensor.coordStream(true).collect(Collectors.toMap(o -> o, o -> {
        @Nonnull Tensor blank = new Tensor(ksize);
        List<int[]> collect = blank.coordStream(true).map(kernelCoord -> {
          int[] coords = o.getCoords();
          @Nonnull final int[] r = new int[coords.length];
          for (int i = 0; i < coords.length; i++) {
            r[i] = coords[i] * ksize[i] + kernelCoord.getCoords()[i];
          }
          return r;
        }).collect(Collectors.toList());
        blank.freeRef();
        return collect;
      }));
      tensor.freeRef();
      return coordMap;
    }
  }
}
