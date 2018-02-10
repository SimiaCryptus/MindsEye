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
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.lang.Tuple2;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nullable;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.function.IntToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Max subsample layer.
 */
@SuppressWarnings("serial")
public class MaxPoolingLayer extends NNLayer {
  
  private static final Function<MaxPoolingLayer.CalcRegionsParameter, List<Tuple2<Integer, int[]>>> calcRegionsCache = Util.cache(MaxPoolingLayer::calcRegions);
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxPoolingLayer.class);
  private int[] kernelDims;
  
  
  /**
   * Instantiates a new Max subsample layer.
   */
  protected MaxPoolingLayer() {
    super();
  }
  
  /**
   * Instantiates a new Max subsample layer.
   *
   * @param kernelDims the kernel dims
   */
  public MaxPoolingLayer(@javax.annotation.Nonnull final int... kernelDims) {
    
    this.kernelDims = Arrays.copyOf(kernelDims, kernelDims.length);
  }
  
  /**
   * Instantiates a new Max subsample layer.
   *
   * @param id         the id
   * @param kernelDims the kernel dims
   */
  protected MaxPoolingLayer(@javax.annotation.Nonnull final JsonObject id, @javax.annotation.Nonnull final int... kernelDims) {
    super(id);
    this.kernelDims = Arrays.copyOf(kernelDims, kernelDims.length);
  }
  
  private static List<Tuple2<Integer, int[]>> calcRegions(@javax.annotation.Nonnull final MaxPoolingLayer.CalcRegionsParameter p) {
    @javax.annotation.Nonnull final Tensor input = new Tensor(p.inputDims);
    final int[] newDims = IntStream.range(0, p.inputDims.length).map(i -> {
      //assert 0 == p.inputDims[i] % p.kernelDims[i];
      return (int) Math.ceil(p.inputDims[i] * 1.0 / p.kernelDims[i]);
    }).toArray();
    @javax.annotation.Nonnull final Tensor output = new Tensor(newDims);
  
    return output.coordStream(true).map(o -> {
      final int[] inCoords = new Tensor(p.kernelDims).coordStream(true).mapToInt(kernelCoord -> {
        @javax.annotation.Nonnull final int[] result = new int[o.getCoords().length];
        for (int index = 0; index < o.getCoords().length; index++) {
          final int outputCoordinate = o.getCoords()[index];
          final int kernelSize = p.kernelDims[index];
          final int baseCoordinate = Math.min(outputCoordinate * kernelSize, p.inputDims[index] - kernelSize);
          final int kernelCoordinate = kernelCoord.getCoords()[index];
          result[index] = baseCoordinate + kernelCoordinate;
        }
        return input.index(result);
      }).toArray();
      return new Tuple2<>(o.getIndex(), inCoords);
    }).collect(Collectors.toList());
  }
  
  /**
   * From json max subsample layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the max subsample layer
   */
  public static MaxPoolingLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new MaxPoolingLayer(json,
      JsonUtil.getIntArray(json.getAsJsonArray("heapCopy")));
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNResult eval(@javax.annotation.Nonnull final NNResult... inObj) {
    
    Arrays.stream(inObj).forEach(nnResult -> nnResult.addRef());
    
    final NNResult in = inObj[0];
    in.getData().length();
    
    @javax.annotation.Nonnull final int[] inputDims = in.getData().get(0).getDimensions();
    final List<Tuple2<Integer, int[]>> regions = MaxPoolingLayer.calcRegionsCache.apply(new MaxPoolingLayer.CalcRegionsParameter(inputDims, kernelDims));
    final Tensor[] outputA = IntStream.range(0, in.getData().length()).mapToObj(dataIndex -> {
      final int[] newDims = IntStream.range(0, inputDims.length).map(i -> {
        return (int) Math.ceil(inputDims[i] * 1.0 / kernelDims[i]);
      }).toArray();
      @javax.annotation.Nonnull final Tensor output = new Tensor(newDims);
      return output;
    }).toArray(i -> new Tensor[i]);
    Arrays.stream(outputA).mapToInt(x -> x.dim()).sum();
    @javax.annotation.Nonnull final int[][] gradientMapA = new int[in.getData().length()][];
    IntStream.range(0, in.getData().length()).forEach(dataIndex -> {
      @javax.annotation.Nullable final Tensor input = in.getData().get(dataIndex);
      final Tensor output = outputA[dataIndex];
      @javax.annotation.Nonnull final IntToDoubleFunction keyExtractor = inputCoords -> input.get(inputCoords);
      @javax.annotation.Nonnull final int[] gradientMap = new int[input.dim()];
      regions.parallelStream().forEach(tuple -> {
        final Integer from = tuple.getFirst();
        final int[] toList = tuple.getSecond();
        int toMax = -1;
        double bestValue = Double.NEGATIVE_INFINITY;
        for (final int c : toList) {
          final double value = keyExtractor.applyAsDouble(c);
          if (-1 == toMax || bestValue < value) {
            bestValue = value;
            toMax = c;
          }
        }
        gradientMap[from] = toMax;
        output.set(from, input.get(toMax));
      });
      gradientMapA[dataIndex] = gradientMap;
    });
    return new NNResult(TensorArray.wrap(outputA), (@javax.annotation.Nonnull final DeltaSet<NNLayer> buffer, @javax.annotation.Nonnull final TensorList data) -> {
      if (in.isAlive()) {
        @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, in.getData().length()).parallel().mapToObj(dataIndex -> {
          @javax.annotation.Nonnull final Tensor backSignal = new Tensor(inputDims);
          final int[] ints = gradientMapA[dataIndex];
          @javax.annotation.Nullable final Tensor datum = data.get(dataIndex);
          for (int i = 0; i < datum.dim(); i++) {
            backSignal.add(ints[i], datum.get(i));
          }
          return backSignal;
        }).toArray(i -> new Tensor[i]));
        in.accumulate(buffer, tensorArray);
        tensorArray.freeRef();
      }
    }) {
      
      @Override
      protected void _free() {
        Arrays.stream(inObj).forEach(nnResult -> nnResult.freeRef());
      }
      
      @Override
      public boolean isAlive() {
        return in.isAlive();
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
      @javax.annotation.Nonnull final MaxPoolingLayer.CalcRegionsParameter other = (MaxPoolingLayer.CalcRegionsParameter) obj;
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
