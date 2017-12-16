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

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import com.simiacryptus.util.lang.Tuple2;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.function.Function;
import java.util.function.IntToDoubleFunction;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Max subsample layer.
 */
public class MaxSubsampleLayer extends NNLayer {
  
  private static final Function<MaxSubsampleLayer.CalcRegionsParameter, List<Tuple2<Integer, int[]>>> calcRegionsCache = Util.cache(MaxSubsampleLayer::calcRegions);
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxSubsampleLayer.class);
  private int[] kernelDims;
  
  
  /**
   * Instantiates a new Max subsample layer.
   *
   * @param id         the id
   * @param kernelDims the kernel dims
   */
  protected MaxSubsampleLayer(JsonObject id, int... kernelDims) {
    super(id);
    this.kernelDims = Arrays.copyOf(kernelDims, kernelDims.length);
  }
  
  /**
   * Instantiates a new Max subsample layer.
   */
  protected MaxSubsampleLayer() {
    super();
  }
  
  /**
   * Instantiates a new Max subsample layer.
   *
   * @param kernelDims the kernel dims
   */
  public MaxSubsampleLayer(final int... kernelDims) {
    
    this.kernelDims = Arrays.copyOf(kernelDims, kernelDims.length);
  }
  
  /**
   * From json max subsample layer.
   *
   * @param json the json
   * @return the max subsample layer
   */
  public static MaxSubsampleLayer fromJson(JsonObject json) {
    return new MaxSubsampleLayer(json,
      JsonUtil.getIntArray(json.getAsJsonArray("inner")));
  }
  
  private static List<Tuple2<Integer, int[]>> calcRegions(final MaxSubsampleLayer.CalcRegionsParameter p) {
    final Tensor input = new Tensor(p.inputDims);
    final int[] newDims = IntStream.range(0, p.inputDims.length).map(i -> {
      //assert 0 == p.inputDims[i] % p.kernelDims[i];
      return (int) Math.ceil(p.inputDims[i] * 1.0 / p.kernelDims[i]);
    }).toArray();
    final Tensor output = new Tensor(newDims);
    
    return output.coordStream().map(o -> {
      final int[] inCoords = new Tensor(p.kernelDims).coordStream().mapToInt(kernelCoord -> {
        final int[] result = new int[o.getCoords().length];
        for (int index = 0; index < o.getCoords().length; index++) {
          int outputCoordinate = o.getCoords()[index];
          int kernelSize = p.kernelDims[index];
          int baseCoordinate = Math.min(outputCoordinate * kernelSize, p.inputDims[index] - kernelSize);
          int kernelCoordinate = kernelCoord.getCoords()[index];
          result[index] = baseCoordinate + kernelCoordinate;
        }
        return input.index(result);
      }).toArray();
      return new Tuple2<>(o.getIndex(), inCoords);
    }).collect(Collectors.toList());
  }
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("inner", JsonUtil.getJson(kernelDims));
    return json;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    
    final NNResult in = inObj[0];
    int itemCnt = in.getData().length();
    
    final int[] inputDims = in.getData().get(0).getDimensions();
    final List<Tuple2<Integer, int[]>> regions = calcRegionsCache.apply(new MaxSubsampleLayer.CalcRegionsParameter(inputDims, this.kernelDims));
    Tensor[] outputA = IntStream.range(0, in.getData().length()).mapToObj(dataIndex -> {
      final int[] newDims = IntStream.range(0, inputDims.length).map(i -> {
        return (int) Math.ceil(inputDims[i] * 1.0 / this.kernelDims[i]);
      }).toArray();
      final Tensor output = new Tensor(newDims);
      return output;
    }).toArray(i -> new Tensor[i]);
    int sum = Arrays.stream(outputA).mapToInt(x -> x.dim()).sum();
    @SuppressWarnings("unchecked") final int[][] gradientMapA = new int[in.getData().length()][];
    IntStream.range(0, in.getData().length()).forEach(dataIndex -> {
      final Tensor input = in.getData().get(dataIndex);
      final Tensor output = outputA[dataIndex];
      final IntToDoubleFunction keyExtractor = inputCoords -> input.get(inputCoords);
      int[] gradientMap = new int[input.dim()];
      regions.parallelStream().forEach(tuple -> {
        final Integer from = tuple.getFirst();
        int[] toList = tuple.getSecond();
        int toMax = -1;
        double bestValue = Double.NEGATIVE_INFINITY;
        for (int c : toList) {
          double value = keyExtractor.applyAsDouble(c);
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
    return new NNResult(outputA) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        if (in.isAlive()) {
          Tensor[] passbackA = IntStream.range(0, in.getData().length()).parallel().mapToObj(dataIndex -> {
            final Tensor backSignal = new Tensor(inputDims);
            int[] ints = gradientMapA[dataIndex];
            Tensor datum = data.get(dataIndex);
            for (int i = 0; i < datum.dim(); i++) {
              backSignal.add(ints[i], datum.get(i));
            }
            return backSignal;
          }).toArray(i -> new Tensor[i]);
          in.accumulate(buffer, new TensorArray(passbackA));
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
      final MaxSubsampleLayer.CalcRegionsParameter other = (MaxSubsampleLayer.CalcRegionsParameter) obj;
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
