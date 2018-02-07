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
import com.simiacryptus.util.data.IntArray;
import com.simiacryptus.util.io.JsonUtil;
import org.jetbrains.annotations.Nullable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Selects the maximum value in each NxN cell, setting all other values to zero. This introduces sparsity into the
 * signal, but does not reduce resolution.
 */
@SuppressWarnings("serial")
public class MaxDropoutNoiseLayer extends NNLayer {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxDropoutNoiseLayer.class);
  private final @Nullable int[] kernelSize;
  private final Function<IntArray, List<List<Coordinate>>> getCellMap_cached = Util.cache(this::getCellMap);
  
  /**
   * Instantiates a new Max dropout noise layer.
   */
  public MaxDropoutNoiseLayer() {
    this(2, 2);
  }
  
  /**
   * Instantiates a new Max dropout noise layer.
   *
   * @param dims the dims
   */
  public MaxDropoutNoiseLayer(final int... dims) {
    super();
    kernelSize = dims;
  }
  
  /**
   * Instantiates a new Max dropout noise layer.
   *
   * @param json the json
   */
  protected MaxDropoutNoiseLayer(@javax.annotation.Nonnull final JsonObject json) {
    super(json);
    kernelSize = JsonUtil.getIntArray(json.getAsJsonArray("kernelSize"));
  }
  
  /**
   * From json max dropout noise layer.
   *
   * @param json the json
   * @param rs   the rs
   * @return the max dropout noise layer
   */
  public static MaxDropoutNoiseLayer fromJson(@javax.annotation.Nonnull final JsonObject json, Map<String, byte[]> rs) {
    return new MaxDropoutNoiseLayer(json);
  }
  
  @javax.annotation.Nonnull
  @Override
  public NNResult eval(final NNResult... inObj) {
    final int itemCnt = inObj[0].getData().length();
    inObj[0].addRef();
    final Tensor[] mask = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final Tensor input = inObj[0].getData().get(dataIndex);
      final @Nullable Tensor output = input.map(x -> 0);
      final List<List<Coordinate>> cells = getCellMap_cached.apply(new IntArray(output.getDimensions()));
      cells.forEach(cell -> {
        output.set(cell.stream().max(Comparator.comparingDouble(c -> input.get(c))).get(), 1);
      });
      return output;
    }).toArray(i -> new Tensor[i]);
    final Tensor[] outputA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final @Nullable double[] input = inObj[0].getData().get(dataIndex).getData();
      final @Nullable double[] maskT = mask[dataIndex].getData();
      @javax.annotation.Nonnull final Tensor output = new Tensor(inObj[0].getData().get(dataIndex).getDimensions());
      final @Nullable double[] outputData = output.getData();
      for (int i = 0; i < outputData.length; i++) {
        outputData[i] = input[i] * maskT[i];
      }
      return output;
    }).toArray(i -> new Tensor[i]);
    return new NNResult(TensorArray.wrap(outputA), (@javax.annotation.Nonnull final DeltaSet<NNLayer> buffer, @javax.annotation.Nonnull final TensorList delta) -> {
      if (inObj[0].isAlive()) {
        @javax.annotation.Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          final @Nullable double[] deltaData = delta.get(dataIndex).getData();
          @javax.annotation.Nonnull final int[] dims = inObj[0].getData().get(dataIndex).getDimensions();
          final @Nullable double[] maskData = mask[dataIndex].getData();
          @javax.annotation.Nonnull final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            passback.set(i, maskData[i] * deltaData[i]);
          }
          return passback;
        }).toArray(i -> new Tensor[i]));
        inObj[0].accumulate(buffer, tensorArray);
        tensorArray.freeRef();
      }
    }) {
      
      @Override
      protected void _free() {
        inObj[0].freeRef();
      }
      
      @Override
      public boolean isAlive() {
        return inObj[0].isAlive() || !isFrozen();
      }
      
    };
  }
  
  private List<List<Coordinate>> getCellMap(@javax.annotation.Nonnull final IntArray dims) {
    return new ArrayList<>(new Tensor(dims.data).coordStream(true).collect(Collectors.groupingBy((@javax.annotation.Nonnull final Coordinate c) -> {
      int cellId = 0;
      int max = 0;
      for (int dim = 0; dim < dims.size(); dim++) {
        final int pos = c.getCoords()[dim] / kernelSize[dim];
        cellId = cellId * max + pos;
        max = dims.get(dim) / kernelSize[dim];
      }
      return cellId;
    })).values());
  }
  
  @javax.annotation.Nonnull
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    @javax.annotation.Nonnull final JsonObject json = super.getJsonStub();
    json.add("kernelSize", JsonUtil.getJson(kernelSize));
    return json;
  }
  
  @javax.annotation.Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  
}
