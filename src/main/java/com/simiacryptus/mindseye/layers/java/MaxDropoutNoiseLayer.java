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
import com.simiacryptus.mindseye.lang.Coordinate;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.lang.DeltaSet;
import com.simiacryptus.mindseye.lang.Layer;
import com.simiacryptus.mindseye.lang.LayerBase;
import com.simiacryptus.mindseye.lang.ReferenceCounting;
import com.simiacryptus.mindseye.lang.Result;
import com.simiacryptus.mindseye.lang.Tensor;
import com.simiacryptus.mindseye.lang.TensorArray;
import com.simiacryptus.mindseye.lang.TensorList;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.data.IntArray;
import com.simiacryptus.util.io.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import javax.annotation.Nullable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * Selects the maximum value in each NxN cell, setting all other values to zero. This introduces sparsity into the
 * signal, but does not reduce resolution.
 */
@SuppressWarnings("serial")
public class MaxDropoutNoiseLayer extends LayerBase {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxDropoutNoiseLayer.class);
  @Nullable
  private final int[] kernelSize;
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
  protected MaxDropoutNoiseLayer(@Nonnull final JsonObject json) {
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
  public static MaxDropoutNoiseLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MaxDropoutNoiseLayer(json);
  }
  
  @Nonnull
  @Override
  public Result eval(final Result... inObj) {
    final Result in0 = inObj[0];
    final TensorList data0 = in0.getData();
    final int itemCnt = data0.length();
    in0.addRef();
    data0.addRef();
    final Tensor[] mask = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      @Nullable final Tensor input = data0.get(dataIndex);
      @Nullable final Tensor output = input.map(x -> 0);
      final List<List<Coordinate>> cells = getCellMap_cached.apply(new IntArray(output.getDimensions()));
      cells.forEach(cell -> {
        output.set(cell.stream().max(Comparator.comparingDouble(c -> input.get(c))).get(), 1);
      });
      input.freeRef();
      return output;
    }).toArray(i -> new Tensor[i]);
    return new Result(TensorArray.wrap(IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      Tensor inputData = data0.get(dataIndex);
      @Nullable final double[] input = inputData.getData();
      @Nullable final double[] maskT = mask[dataIndex].getData();
      @Nonnull final Tensor output = new Tensor(inputData.getDimensions());
      @Nullable final double[] outputData = output.getData();
      for (int i = 0; i < outputData.length; i++) {
        outputData[i] = input[i] * maskT[i];
      }
      inputData.freeRef();
      return output;
    }).toArray(i -> new Tensor[i])), (@Nonnull final DeltaSet<Layer> buffer, @Nonnull final TensorList delta) -> {
      if (in0.isAlive()) {
        @Nonnull TensorArray tensorArray = TensorArray.wrap(IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          Tensor deltaTensor = delta.get(dataIndex);
          @Nullable final double[] deltaData = deltaTensor.getData();
          @Nonnull final int[] dims = data0.getDimensions();
          @Nullable final double[] maskData = mask[dataIndex].getData();
          @Nonnull final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.length(); i++) {
            passback.set(i, maskData[i] * deltaData[i]);
          }
          deltaTensor.freeRef();
          return passback;
        }).toArray(i -> new Tensor[i]));
        in0.accumulate(buffer, tensorArray);
      }
    }) {
      
      @Override
      protected void _free() {
        in0.freeRef();
        data0.freeRef();
        Arrays.stream(mask).forEach(ReferenceCounting::freeRef);
      }
      
      @Override
      public boolean isAlive() {
        return in0.isAlive() || !isFrozen();
      }
      
    };
  }
  
  private List<List<Coordinate>> getCellMap(@Nonnull final IntArray dims) {
    Tensor tensor = new Tensor(dims.data);
    ArrayList<List<Coordinate>> lists = new ArrayList<>(tensor.coordStream(true).collect(Collectors.groupingBy((@Nonnull final Coordinate c) -> {
      int cellId = 0;
      int max = 0;
      for (int dim = 0; dim < dims.size(); dim++) {
        final int pos = c.getCoords()[dim] / kernelSize[dim];
        cellId = cellId * max + pos;
        max = dims.get(dim) / kernelSize[dim];
      }
      return cellId;
    })).values());
    tensor.freeRef();
    return lists;
  }
  
  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    json.add("kernelSize", JsonUtil.getJson(kernelSize));
    return json;
  }
  
  @Nonnull
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  
}
