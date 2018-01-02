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
  protected MaxDropoutNoiseLayer(final JsonObject json) {
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
  public static MaxDropoutNoiseLayer fromJson(final JsonObject json, Map<String, byte[]> rs) {
    return new MaxDropoutNoiseLayer(json);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    final int itemCnt = inObj[0].getData().length();
    final Tensor[] mask = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final Tensor input = inObj[0].getData().get(dataIndex);
      final Tensor output = input.map(x -> 0);
      final List<List<Coordinate>> cells = getCellMap_cached.apply(new IntArray(output.getDimensions()));
      cells.forEach(cell -> {
        output.set(cell.stream().max(Comparator.comparingDouble(c -> input.get(c))).get(), 1);
      });
      return output;
    }).toArray(i -> new Tensor[i]);
    final Tensor[] outputA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final double[] input = inObj[0].getData().get(dataIndex).getData();
      final double[] maskT = mask[dataIndex].getData();
      final Tensor output = new Tensor(inObj[0].getData().get(dataIndex).getDimensions());
      final double[] outputData = output.getData();
      for (int i = 0; i < outputData.length; i++) {
        outputData[i] = input[i] * maskT[i];
      }
      return output;
    }).toArray(i -> new Tensor[i]);
    return new Result(outputA, inObj[0], mask);
  }
  
  private List<List<Coordinate>> getCellMap(final IntArray dims) {
    return new ArrayList<>(new Tensor(dims.data).coordStream().collect(Collectors.groupingBy((final Coordinate c) -> {
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
  
  @Override
  public JsonObject getJson(Map<String, byte[]> resources, DataSerializer dataSerializer) {
    final JsonObject json = super.getJsonStub();
    json.add("kernelSize", JsonUtil.getJson(kernelSize));
    return json;
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  private final class Result extends NNResult {
  
    private final NNResult inObj;
    private final Tensor[] mask;
  
    private Result(final Tensor[] outputA, final NNResult inObj, final Tensor[] mask) {
      super(outputA);
      this.inObj = inObj;
      this.mask = mask;
    }
  
    @Override
    public void finalize() {
      inObj.finalize();
    }
    
    @Override
    public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList delta) {
      if (inObj.isAlive()) {
        final Tensor[] passbackA = IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          final double[] deltaData = delta.get(dataIndex).getData();
          final int[] dims = inObj.getData().get(dataIndex).getDimensions();
          final double[] maskData = mask[dataIndex].getData();
          final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            passback.set(i, maskData[i] * deltaData[i]);
          }
          return passback;
        }).toArray(i -> new Tensor[i]);
        inObj.accumulate(buffer, new TensorArray(passbackA));
      }
    }
    
    @Override
    public boolean isAlive() {
      return inObj.isAlive() || !isFrozen();
    }
    
  }
  
}
