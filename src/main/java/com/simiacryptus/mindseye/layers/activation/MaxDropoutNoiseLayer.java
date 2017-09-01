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

package com.simiacryptus.mindseye.layers.activation;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.data.Coordinate;
import com.simiacryptus.mindseye.data.Tensor;
import com.simiacryptus.mindseye.data.TensorArray;
import com.simiacryptus.mindseye.data.TensorList;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import com.simiacryptus.util.IntArray;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.io.JsonUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * The type Max dropout noise layer.
 */
public class MaxDropoutNoiseLayer extends NNLayer {
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.add("kernelSize", JsonUtil.getJson(kernelSize));
    return json;
  }
  
  /**
   * From json max dropout noise layer.
   *
   * @param json the json
   * @return the max dropout noise layer
   */
  public static MaxDropoutNoiseLayer fromJson(JsonObject json) {
    return new MaxDropoutNoiseLayer(json);
  }

  /**
   * Instantiates a new Max dropout noise layer.
   *
   * @param json the json
   */
  protected MaxDropoutNoiseLayer(JsonObject json) {
    super(json);
    this.kernelSize = JsonUtil.getIntArray(json.getAsJsonArray("kernelSize"));
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MaxDropoutNoiseLayer.class);
  private static final long serialVersionUID = -2105152439043901220L;
  private final int[] kernelSize;
  
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
  public MaxDropoutNoiseLayer(int... dims) {
    super();
    this.kernelSize = dims;
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    int itemCnt = inObj[0].getData().length();
    Tensor[] mask = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final Tensor input = inObj[0].getData().get(dataIndex);
      final Tensor output = input.map(x -> 0);
      final List<List<Coordinate>> cells = getCellMap_cached.apply(new IntArray(output.getDimensions()));
      cells.forEach(cell -> {
        output.set(cell.stream().max(Comparator.comparingDouble(c -> input.get(c))).get(), 1);
      });
      return output;
    }).toArray(i -> new Tensor[i]);
    Tensor[] outputA = IntStream.range(0, itemCnt).mapToObj(dataIndex -> {
      final double[] input = inObj[0].getData().get(dataIndex).getData();
      final double[] maskT = mask[dataIndex].getData();
      final Tensor output = new Tensor(inObj[0].getData().get(dataIndex).getDimensions());
      double[] outputData = output.getData();
      for (int i = 0; i < outputData.length; i++) {
        outputData[i] = input[i] * maskT[i];
      }
      return output;
    }).toArray(i -> new Tensor[i]);
    return new Result(outputA, inObj[0], mask);
  }
  
  private List<List<Coordinate>> getCellMap(IntArray dims) {
    return new ArrayList<>(new Tensor(dims.data).coordStream().collect(Collectors.groupingBy((Coordinate c) -> {
      int cellId = 0;
      int max = 0;
      for (int dim = 0; dim < dims.size(); dim++) {
        int pos = c.coords[dim] / kernelSize[dim];
        cellId = cellId * max + pos;
        max = dims.get(dim) / kernelSize[dim];
      }
      return cellId;
    })).values());
  }

  private final Function<IntArray, List<List<Coordinate>>> getCellMap_cached = Util.cache(this::getCellMap);
  
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  private final class Result extends NNResult {
    private final NNResult inObj;
    private final Tensor[] mask;
    
    private Result(final Tensor[] outputA, final NNResult inObj, Tensor[] mask) {
      super(outputA);
      this.inObj = inObj;
      this.mask = mask;
    }
    
    @Override
    public void accumulate(final DeltaSet buffer, final TensorList delta) {
      if (this.inObj.isAlive()) {
        Tensor[] passbackA = IntStream.range(0, delta.length()).mapToObj(dataIndex -> {
          final double[] deltaData = delta.get(dataIndex).getData();
          final int[] dims = this.inObj.getData().get(dataIndex).getDimensions();
          double[] maskData = mask[dataIndex].getData();
          final Tensor passback = new Tensor(dims);
          for (int i = 0; i < passback.dim(); i++) {
            passback.set(i, maskData[i] * deltaData[i]);
          }
          return passback;
        }).toArray(i -> new Tensor[i]);
        this.inObj.accumulate(buffer, new TensorArray(passbackA));
      }
    }
    
    @Override
    public boolean isAlive() {
      return this.inObj.isAlive() || !isFrozen();
    }
    
  }
  
}
