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

package com.simiacryptus.mindseye.layers.meta;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;
import java.util.stream.IntStream;

/**
 * The type Sum meta layer.
 */
@SuppressWarnings("serial")
public class CumSumMetaLayer extends NNLayer implements CumSum {
  
  private Tensor accumulation;
  private Tensor priorAccumulation;
  private double carryOver = 0.5;
  private int carryoverDenominator = 1000;
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    if(null != accumulation) json.add("state", accumulation.getJson());
    return json;
  }

  /**
   * From json sum meta layer.
   *
   * @param json the json
   * @return the sum meta layer
   */
  public static CumSumMetaLayer fromJson(JsonObject json) {
    return new CumSumMetaLayer(json);
  }

  /**
   * Instantiates a new Sum meta layer.
   *
   * @param id the id
   */
  protected CumSumMetaLayer(JsonObject id) {
    super(id);
    accumulation = Tensor.fromJson(id.getAsJsonObject("state"));
  }
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(CumSumMetaLayer.class);
  
  /**
   * Instantiates a new Sum meta layer.
   */
  public CumSumMetaLayer() {
  }
  
  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    NNResult input = inObj[0];
    int itemCnt = input.getData().length();
    Tensor thisSum = input.getData().get(0).mapCoordsParallel((v, c) ->
                                                                   IntStream.range(0, itemCnt)
                                                                     .mapToDouble(dataIndex -> input.getData().get(dataIndex).get(c))
                                                                     .sum());
    Tensor prior;
    if(null != this.priorAccumulation && nncontext.staticEvaluation()) {
      prior = this.priorAccumulation;
    } else {
      prior = this.accumulation;
    }
    Tensor returnValue;
    if(null != prior) {
      returnValue = prior.scale(Math.pow(carryOver, itemCnt / carryoverDenominator)).add(thisSum);
    } else {
      returnValue = thisSum;
    }
    if(!isFrozen() && !nncontext.staticEvaluation()) {
      this.priorAccumulation = this.accumulation;
      this.accumulation = returnValue;
    }
    return new NNResult(returnValue) {
      @Override
      public void accumulate(final DeltaSet buffer, final TensorList data) {
        if (input.isAlive()) {
          Tensor delta = data.get(0);
          Tensor feedback[] = new Tensor[itemCnt];
          Arrays.parallelSetAll(feedback, i -> new Tensor(delta.getDimensions()));
          delta.mapCoordsParallel((rho, inputCoord) -> {
            for (int inputItem = 0; inputItem < itemCnt; inputItem++) {
              feedback[inputItem].add(inputCoord, rho);
            }
            return 0;
          });
          input.accumulate(buffer, new TensorArray(feedback));
        }
      }
      
      @Override
      public boolean isAlive() {
        return input.isAlive();
      }
      
    };
  }
  
  @Override
  public List<double[]> state() {
    return Arrays.asList();
  }
  
  @Override
  public double getCarryOver() {
    return carryOver;
  }
  
  @Override
  public CumSumMetaLayer setCarryOver(double carryOver) {
    this.carryOver = carryOver;
    return this;
  }
  
  @Override
  public int getCarryoverDenominator() {
    return carryoverDenominator;
  }
  
  @Override
  public CumSumMetaLayer setCarryoverDenominator(int carryoverDenominator) {
    this.carryoverDenominator = carryoverDenominator;
    return this;
  }
}
