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

package com.simiacryptus.mindseye.layers.cudnn;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.layers.java.AvgReducerLayer;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The type Mean sq loss layer.
 */
public class MeanSqLossLayer extends PipelineNetwork {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MeanSqLossLayer.class);
  
  /**
   * Instantiates a new Mean sq loss layer.
   *
   * @param id the id
   */
  protected MeanSqLossLayer(JsonObject id) {
    super(id);
  }
  
  /**
   * Instantiates a new Mean sq loss layer.
   */
  public MeanSqLossLayer() {
    super(2);
    add(new BinarySumLayer(1.0, -1.0), getInput(0), getInput(1));
    add(new ProductLayer(), getHead(), getHead());
    add(new BandReducerLayer().setMode(PoolingLayer.PoolingMode.Avg));
    add(new AvgReducerLayer());
  }
  
  /**
   * From json mean sq loss layer.
   *
   * @param json the json
   * @return the mean sq loss layer
   */
  public static MeanSqLossLayer fromJson(JsonObject json) {
    return new MeanSqLossLayer(json);
  }
  
  public JsonObject getJson() {
    return super.getJson();
  }
  
}
