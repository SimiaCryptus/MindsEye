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
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The type Normalization meta layer.
 */
@SuppressWarnings("serial")
public class NormalizationMetaLayer extends PipelineNetwork {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(NormalizationMetaLayer.class);
  
  /**
   * Instantiates a new Normalization meta layer.
   */
  public NormalizationMetaLayer() {
    super(1);
    getInput(0);
    add(new SqActivationLayer());
    add(new AvgReducerLayer());
    add(new AvgMetaLayer());
    add(new NthPowerActivationLayer().setPower(-0.5));
    add(new ProductInputsLayer(), getHead(), getInput(0));
  }
  
  /**
   * Instantiates a new Normalization meta layer.
   *
   * @param json the json
   */
  protected NormalizationMetaLayer(final JsonObject json) {
    super(json);
  }
  
  /**
   * From json normalization meta layer.
   *
   * @param json the json
   * @return the normalization meta layer
   */
  public static NormalizationMetaLayer fromJson(final JsonObject json) {
    return new NormalizationMetaLayer(json);
  }
  
}
