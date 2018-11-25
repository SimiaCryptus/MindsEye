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
import com.simiacryptus.mindseye.network.DAGNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Map;

/**
 * The type Std dev meta key.
 */
@SuppressWarnings("serial")
public class AutoEntropyLayer extends PipelineNetwork {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(AutoEntropyLayer.class);

  /**
   * Instantiates a new Std dev meta key.
   */
  public AutoEntropyLayer() {
    super(1);
    DAGNode input = getInput(0);
    wrap(new EntropyLossLayer(), input, input).freeRef();
  }

  /**
   * Instantiates a new Std dev meta key.
   *
   * @param json the json
   * @param rs   the rs
   */
  protected AutoEntropyLayer(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    super(json, rs);
  }

  /**
   * From json std dev meta key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the std dev meta key
   */
  public static AutoEntropyLayer fromJson(final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AutoEntropyLayer(json, rs);
  }

}
