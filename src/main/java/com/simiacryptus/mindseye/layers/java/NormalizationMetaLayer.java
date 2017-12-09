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
import com.simiacryptus.mindseye.lang.NNLayer;
import com.simiacryptus.mindseye.network.DAGNetwork;
import com.simiacryptus.mindseye.network.DAGNode;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.UUID;

/**
 * The type Normalization meta layer.
 */
@SuppressWarnings("serial")
public class NormalizationMetaLayer extends DAGNetwork {
  
  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(NormalizationMetaLayer.class);
  private final DAGNode head;
  
  /**
   * Instantiates a new Normalization meta layer.
   *
   * @param json the json
   */
  protected NormalizationMetaLayer(JsonObject json) {
    super(json);
    head = nodesById.get(UUID.fromString(json.getAsJsonPrimitive("head").getAsString()));
  }
  
  /**
   * Instantiates a new Normalization meta layer.
   */
  public NormalizationMetaLayer() {
    super(1);
    DAGNode input = getInput(0);
//    DAGNode mean = fn(
//      new AvgMetaLayer(),
//      input
//    );
//    DAGNode recentered = fn(new SumInputsLayer(),
//      input,
//      fn(
//        new LinearActivationLayer().setScale(-1).freeze(),
//        mean
//      )
//    );
//    DAGNode variance = fn(new AvgMetaLayer(),
//      fn(new AvgReducerLayer(),
//        fn(new SqActivationLayer(),
//          recentered
//        )
//      )
//    );
//    DAGNode rescaled = fn(new ProductLayer(),
//      recentered,
//      fn(new NthPowerActivationLayer().setPower(-0.5),
//        variance
//      )
//    );
//    DAGNode reoffset = fn(new SumInputsLayer(),
//      mean,
//      rescaled
//    );
//
    DAGNode rescaled1 = add(new ProductInputsLayer(),
      input,
      add(new NthPowerActivationLayer().setPower(-0.5),
        add(new AvgMetaLayer(),
          add(new AvgReducerLayer(),
            add(new SqActivationLayer(),
              input
            )
          )
        )
      )
    );
    
    this.head = rescaled1;
  }
  
  /**
   * From json nn layer.
   *
   * @param inner the inner
   * @return the nn layer
   */
  public static NNLayer fromJson(JsonObject inner) {
    return new NormalizationMetaLayer(inner);
  }
  
  @Override
  public DAGNode getHead() {
    return head;
  }
}
