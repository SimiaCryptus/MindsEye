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

package com.simiacryptus.mindseye.layers.cudnn;

import com.google.gson.JsonObject;
import com.simiacryptus.mindseye.lang.DataSerializer;
import com.simiacryptus.mindseye.network.InnerNode;
import com.simiacryptus.mindseye.network.PipelineNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.annotation.Nonnull;
import java.util.Map;
import java.util.UUID;

/**
 * Implements the RMS loss key (without the final square root). Implemented as a sutnetwork.
 */
@SuppressWarnings("serial")
public class MeanSqLossLayer extends PipelineNetwork {

  @SuppressWarnings("unused")
  private static final Logger log = LoggerFactory.getLogger(MeanSqLossLayer.class);
  private final InnerNode binaryNode;
  private double alpha = 1.0;

  /**
   * Instantiates a new Mean sq loss key.
   */
  public MeanSqLossLayer() {
    super(2);
    this.binaryNode = wrap(new BinarySumLayer(alpha, -alpha), getInput(0), getInput(1));
    wrap(new SquareActivationLayer()).freeRef();
    wrap(new AvgReducerLayer()).freeRef();
  }

  /**
   * Instantiates a new Mean sq loss key.
   *
   * @param id the id
   * @param rs the rs
   */
  protected MeanSqLossLayer(@Nonnull final JsonObject id, Map<CharSequence, byte[]> rs) {
    super(id, rs);
    alpha = id.get("alpha").getAsDouble();
    binaryNode = (InnerNode) getNodeById(UUID.fromString(id.get("binaryNode").getAsString()));
  }

  /**
   * From json mean sq loss key.
   *
   * @param json the json
   * @param rs   the rs
   * @return the mean sq loss key
   */
  public static MeanSqLossLayer fromJson(final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new MeanSqLossLayer(json, rs);
  }

  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    JsonObject json = super.getJson(resources, dataSerializer);
    json.addProperty("alpha", alpha);
    json.addProperty("binaryNode", binaryNode.id.toString());
    return json;
  }

  /**
   * Gets alphaList.
   *
   * @return the alphaList
   */
  public double getAlpha() {
    return alpha;
  }

  /**
   * Sets alphaList.
   *
   * @param alpha the alphaList
   * @return the alphaList
   */
  public MeanSqLossLayer setAlpha(final double alpha) {
    this.alpha = alpha;
    BinarySumLayer binarySumLayer = binaryNode.getLayer();
    binarySumLayer.setLeftFactor(alpha);
    binarySumLayer.setRightFactor(-alpha);
    return this;
  }
}
