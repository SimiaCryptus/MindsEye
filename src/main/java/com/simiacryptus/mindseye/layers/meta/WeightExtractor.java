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
import com.simiacryptus.mindseye.data.Tensor;
import com.simiacryptus.mindseye.data.TensorList;
import com.simiacryptus.mindseye.layers.DeltaSet;
import com.simiacryptus.mindseye.layers.NNLayer;
import com.simiacryptus.mindseye.layers.NNResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

/**
 * The type Weight extractor.
 */
@SuppressWarnings("serial")
public final class WeightExtractor extends NNLayer {
  
  public JsonObject getJson() {
    JsonObject json = super.getJsonStub();
    json.addProperty("innerId", getInner().getId().toString());
    json.addProperty("index", index);
    return json;
  }

  /**
   * From json weight extractor.
   *
   * @param json the json
   * @return the weight extractor
   */
  public static WeightExtractor fromJson(JsonObject json) {
    return new WeightExtractor(json);
  }

  /**
   * Instantiates a new Weight extractor.
   *
   * @param json the json
   */
  protected WeightExtractor(JsonObject json) {
    super(json);
    this.setInner(null);
    this.index = json.get("index").getAsInt();
    this.innerId = UUID.fromString(json.getAsJsonPrimitive("innerId").getAsString());
  }
  
  /**
   * The Log.
   */
  static final Logger log = LoggerFactory.getLogger(WeightExtractor.class);
  
  private NNLayer inner;
  private UUID innerId;
  private final int index;

  /**
   * Instantiates a new Weight extractor.
   *
   * @param index the index
   * @param inner the inner
   */
  public WeightExtractor(final int index, final NNLayer inner) {
    this.setInner(inner);
    this.index = index;
    this.innerId = inner.id;
  }

  @Override
  public NNResult eval(NNExecutionContext nncontext, final NNResult... inObj) {
    double[] doubles = getInner().state().get(index);
    return new NNResult(new Tensor(doubles)) {
      
      @Override
      public boolean isAlive() {
        return !isFrozen() && !inner.isFrozen();
      }
      
      @Override
      public void accumulate(DeltaSet buffer, TensorList data) {
        assert (data.length() == 1);
        if (!isFrozen() && !inner.isFrozen()) buffer.get(inner, doubles).accumulate(data.get(0).getData());
      }
    };
  }

  @Override
  public List<double[]> state() {
    return new ArrayList<>();
  }
  
  /**
   * Gets inner id.
   *
   * @return the inner id
   */
  public UUID getInnerId() {
    return innerId;
  }
  
  /**
   * Gets inner.
   *
   * @return the inner
   */
  public NNLayer getInner() {
    return inner;
  }
  
  /**
   * Sets inner.
   *
   * @param inner the inner
   */
  public void setInner(NNLayer inner) {
    this.inner = inner;
    this.innerId = null == inner ? null : inner.id;
  }
}
