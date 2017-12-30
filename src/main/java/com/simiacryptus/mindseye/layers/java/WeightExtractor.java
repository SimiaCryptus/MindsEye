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
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

/**
 * This input-less layer emits the weights of the referenced layer. This can be used to cause targeted normalization
 * effects.
 */
@SuppressWarnings("serial")
public final class WeightExtractor extends NNLayer {
  
  /**
   * The Log.
   */
  static final Logger log = LoggerFactory.getLogger(WeightExtractor.class);
  private final int index;
  private NNLayer inner;
  private Object innerId;
  
  /**
   * Instantiates a new Weight extractor.
   *
   * @param index the index
   * @param inner the inner
   */
  public WeightExtractor(final int index, final NNLayer inner) {
    setInner(inner);
    this.index = index;
  }
  
  /**
   * Instantiates a new Weight extractor.
   *
   * @param json the json
   */
  protected WeightExtractor(final JsonObject json) {
    super(json);
    index = json.get("index").getAsInt();
    JsonPrimitive innerId = json.getAsJsonPrimitive("innerId");
    this.innerId = null == innerId ? null : innerId.getAsString();
  }
  
  /**
   * From json weight extractor.
   *
   * @param json the json
   * @return the weight extractor
   */
  public static WeightExtractor fromJson(final JsonObject json) {
    return new WeightExtractor(json);
  }
  
  @Override
  public NNResult eval(final NNExecutionContext nncontext, final NNResult... inObj) {
    final double[] doubles = null == getInner() ? new double[]{} : getInner().state().get(index);
    return new NNResult(new Tensor(doubles)) {
      
      @Override
      public void accumulate(final DeltaSet<NNLayer> buffer, final TensorList data) {
        assert data.length() == 1;
        if (!isFrozen() && !inner.isFrozen()) {
          final Delta<NNLayer> delta = buffer.get(inner, doubles);
          final Tensor tensor = data.get(0);
          final double[] tensorData = tensor.getData();
          delta.addInPlace(tensorData);
        }
      }
  
      @Override
      public boolean isAlive() {
        return !isFrozen() && !inner.isFrozen();
      }
    };
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
  public void setInner(final NNLayer inner) {
    this.inner = inner;
    innerId = null == inner ? null : inner.getId();
  }
  
  /**
   * Gets inner id.
   *
   * @return the inner id
   */
  public Object getInnerId() {
    return innerId;
  }
  
  @Override
  public JsonObject getJson() {
    final JsonObject json = super.getJsonStub();
    NNLayer inner = getInner();
    Object id = null == inner ? innerId : inner.getId();
    if (null != id) json.addProperty("innerId", id.toString());
    json.addProperty("index", index);
    return json;
  }
  
  @Override
  public List<double[]> state() {
    return new ArrayList<>();
  }
}
