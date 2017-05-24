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

package com.simiacryptus.mindseye.layers;

import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import com.simiacryptus.util.Util;
import com.simiacryptus.util.ml.Tensor;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;
import java.util.UUID;

/**
 * Nonlinear Network Layer (aka Neural Network Layer)
 *
 * @author Andrew Charneski
 */
public abstract class NNLayer implements Serializable {
  
  /**
   *
   */
  private static final long serialVersionUID = 8741041477497062122L;
  public final String[] createdBy = Util.currentStack();
  public final UUID id = Util.uuid();
  private boolean frozen = false;
  
  @Override
  public boolean equals(final Object obj) {
    if (this == obj)
      return true;
    if (obj == null)
      return false;
    if (getClass() != obj.getClass())
      return false;
    final NNLayer other = (NNLayer) obj;
    if (this.id == null) {
      if (other.id != null)
        return false;
    } else if (!this.id.equals(other.id))
      return false;
    return true;
  }
  
  public final NNResult eval(final Tensor... array) {
    return eval(NNResult.singleResultArray(array));
  }
  
  public final NNResult eval(final Tensor[][] array) {
    return eval(NNResult.singleResultArray(array));
  }
  
  public abstract NNResult eval(NNResult... array);
  
  public NNLayer evolve() {
    return null;
  }
  
  public NNLayer freeze() {
    return setFrozen(true);
  }
  
  public NNLayer getChild(final UUID id) {
    if (this.id.equals(id))
      return this;
    return null;
  }
  
  public List<NNLayer> getChildren() {
    return Arrays.asList(this);
  }
  
  public final UUID getId() {
    return this.id;
  }
  
  public JsonObject getJson() {
    final JsonObject json = new JsonObject();
    json.addProperty("class", getClass().getSimpleName());
    json.addProperty("id", getId().toString());
    return json;
  }
  
  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + (this.id == null ? 0 : this.id.hashCode());
    return result;
  }
  
  public final boolean isFrozen() {
    return this.frozen;
  }
  
  public final NNLayer setFrozen(final boolean frozen) {
    this.frozen = frozen;
    return self();
  }
  
  @SuppressWarnings("unchecked")
  protected final NNLayer self() {
    return this;
  }
  
  public abstract List<double[]> state();
  
  @Override
  public final String toString() {
    return new GsonBuilder().setPrettyPrinting().create().toJson(getJson());
  }
  
  public static final class ConstNNResult extends NNResult {
    
    public ConstNNResult(final Tensor... data) {
      super(data);
    }
    
    @Override
    public void accumulate(final DeltaSet buffer, final Tensor[] data) {
      // Do Nothing
    }
    
    @Override
    public boolean isAlive() {
      return false;
    }
  }
  
}
