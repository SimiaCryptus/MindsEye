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
import com.simiacryptus.mindseye.data.Tensor;
import com.simiacryptus.mindseye.data.TensorList;
import com.simiacryptus.util.Util;

import java.io.Serializable;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
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
   * The interface Nn execution context.
   */
  public interface NNExecutionContext {
  }
  
  /**
   * The Id.
   */
  public final UUID id;
  private boolean frozen = false;
  private String name;
  
  /**
   * Instantiates a new Nn layer.
   *
   * @param json the json
   */
  protected NNLayer(JsonObject json) {
    if (!getClass().getCanonicalName().equals(json.get("class").getAsString())) {
      throw new IllegalArgumentException(getClass().getCanonicalName() + " != " + json.get("class").getAsString());
    }
    this.id = UUID.fromString(json.get("id").getAsString());
    if (json.has("isFrozen")) setFrozen(json.get("isFrozen").getAsBoolean());
    if (json.has("name")) setName(json.get("name").getAsString());
  }
  
  /**
   * Instantiates a new Nn layer.
   */
  protected NNLayer() {
    this.id = Util.uuid();
    this.name = getClass().getSimpleName() + "/" + id;
  }
  
  @Override
  public boolean equals(final Object obj) {
    if (this == obj) {
      return true;
    }
    if (obj == null) {
      return false;
    }
    if (getClass() != obj.getClass()) {
      return false;
    }
    final NNLayer other = (NNLayer) obj;
    if (this.id == null) {
      if (other.id != null) {
        return false;
      }
    }
    else if (!this.id.equals(other.id)) {
      return false;
    }
    return true;
  }
  
  /**
   * Eval nn result.
   *
   * @param nncontext the nncontext
   * @param array     the array
   * @return the nn result
   */
  public final NNResult eval(NNExecutionContext nncontext, final Tensor... array) {
    return eval(nncontext, NNResult.singleResultArray(array));
  }
  
  /**
   * Eval nn result.
   *
   * @param nncontext the nncontext
   * @param array     the array
   * @return the nn result
   */
  public final NNResult eval(NNExecutionContext nncontext, final Tensor[][] array) {
    return eval(nncontext, NNResult.singleResultArray(array));
  }
  
  /**
   * Eval nn result.
   *
   * @param nncontext the nncontext
   * @param array     the array
   * @return the nn result
   */
  public abstract NNResult eval(NNExecutionContext nncontext, NNResult[] array);
  
  /**
   * Freeze nn layer.
   *
   * @return the nn layer
   */
  public NNLayer freeze() {
    return setFrozen(true);
  }
  
  /**
   * Gets child.
   *
   * @param id the id
   * @return the child
   */
  public NNLayer getChild(final UUID id) {
    if (this.id.equals(id)) {
      return this;
    }
    return null;
  }
  
  /**
   * Gets children.
   *
   * @return the children
   */
  public List<NNLayer> getChildren() {
    return Arrays.asList(this);
  }
  
  /**
   * Gets id.
   *
   * @return the id
   */
  public final UUID getId() {
    return this.id;
  }
  
  /**
   * From json nn layer.
   *
   * @param inner the inner
   * @return the nn layer
   */
  public static NNLayer fromJson(JsonObject inner) {
    String className = inner.get("class").getAsString();
    try {
      Class<?> clazz = Class.forName(className);
      if (null == clazz) throw new ClassNotFoundException(className);
      Method method = clazz.getMethod("fromJson", JsonObject.class);
      if (method.getDeclaringClass() == NNLayer.class) throw new IllegalArgumentException("Cannot find deserialization method for " + className);
      return (NNLayer) method.invoke(null, inner);
    } catch (IllegalAccessException | InvocationTargetException | NoSuchMethodException | ClassNotFoundException e) {
      throw new RuntimeException(e);
    }
  }
  
  /**
   * Gets json string.
   *
   * @return the json string
   */
  public String getJsonString() {
    return new GsonBuilder().setPrettyPrinting().create().toJson(getJson());
  }
  
  /**
   * Gets json.
   *
   * @return the json
   */
  public abstract JsonObject getJson();
  
  /**
   * Gets json stub.
   *
   * @return the json stub
   */
  public JsonObject getJsonStub() {
    final JsonObject json = new JsonObject();
    json.addProperty("class", getClass().getCanonicalName());
    json.addProperty("id", getId().toString());
    json.addProperty("isFrozen", isFrozen());
    json.addProperty("name", getName());
    return json;
  }
  
  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + (this.id == null ? 0 : this.id.hashCode());
    return result;
  }
  
  /**
   * Is frozen boolean.
   *
   * @return the boolean
   */
  public boolean isFrozen() {
    return this.frozen;
  }
  
  /**
   * Sets frozen.
   *
   * @param frozen the frozen
   * @return the frozen
   */
  public NNLayer setFrozen(final boolean frozen) {
    this.frozen = frozen;
    return self();
  }
  
  /**
   * Self nn layer.
   *
   * @return the nn layer
   */
  @SuppressWarnings("unchecked")
  protected final NNLayer self() {
    return this;
  }
  
  /**
   * State list.
   *
   * @return the list
   */
  public abstract List<double[]> state();
  
  @Override
  public final String toString() {
    return getName();
  }
  
  /**
   * Gets name.
   *
   * @return the name
   */
  public String getName() {
    return name;
  }
  
  /**
   * Sets name.
   *
   * @param name the name
   * @return the name
   */
  public NNLayer setName(String name) {
    this.name = name;
    return this;
  }
  
  /**
   * The type Const nn result.
   */
  public static final class ConstNNResult extends NNResult {
    
    /**
     * Instantiates a new Const nn result.
     *
     * @param data the data
     */
    public ConstNNResult(final Tensor... data) {
      super(data);
    }
    
    @Override
    public void accumulate(final DeltaSet buffer, final TensorList data) {
      // Do Nothing
    }
    
    @Override
    public boolean isAlive() {
      return false;
    }
  }
  
}
